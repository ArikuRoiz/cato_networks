"""Postgres checkpointer setup for the LangGraph pipeline.

The checkpointer persists graph state between node executions, enabling:
  - Durable HITL: the graph can be interrupted and resumed across process restarts.
  - Crash recovery: in-progress cycles can resume from the last checkpoint.

``langgraph-checkpoint-postgres`` requires ``psycopg`` (v3) in autocommit mode.
The ``setup()`` call is idempotent — it creates the checkpoint tables if they
do not already exist, making it safe to call on every startup.

Connection ownership:
    The caller owns the ``psycopg.Connection`` passed to
    :func:`setup_checkpointer`.  Use the helper :func:`open_connection` (or
    ``psycopg.connect`` directly) in a ``with`` block to guarantee the
    connection is closed when the graph is no longer needed:

        with open_connection(db_url) as conn:
            saver = setup_checkpointer(conn)
            graph = build_graph(saver)
            # … use graph …
        # conn is closed here
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import psycopg
import psycopg.rows
from langgraph.checkpoint.postgres import PostgresSaver

# SQLAlchemy dialect prefixes that psycopg3 does not understand.  We strip them
# so callers can pass any of the common URL forms produced by SQLAlchemy /
# testcontainers without ceremony.
_SQLALCHEMY_PREFIXES: tuple[str, ...] = (
    "postgresql+psycopg2://",
    "postgresql+psycopg://",
    "postgres+psycopg2://",
    "postgres+psycopg://",
)
_LIBPQ_SCHEME = "postgresql://"


def _normalise_database_url(url: str) -> str:
    """Strip SQLAlchemy dialect prefixes so psycopg3 can parse the URL.

    psycopg3 accepts ``postgresql://`` (libpq format) and DSN key=value strings;
    it rejects ``postgresql+psycopg://`` style URIs used by SQLAlchemy drivers.
    This helper is idempotent: plain libpq URLs are returned unchanged.
    """
    for prefix in _SQLALCHEMY_PREFIXES:
        if url.startswith(prefix):
            return _LIBPQ_SCHEME + url[len(prefix) :]
    return url


@contextmanager
def open_connection(
    database_url: str,
) -> Generator[psycopg.Connection[psycopg.rows.DictRow], None, None]:
    """Context manager that opens and closes a :class:`psycopg.Connection`.

    The connection is configured as required by ``PostgresSaver``:
      - ``autocommit=True``        — PostgresSaver issues its own transaction
                                     control; autocommit prevents conflicts.
      - ``prepare_threshold=0``    — disables prepared statements, which are
                                     incompatible with most connection poolers
                                     (PgBouncer in transaction mode, etc.).
      - ``row_factory=dict_row``   — required by the ``Connection[DictRow]``
                                     type alias used internally by
                                     ``langgraph-checkpoint-postgres``.

    Accepts both plain libpq URLs and SQLAlchemy-style dialect URLs; see
    :func:`_normalise_database_url` for details.

    Usage::

        with open_connection(db_url) as conn:
            saver = setup_checkpointer(conn)
            graph = build_graph(saver)
    """
    libpq_url = _normalise_database_url(database_url)
    conn: psycopg.Connection[psycopg.rows.DictRow] = psycopg.connect(  # pyright: ignore[reportArgumentType]
        libpq_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=psycopg.rows.dict_row,
    )
    try:
        yield conn
    finally:
        conn.close()


def setup_checkpointer(
    conn: psycopg.Connection[psycopg.rows.DictRow],
) -> PostgresSaver:
    """Wrap a caller-managed connection in a ready-to-use :class:`PostgresSaver`.

    The caller owns the connection lifetime.  Use :func:`open_connection` as a
    context manager to ensure the connection is closed when the graph exits:

        with open_connection(db_url) as conn:
            saver = setup_checkpointer(conn)
            graph = build_graph(saver)

    Args:
        conn: An open ``psycopg`` connection configured with
            ``autocommit=True``, ``prepare_threshold=0``, and
            ``row_factory=dict_row``.  See :func:`open_connection`.

    Returns:
        A fully initialised :class:`PostgresSaver` ready to be passed to
        ``build_graph(checkpointer=...)``.
    """
    saver = PostgresSaver(conn)
    saver.setup()
    return saver
