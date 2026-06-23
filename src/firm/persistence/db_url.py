"""Database-URL normalisation for SQLAlchemy.

SQLAlchemy maps a bare ``postgresql://`` URL to the **psycopg2** driver, which
this project does not install — it uses **psycopg v3**. Every SQLAlchemy engine
(the ledger engine and Alembic) must therefore use the ``postgresql+psycopg://``
dialect. ``to_sqlalchemy_url`` forces that, idempotently.

(The psycopg-native connection path used by the LangGraph checkpointer and the
pgvector store goes the other way — see
``firm.orchestration.checkpointer._normalise_database_url``.)
"""

from __future__ import annotations

_PSYCOPG_DIALECT = "postgresql+psycopg://"

_KNOWN_PREFIXES = (
    "postgresql+psycopg2://",
    "postgresql+psycopg://",
    "postgresql://",
    "postgres+psycopg2://",
    "postgres+psycopg://",
    "postgres://",
)


def to_sqlalchemy_url(url: str) -> str:
    """Return *url* rewritten to use the psycopg v3 SQLAlchemy dialect.

    ``postgresql://…`` / ``postgresql+psycopg2://…`` → ``postgresql+psycopg://…``.
    Already-correct URLs are returned unchanged; unknown schemes pass through.
    """
    for prefix in _KNOWN_PREFIXES:
        if url.startswith(prefix):
            return f"{_PSYCOPG_DIALECT}{url[len(prefix):]}"
    return url
