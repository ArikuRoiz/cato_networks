"""Integration tests for the RAG / EvidenceStore stack.

Uses testcontainers to spin up Postgres + pgvector, then exercises
PgvectorEvidenceStore against acceptance-criteria scenarios from FIRM-9:

  test_no_lookahead            — retrieval never returns a future doc.
  test_insufficient_evidence_refuses — empty store → cite([]) → Insufficient.

These tests remove the xfail stubs from tests/integration/test_mandatory.py.

Requires: testcontainers[postgres], psycopg[binary], pgvector, sentence-transformers.
The sentence-transformers model (all-MiniLM-L6-v2) must be downloadable or
already cached.  If the model cannot be loaded the entire module is skipped
with a clear reason rather than failing with an obscure import error.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
from firm.ports.types import NewsDoc
from firm.rag.citation import Insufficient, cite

# ---------------------------------------------------------------------------
# Guard: verify the embedding model can actually be loaded before spending
# time spinning up a Postgres container.  Skip the whole module cleanly if
# the model weights are unavailable (e.g. air-gapped CI without a cache).
# ---------------------------------------------------------------------------

_EMBEDDER_SKIP_REASON: str | None = None

try:
    from firm.adapters.embeddings import SentenceTransformerEmbedder as _SE

    _SE().embed("warmup")  # triggers lazy load; raises if weights missing
except Exception as _exc:
    _EMBEDDER_SKIP_REASON = f"sentence-transformers model unavailable: {_exc}"

pytestmark = pytest.mark.skipif(
    _EMBEDDER_SKIP_REASON is not None,
    reason=_EMBEDDER_SKIP_REASON or "",
)

# ---------------------------------------------------------------------------
# Session-scoped Postgres + pgvector container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_container():
    """Start a pgvector-enabled Postgres container for the session."""
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_conn(pg_container: PostgresContainer) -> psycopg.Connection[Any]:
    """Open a psycopg3 connection to the test container and install the schema.

    The schema is bootstrapped once for the session.  ``migrate()`` calls
    ``register_vector`` after ``CREATE EXTENSION IF NOT EXISTS vector`` so the
    OID lookup succeeds.
    """
    raw_url = pg_container.get_connection_url()
    # testcontainers may return a psycopg2-style URL; normalise to plain postgres
    dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
    conn = psycopg.connect(dsn, autocommit=False)
    # Bootstrap the schema once for the whole session.
    # migrate() installs the extension then calls register_vector.
    store = PgvectorEvidenceStore(conn)
    store.migrate()
    return conn


# ---------------------------------------------------------------------------
# Per-test store with a clean slate
# ---------------------------------------------------------------------------


@pytest.fixture
def store(pg_conn: psycopg.Connection[Any]) -> Generator[PgvectorEvidenceStore, None, None]:
    """Return a fresh PgvectorEvidenceStore with a clean table for each test.

    Isolation strategy:
    - TRUNCATE + commit before yielding guarantees a clean slate.
    - Tests that call ``conn.commit()`` after ``embed_and_store`` verify
      that rows are durably written (visible to a fresh read after commit).
    - On unhandled exception the connection is rolled back so subsequent
      tests are not blocked by an InFailedSqlTransaction error.
    """
    pg_conn.execute("TRUNCATE TABLE news_chunks")
    pg_conn.commit()
    try:
        yield PgvectorEvidenceStore(pg_conn)
    except Exception:
        pg_conn.rollback()
        raise


# ---------------------------------------------------------------------------
# test_no_lookahead
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_lookahead(store: PgvectorEvidenceStore) -> None:
    """Retrieval must never return a document published after *before*.

    Steps:
    1. Insert a chunk with ``published_at = 2024-10-22T12:00:00Z``.
    2. ``search(before=2024-10-21T00:00:00Z)`` → must return empty list.
    3. ``search(before=2024-10-23T00:00:00Z)`` → must return the chunk.

    Implements: FIRM-9 / test_no_lookahead (SPEC §Testing Strategy).
    """
    doc = NewsDoc(
        symbol="NVDA",
        text="NVDA reported strong earnings driven by data center GPU demand.",
        source_url="https://example.com/nvda-lookahead-test",
        published_at=datetime(2024, 10, 22, 12, 0, 0, tzinfo=UTC),
    )
    store.embed_and_store(doc)
    store._conn.commit()

    before_doc = datetime(2024, 10, 21, 0, 0, 0, tzinfo=UTC)
    chunks_before = store.search("NVDA", before=before_doc)
    assert chunks_before == [], (
        f"No chunk should be returned for before={before_doc}; "
        f"chunk published_at is after the query cutoff. Got: {chunks_before}"
    )

    after_doc = datetime(2024, 10, 23, 0, 0, 0, tzinfo=UTC)
    chunks_after = store.search("NVDA", before=after_doc)
    assert len(chunks_after) >= 1, (
        f"The chunk with published_at=2024-10-22 should be returned for "
        f"before={after_doc}. Got: {chunks_after}"
    )
    assert all(c.published_at <= after_doc for c in chunks_after), (
        "All returned chunks must have published_at <= before (no-lookahead invariant)."
    )


# ---------------------------------------------------------------------------
# test_insufficient_evidence_refuses
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_insufficient_evidence_refuses(store: PgvectorEvidenceStore) -> None:
    """An empty evidence store must yield an Insufficient sentinel, not an exception.

    Steps:
    1. Empty store (table truncated by fixture).
    2. ``search(symbol='NVDA', before=<any>)`` → returns ``[]``.
    3. ``cite([])`` → returns ``Insufficient``, not ``list``.

    Implements: FIRM-9 / test_insufficient_evidence_refuses (SPEC §FR-4 grounding).
    """
    result = store.search(
        "NVDA",
        before=datetime(2024, 10, 25, 0, 0, 0, tzinfo=UTC),
    )
    assert result == [], f"Empty store must return an empty list; got {result!r}"

    citation_result = cite(result)
    assert isinstance(citation_result, Insufficient), (
        f"cite([]) must return Insufficient, got {type(citation_result).__name__!r}"
    )
    assert citation_result.reason == "no_relevant_chunks"


# ---------------------------------------------------------------------------
# Additional coverage: multiple chunks, symbol isolation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_symbol_isolation(store: PgvectorEvidenceStore) -> None:
    """search(symbol='AAPL') must not return NVDA chunks."""
    nvda_doc = NewsDoc(
        symbol="NVDA",
        text="NVIDIA GPU demand remains strong as hyperscalers ramp infrastructure.",
        source_url="https://example.com/nvda-symbol-isolation",
        published_at=datetime(2024, 10, 22, 9, 0, 0, tzinfo=UTC),
    )
    store.embed_and_store(nvda_doc)
    store._conn.commit()

    results = store.search(
        "AAPL",
        before=datetime(2024, 10, 25, 0, 0, 0, tzinfo=UTC),
    )
    assert results == [], "search('AAPL') must not return NVDA documents."


@pytest.mark.integration
def test_k_limits_results(store: PgvectorEvidenceStore) -> None:
    """search with k=2 must return at most 2 chunks even with more stored."""
    for i in range(5):
        doc = NewsDoc(
            symbol="MSFT",
            text=f"Microsoft Azure cloud services article number {i}.",
            source_url=f"https://example.com/msft-klimit-{i}",
            published_at=datetime(2024, 10, 21, i + 1, 0, 0, tzinfo=UTC),
        )
        store.embed_and_store(doc)
        store._conn.commit()

    results = store.search(
        "MSFT",
        before=datetime(2024, 10, 25, 0, 0, 0, tzinfo=UTC),
        k=2,
    )
    assert len(results) <= 2, f"search(k=2) must return at most 2 chunks; got {len(results)}"


@pytest.mark.integration
def test_cite_returns_chunks_with_source_url(store: PgvectorEvidenceStore) -> None:
    """cite(non-empty) must return the same list with source_url intact."""
    doc = NewsDoc(
        symbol="GOOGL",
        text="Alphabet Q3 earnings beat on Search and Cloud strength.",
        source_url="https://example.com/googl-cite-test",
        published_at=datetime(2024, 10, 23, 20, 0, 0, tzinfo=UTC),
    )
    store.embed_and_store(doc)
    store._conn.commit()

    chunks = store.search(
        "GOOGL",
        before=datetime(2024, 10, 25, 0, 0, 0, tzinfo=UTC),
    )
    assert chunks, "At least one chunk must be returned for the inserted doc."

    cited = cite(chunks)
    assert isinstance(cited, list), "cite(non-empty) must return list[Chunk]."
    assert all(c.source_url for c in cited), "Every cited chunk must carry a source_url."
    assert all(c.chunk_id for c in cited), "Every cited chunk must carry a chunk_id."
