"""Concrete EvidenceStore backed by Postgres + pgvector.

Implements the ``EvidenceStore`` port (``firm.ports.evidence``) using a
``news_chunks`` table with a ``vector(384)`` column for dense retrieval.

Embedding strategy:
  - Uses :class:`firm.adapters.embeddings.SentenceTransformerEmbedder` with
    ``all-MiniLM-L6-v2`` (384-dim) for deterministic local embeddings.
  - The embedder is module-level and lazy-loaded: the model weights are
    pulled from disk only on the first ``embed_and_store`` or ``search`` call,
    keeping import time negligible.

Chunking:
  - Text is split into ≤512 token pieces (approximated as ≤512 words, which
    is conservative and keeps the implementation free of tokeniser imports).
  - Each chunk gets its own UUID and a ``chunk_id`` of the form
    ``<source_url_hash>-<chunk_index>``.

Search:
  - Dense: ``ORDER BY embedding <=> query_vec`` (cosine distance) filtered by
    ``symbol`` and ``published_at <= before``, ``LIMIT k*2``.
  - Keyword boost via ``firm.rag.retriever.combine_scores``.
  - Reranked and deduplicated; top *k* returned.
  - Returns an empty list (never raises) when no results are found.

Transaction responsibility:
  - ``embed_and_store`` does NOT call ``commit()``.  The caller owns the
    transaction boundary.  Call ``conn.commit()`` after ``embed_and_store``
    when you want to flush to disk.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from firm.adapters.embeddings import SentenceTransformerEmbedder
from firm.ports.types import Chunk, NewsDoc
from firm.rag.reranker import rerank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_DIM: int = 384
_MAX_CHUNK_WORDS: int = 512

# Module-level embedder — instantiated once, model weights loaded on first use.
_embedder: SentenceTransformerEmbedder = SentenceTransformerEmbedder()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed(text: str) -> list[float]:
    """Embed *text* using the local sentence-transformer model.

    Delegates to the module-level :data:`_embedder` (``all-MiniLM-L6-v2``).
    Returns a 384-dimensional unit-normalised vector.  Deterministic: the same
    input always produces the same output.
    """
    return _embedder.embed(text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_text(text: str, max_words: int = _MAX_CHUNK_WORDS) -> list[str]:
    """Split *text* into pieces of at most *max_words* words.

    Words are split on whitespace; sentences are not respected (keeps the
    implementation simple and the corpus small enough that context loss is
    acceptable).
    """
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _chunk_id_for(source_url: str, index: int) -> str:
    """Stable chunk identifier: ``<sha256_prefix>-<index>``."""
    digest = hashlib.sha256(source_url.encode()).hexdigest()[:12]
    return f"{digest}-{index}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS news_chunks (
    id          UUID        PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    text        TEXT        NOT NULL,
    source_url  TEXT        NOT NULL,
    chunk_id    TEXT        NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    embedding   vector(384) NOT NULL,
    UNIQUE (chunk_id)
);

CREATE INDEX IF NOT EXISTS news_chunks_symbol_idx
    ON news_chunks (symbol);

CREATE INDEX IF NOT EXISTS news_chunks_published_at_idx
    ON news_chunks (published_at);

CREATE INDEX IF NOT EXISTS news_chunks_embedding_idx
    ON news_chunks USING hnsw (embedding vector_cosine_ops);
"""


# ---------------------------------------------------------------------------
# PgvectorEvidenceStore
# ---------------------------------------------------------------------------


class PgvectorEvidenceStore:
    """``EvidenceStore`` backed by Postgres + pgvector.

    Parameters
    ----------
    conn:
        An open ``psycopg.Connection`` (sync).  The connection must have
        autocommit or the caller manages transactions.  ``register_vector``
        is called on the connection during ``__init__``.
    """

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Schema bootstrap (idempotent — safe to call every startup)
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """Create the ``news_chunks`` table and register the vector type.

        ``register_vector`` is called *after* ``CREATE EXTENSION IF NOT EXISTS
        vector`` so that the OID lookup in ``pgvector`` succeeds.  Callers must
        invoke this once before using ``embed_and_store`` or ``search``.
        """
        self._conn.execute(_DDL)
        self._conn.commit()
        register_vector(self._conn)

    # ------------------------------------------------------------------
    # EvidenceStore.embed_and_store
    # ------------------------------------------------------------------

    def embed_and_store(self, doc: NewsDoc) -> None:
        """Chunk *doc*, embed each piece, and upsert into ``news_chunks``.

        This method does NOT commit.  The caller owns the transaction
        boundary — call ``conn.commit()`` (or use a context manager) after
        this returns to make changes durable.  This allows callers that
        batch multiple documents to control exactly when work is flushed.
        """
        pieces = _chunk_text(doc.text)
        for idx, piece in enumerate(pieces):
            vec = _embed(piece)
            chunk_id = _chunk_id_for(doc.source_url, idx)
            self._conn.execute(
                """
                INSERT INTO news_chunks
                    (id, symbol, text, source_url, chunk_id, published_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    doc.symbol,
                    piece,
                    doc.source_url,
                    chunk_id,
                    doc.published_at,
                    vec,
                ),
            )

    # ------------------------------------------------------------------
    # EvidenceStore.search
    # ------------------------------------------------------------------

    def search(
        self,
        symbol: str,
        *,
        before: datetime,
        k: int = 10,
        query: str | None = None,
    ) -> list[Chunk]:
        """Return up to *k* chunks for *symbol* published at or before *before*.

        Uses pgvector cosine distance for dense retrieval, then applies keyword
        boost and reranks via ``firm.rag.reranker.rerank``.  Returns an empty
        list when no rows match — never raises.

        Parameters
        ----------
        query:
            Free-text query to embed for dense retrieval.  When ``None`` the
            symbol ticker is used as the query so that retrieval is at least
            weakly relevant.  Callers with a specific question (e.g. an agent's
            research question) should supply it here for best recall.
        """
        query_text = query if query is not None else symbol
        query_vec = _embed(query_text)
        rows = _fetch_raw_rows(self._conn, symbol, before, query_vec, k)
        if not rows:
            return []
        chunks = [_row_to_chunk(row) for row in rows]
        return rerank(chunks, query=query_text, before=before, k=k)


# ---------------------------------------------------------------------------
# Internal SQL helper
# ---------------------------------------------------------------------------


def _fetch_raw_rows(
    conn: psycopg.Connection[Any],
    symbol: str,
    before: datetime,
    query_vec: list[float],
    k: int,
) -> list[tuple[Any, ...]]:
    """Execute the dense-retrieval query and return raw DB rows."""
    return conn.execute(
        """
        SELECT
            id,
            symbol,
            text,
            source_url,
            chunk_id,
            published_at,
            embedding,
            1.0 - (embedding <=> %s::vector) AS dense_score
        FROM news_chunks
        WHERE symbol = %s
          AND published_at <= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, symbol, before, query_vec, k * 2),
    ).fetchall()


# ---------------------------------------------------------------------------
# Row → Chunk conversion
# ---------------------------------------------------------------------------


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    """Convert a DB row from ``search`` into a ``Chunk``."""
    (
        row_id,
        symbol,
        text,
        source_url,
        chunk_id,
        published_at,
        embedding,
        dense_score,
    ) = row
    embedding_list: list[float] = list(embedding) if embedding is not None else []
    return Chunk(
        id=uuid.UUID(str(row_id)),
        symbol=symbol,
        text=text,
        source_url=source_url,
        chunk_id=chunk_id,
        published_at=published_at,
        score=float(dense_score),
        embedding=embedding_list,
    )
