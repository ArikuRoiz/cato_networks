"""Change news_chunks.embedding from vector(1024) to vector(384).

Aligns the pgvector column dimension with the all-MiniLM-L6-v2 model
output (384 dimensions) used by SentenceTransformerEmbedder.

The ``news_chunks`` table is bootstrapped outside Alembic by
``PgvectorEvidenceStore.migrate()`` (requires pgvector extension).  This
migration is therefore a no-op on databases that do not have the table yet
(e.g. the plain-Postgres ledger test container).  When the table exists, the
old random-vector column (1024-dim) is replaced with a 384-dim column.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-23 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _news_chunks_exists() -> bool:
    """Return True if the news_chunks table already exists in this DB."""
    conn = op.get_bind()
    result = conn.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'news_chunks' AND table_schema = 'public'"
        )
    )
    return result.fetchone() is not None


def upgrade() -> None:
    if not _news_chunks_exists():
        # Table managed by PgvectorEvidenceStore.migrate(); not present here.
        return

    # pgvector does not support ALTER COLUMN … TYPE for vector columns
    # in-place, so we drop and re-add.  Existing embeddings (random vectors
    # from the old stub) are discarded — they were meaningless anyway.
    op.execute(text("DROP INDEX IF EXISTS news_chunks_embedding_idx"))
    op.execute(text("ALTER TABLE news_chunks DROP COLUMN embedding"))
    op.execute(
        text(
            "ALTER TABLE news_chunks ADD COLUMN embedding vector(384) NOT NULL "
            "DEFAULT array_fill(0, ARRAY[384])::vector(384)"
        )
    )
    op.execute(text("ALTER TABLE news_chunks ALTER COLUMN embedding DROP DEFAULT"))
    op.execute(
        text(
            "CREATE INDEX news_chunks_embedding_idx "
            "ON news_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    if not _news_chunks_exists():
        return

    op.execute(text("DROP INDEX IF EXISTS news_chunks_embedding_idx"))
    op.execute(text("ALTER TABLE news_chunks DROP COLUMN embedding"))
    op.execute(
        text(
            "ALTER TABLE news_chunks ADD COLUMN embedding vector(1024) NOT NULL "
            "DEFAULT array_fill(0, ARRAY[1024])::vector(1024)"
        )
    )
    op.execute(text("ALTER TABLE news_chunks ALTER COLUMN embedding DROP DEFAULT"))
    op.execute(
        text(
            "CREATE INDEX news_chunks_embedding_idx "
            "ON news_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    )
