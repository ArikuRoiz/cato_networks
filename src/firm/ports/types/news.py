"""Evidence / RAG value objects."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Chunk(BaseModel):
    """A retrieved, scored document fragment from the evidence store."""

    id: UUID
    symbol: str
    text: str
    source_url: str
    chunk_id: str
    published_at: datetime
    score: float = 0.0

    model_config = {"frozen": True}


class NewsDoc(BaseModel):
    """A raw news document to be embedded and stored in the evidence store."""

    symbol: str
    text: str
    source_url: str
    published_at: datetime

    model_config = {"frozen": True}
