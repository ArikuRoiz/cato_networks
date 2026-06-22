"""News ingestion agent I/O schemas."""

from __future__ import annotations

from pydantic import BaseModel


class NewsIngestionInput(BaseModel):
    symbols: list[str]
    lookback_hours: int = 24
    correlation_id: str

    model_config = {"frozen": True}


class NewsIngested(BaseModel):
    articles_added: int
    symbols_updated: list[str]

    model_config = {"frozen": True}


class NewsIngestionFailure(BaseModel):
    reason: str
    symbol: str | None = None

    model_config = {"frozen": True}
