"""Protocol interfaces defining the IO seams: MarketDataSource, EvidenceStore, LLM, ReportSink."""

from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.ports.report import ReportSink
from firm.ports.types import (
    ApprovalResult,
    Chunk,
    DailyReport,
    HITLRequest,
    LLMError,
    LLMMessage,
    LLMResponse,
    NewsDoc,
)

__all__ = [
    "LLM",
    "ApprovalResult",
    "Chunk",
    "DailyReport",
    "EvidenceStore",
    "HITLRequest",
    "LLMError",
    "LLMMessage",
    "LLMResponse",
    "MarketDataSource",
    "NewsDoc",
    "ReportSink",
]
