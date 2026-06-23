"""Domain string enumerations.

All values are plain strings so they round-trip through JSON
(LangGraph checkpointer, Pydantic ``model_dump(mode="json")``).
"""

from __future__ import annotations

from enum import StrEnum


class LLMModel(StrEnum):
    """Short model aliases routed to versioned model IDs by the LLM adapter.

    ``HAIKU`` is the cheap extraction model; ``SONNET`` the strong decision model.
    """

    HAIKU = "haiku"
    SONNET = "sonnet"


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class TriggerType(StrEnum):
    SCHEDULED = "scheduled"
    EVENT = "event"


class CycleOutcome(StrEnum):
    FILLED = "filled"
    REJECTED = "rejected"
    REJECTED_TIMEOUT = "rejected_timeout"
    REJECTED_MARKET_CLOSED = "rejected_market_closed"
    HOLD = "hold"
    ERROR = "error"


class HITLStatus(StrEnum):
    """Unified HITL / approval status enum.

    ``ApprovalStatus`` is an alias for backward compatibility.
    ``PENDING`` is reserved for future use (no active code path sets it yet).
    ``EDITED`` has been removed — no code path ever set it.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Backward-compatible alias — prefer HITLStatus in new code.
ApprovalStatus = HITLStatus


class Recommendation(StrEnum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class VerdictAlignment(StrEnum):
    ALIGNED = "aligned"
    PARTIAL = "partial"
    MISALIGNED = "misaligned"


class TechnicalBias(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MACDCross(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"


class RefusalReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STORE_UNAVAILABLE = "store_unavailable"
    INJECTION_DETECTED = "injection_detected"
    LLM_ERROR_RETRYABLE = "llm_error_retryable"
    LLM_ERROR_NON_RETRYABLE = "llm_error_non_retryable"
