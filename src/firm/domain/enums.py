"""Domain string enumerations.

All values are plain strings so they round-trip through JSON
(LangGraph checkpointer, Pydantic ``model_dump(mode="json")``).
"""

from __future__ import annotations

from enum import StrEnum


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
    HOLD = "hold"
    ERROR = "error"


class HITLStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    EXPIRED = "expired"


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
