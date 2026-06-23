"""Web-layer request/response schemas.

All JSON bodies entering the API are validated here; downstream code consumes
typed objects, never raw dicts.

Module order: request types → response types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Request types (wrap inbound JSON at the boundary)
# ---------------------------------------------------------------------------


class RunRequest(BaseModel, frozen=True):
    """Body of ``POST /api/run``."""

    tickers: list[str]
    lookback_days: int = 7
    force_buy: bool = False
    """Demo/override flag: inject a synthetic high-conviction BUY plan so the
    pipeline produces a trade above the HITL threshold, guaranteeing a pause
    for human-in-the-loop approval.  Does NOT affect the real LLM decision
    path; the flag is clearly named as an override.
    """

    @field_validator("tickers")
    @classmethod
    def _tickers_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("tickers must be a non-empty list")
        return [t.strip().upper() for t in v if t.strip()]

    @field_validator("lookback_days")
    @classmethod
    def _lookback_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("lookback_days must be > 0")
        return v

    @property
    def is_demo(self) -> bool:
        """True when force_buy override is active."""
        return self.force_buy


class ApprovalRequest(BaseModel, frozen=True):
    """Body of ``POST /api/approvals/{thread_id}``."""

    decision: str  # "approve" | "reject"
    edited_qty: Decimal | None = None

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v: str) -> str:
        allowed = {"approve", "reject"}
        if v not in allowed:
            raise ValueError(f"decision must be one of {allowed}")
        return v

    @property
    def is_approve(self) -> bool:
        return self.decision == "approve"


# ---------------------------------------------------------------------------
# Response types (outbound typed shapes; serialised via .to_dict())
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldingDTO:
    symbol: str
    quantity: str
    avg_cost: str

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "quantity": self.quantity, "avg_cost": self.avg_cost}


@dataclass(frozen=True)
class PortfolioDTO:
    cash: str
    nav: str
    pnl: str
    holdings: list[HoldingDTO] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "nav": self.nav,
            "pnl": self.pnl,
            "holdings": [h.to_dict() for h in self.holdings],
        }


@dataclass(frozen=True)
class TradeDTO:
    id: str
    symbol: str
    side: str
    qty: str
    status: str
    fill_price: str | None
    filled_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "status": self.status,
            "fill_price": self.fill_price,
            "filled_at": self.filled_at,
        }


@dataclass(frozen=True)
class CycleDTO:
    id: str
    correlation_id: str | None
    trigger_type: str
    outcome: str | None
    started_at: str
    judge_score: int | None
    alignment: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "trigger_type": self.trigger_type,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "judge_score": self.judge_score,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class AuditEntryDTO:
    id: str
    correlation_id: str
    actor: str
    action: str
    payload: dict[str, Any]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "actor": self.actor,
            "action": self.action,
            "payload": self.payload,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class PendingApprovalDTO:
    thread_id: str
    correlation_id: str
    symbol: str
    notional: str
    interrupt_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "correlation_id": self.correlation_id,
            "symbol": self.symbol,
            "notional": self.notional,
            "interrupt_payload": self.interrupt_payload,
        }


@dataclass(frozen=True)
class ApprovalResultDTO:
    thread_id: str
    outcome: str
    hitl_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "outcome": self.outcome,
            "hitl_status": self.hitl_status,
        }


@dataclass(frozen=True)
class RunStartedDTO:
    thread_ids: list[str]
    tickers: list[str]
    lookback_days: int
    force_buy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_ids": self.thread_ids,
            "tickers": self.tickers,
            "lookback_days": self.lookback_days,
            "force_buy": self.force_buy,
        }


# ---------------------------------------------------------------------------
# Portfolio NAV computation helper
# ---------------------------------------------------------------------------


def compute_nav_and_pnl(
    cash: Decimal,
    holdings: list[tuple[str, Decimal, Decimal]],  # (symbol, qty, avg_cost)
    initial_cash: Decimal = Decimal("100000"),
) -> tuple[Decimal, Decimal]:
    """Return (nav, pnl) from cash + holdings.

    Without live prices the best we can do is cost-basis NAV:
    NAV = cash + sum(qty * avg_cost).  P&L = NAV - initial_cash.
    """
    equity = sum(qty * cost for _, qty, cost in holdings)
    nav = cash + equity
    return nav, nav - initial_cash


# ---------------------------------------------------------------------------
# UUID validation helper (used by routers)
# ---------------------------------------------------------------------------


def parse_uuid(value: str) -> UUID:
    """Parse *value* as UUID; raises ValueError on invalid input."""
    return UUID(value)
