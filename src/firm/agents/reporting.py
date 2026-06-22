"""ReportingAgent — build a DailyReport and dispatch it via the ReportSink.

Input:  ReportingInput(cycle_id, portfolio_id, date, correlation_id)
Output: ReportSent | ReportFailure

Failures from the sink (network, Slack, Excel) are caught and returned as
ReportFailure — the agent never raises for expected sink errors.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from firm.domain import Portfolio, Trade
from firm.persistence.ledger import LedgerRepository
from firm.ports.report import ReportSink
from firm.ports.types import DailyReport, PositionRecord, TradeRecord

# ---------------------------------------------------------------------------
# I/O schemas
# ---------------------------------------------------------------------------


class ReportingInput(BaseModel):
    """Input contract for ReportingAgent."""

    cycle_id: UUID
    portfolio_id: UUID
    report_date: date
    correlation_id: str

    model_config = {"frozen": True}


class ReportSent(BaseModel):
    """Report was successfully dispatched."""

    report_date: date

    model_config = {"frozen": True}


class ReportFailure(BaseModel):
    """Report could not be sent — failure is a value, not an exception."""

    reason: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReportingAgent:
    """Compile a DailyReport from the ledger and dispatch it via the sink."""

    def __init__(
        self,
        report_sink: ReportSink,
        ledger: LedgerRepository,
    ) -> None:
        self._sink = report_sink
        self._ledger = ledger

    def run(self, inp: ReportingInput) -> ReportSent | ReportFailure:
        """Build and send the daily report; return result union, never raise."""
        try:
            portfolio = self._ledger.get_portfolio(inp.portfolio_id)
        except Exception as exc:
            return ReportFailure(reason=f"ledger unavailable: {exc}")

        trades = _load_trades(self._ledger, inp.cycle_id)
        report = _build_report(inp.report_date, portfolio, trades)

        try:
            self._sink.send_daily_report(report)
        except Exception as exc:
            return ReportFailure(reason=f"sink error: {exc}")

        return ReportSent(report_date=inp.report_date)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_trades(ledger: LedgerRepository, cycle_id: UUID) -> list[TradeRecord]:
    """Load filled trades for the cycle; degrade gracefully on error."""
    try:
        return _fetch_trade_records(ledger, cycle_id)
    except Exception:
        return []


def _fetch_trade_records(ledger: LedgerRepository, cycle_id: UUID) -> list[TradeRecord]:
    """Fetch Trade rows for cycle_id and convert to TradeRecord dicts."""
    raw_trades: list[Trade] = ledger.get_trades_for_cycle(cycle_id)
    return [_trade_to_record(t) for t in raw_trades]


def _trade_to_record(trade: Trade) -> TradeRecord:
    """Convert a Trade domain object to a TradeRecord TypedDict."""
    record: TradeRecord = {
        "cycle_id": str(trade.cycle_id),
        "symbol": trade.symbol,
        "side": trade.side,
        "qty": float(trade.qty),
        "fill_price": float(trade.fill_price) if trade.fill_price else 0.0,
        "slippage": float(trade.slippage) if trade.slippage else 0.0,
        "commission": float(trade.commission) if trade.commission else 0.0,
        "status": trade.status.value,
    }
    return record


def _build_positions(portfolio: Portfolio) -> list[PositionRecord]:
    """Derive position records from the portfolio's current holdings."""
    records: list[PositionRecord] = []
    for symbol, holding in portfolio.holdings.items():
        if holding.quantity <= Decimal("0"):
            continue
        records.append(
            PositionRecord(
                symbol=symbol,
                qty=float(holding.quantity),
                avg_cost=float(holding.avg_cost),
                current_price=0.0,  # live price not available in reporting path
                unrealized_pnl=0.0,
            )
        )
    return records


def _build_report(
    report_date: date,
    portfolio: Portfolio,
    trades: list[TradeRecord],
    prices: dict[str, Decimal] | None = None,
) -> DailyReport:
    """Assemble the DailyReport from portfolio state and trade records.

    NAV is computed from all holdings when *prices* is supplied; otherwise it
    falls back to cash-only NAV.  Callers should pass a price snapshot when
    available to avoid a cash-only undercount.
    """
    positions = _build_positions(portfolio)
    nav = _compute_nav(portfolio, prices)
    return DailyReport(
        date=report_date,
        nav=nav,
        pnl=Decimal("0"),
        benchmark_return=0.0,
        trades=trades,
        positions=positions,
        citations=[],
    )


def _compute_nav(portfolio: Portfolio, prices: dict[str, Decimal] | None) -> Decimal:
    """Return full NAV when prices are available, else cash-only NAV.

    NOTE: when *prices* is None the result is cash-only and does not reflect
    open equity positions.  This is a known limitation of the offline reporting
    path; callers should supply a price snapshot where possible.
    """
    if prices is None:
        return portfolio.cash
    try:
        return portfolio.nav(prices)
    except (ValueError, KeyError):
        return portfolio.cash
