"""ReportingAgent — build a DailyReport and dispatch it via the ReportSink.

NAV formula:
    NAV = cash + Σ(qty × current_price) for each held symbol.
    When prices are unavailable for a symbol, avg_cost is used as a fallback
    so NAV is always computed (never hard-coded).

P&L formula:
    pnl = Σ((current_price − avg_cost) × qty) across all open positions.
    This is total unrealised P&L vs cost basis for the cycle snapshot.
    Realised P&L from intra-cycle fills is not separately tracked here because
    Trade objects do not carry a realised_pnl field; callers that need intra-day
    realised P&L should query the ledger directly.

Benchmark formula:
    benchmark_return = (spy_today − spy_prev) / spy_prev
    where spy_today  = prices["SPY"] (close of report_date)
      and spy_prev   = prices["SPY_PREV"] (close of the preceding trading day).
    If either price is absent, benchmark_return is 0.0 and no error is raised.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from uuid import UUID

from firm.agents.base import BaseAgent
from firm.agents.reporting.schemas import ReportFailure, ReportingInput, ReportSent
from firm.domain import Portfolio, Trade
from firm.persistence.ledger import LedgerRepository
from firm.ports.report import ReportSink
from firm.ports.types import DailyReport, PositionRecord, TradeRecord

logger = logging.getLogger(__name__)

# Sentinel key used to pass the previous SPY close through the prices dict.
_SPY_PREV_KEY = "SPY_PREV"


class ReportingAgent(BaseAgent[ReportingInput, ReportSent | ReportFailure]):
    def __init__(self, report_sink: ReportSink, ledger: LedgerRepository) -> None:
        self._sink = report_sink
        self._ledger = ledger

    def run(self, inp: ReportingInput) -> ReportSent | ReportFailure:
        try:
            portfolio = self._ledger.get_portfolio(inp.portfolio_id)
        except Exception as exc:
            return ReportFailure(reason=f"ledger unavailable: {exc}")

        trades = _load_trades(self._ledger, inp.cycle_id)
        report = _build_report(inp.report_date, portfolio, trades, inp.prices)

        try:
            self._sink.send_daily_report(report)
        except Exception as exc:
            return ReportFailure(reason=f"sink error: {exc}")

        return ReportSent(report_date=inp.report_date)


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _load_trades(ledger: LedgerRepository, cycle_id: UUID) -> list[TradeRecord]:
    try:
        return [_trade_to_record(t) for t in ledger.get_trades_for_cycle(cycle_id)]
    except Exception:
        logger.exception("Failed to load trades for cycle %s", cycle_id)
        return []


def _trade_to_record(trade: Trade) -> TradeRecord:
    return TradeRecord(
        cycle_id=str(trade.cycle_id),
        symbol=trade.symbol,
        side=trade.side,
        qty=float(trade.qty),
        fill_price=float(trade.fill_price) if trade.fill_price else 0.0,
        slippage=float(trade.slippage) if trade.slippage else 0.0,
        commission=float(trade.commission) if trade.commission else 0.0,
        status=trade.status.value,
    )


def _effective_price(symbol: str, prices: dict[str, Decimal], fallback: Decimal) -> Decimal:
    """Return the market price for *symbol*, or *fallback* when absent."""
    return prices.get(symbol, fallback)


def _build_positions(
    portfolio: Portfolio,
    prices: dict[str, Decimal],
) -> list[PositionRecord]:
    records: list[PositionRecord] = []
    for symbol, holding in portfolio.holdings.items():
        if holding.quantity <= Decimal("0"):
            continue
        avg_cost = holding.avg_cost
        current_price = _effective_price(symbol, prices, avg_cost)
        unrealized_pnl = (current_price - avg_cost) * holding.quantity
        records.append(
            PositionRecord(
                symbol=symbol,
                qty=float(holding.quantity),
                avg_cost=float(avg_cost),
                current_price=float(current_price),
                unrealized_pnl=float(unrealized_pnl),
            )
        )
    return records


def _compute_nav(portfolio: Portfolio, prices: dict[str, Decimal]) -> Decimal:
    """NAV = cash + Σ(qty × current_price).

    Falls back to avg_cost for any symbol missing from *prices* so NAV is
    always computable even when market data is partially absent.
    """
    equity = sum(
        holding.quantity * _effective_price(symbol, prices, holding.avg_cost)
        for symbol, holding in portfolio.holdings.items()
        if holding.quantity > Decimal("0")
    )
    return portfolio.cash + equity


def _compute_unrealized_pnl(
    portfolio: Portfolio,
    prices: dict[str, Decimal],
) -> Decimal:
    """Total unrealised P&L = Σ((current_price − avg_cost) × qty)."""
    return sum(
        (
            (_effective_price(symbol, prices, holding.avg_cost) - holding.avg_cost)
            * holding.quantity
        )
        for symbol, holding in portfolio.holdings.items()
        if holding.quantity > Decimal("0")
    )


def _compute_benchmark_return(prices: dict[str, Decimal]) -> float:
    """SPY day-over-day return.

    Expects prices to contain:
      "SPY"      — today's close (report_date bar)
      "SPY_PREV" — previous trading day's close

    Returns 0.0 when either price is absent or the previous close is zero.
    """
    spy_today = prices.get("SPY")
    spy_prev = prices.get(_SPY_PREV_KEY)
    if spy_today is None or spy_prev is None or spy_prev == Decimal("0"):
        return 0.0
    return float((spy_today - spy_prev) / spy_prev)


def _build_report(
    report_date: date,
    portfolio: Portfolio,
    trades: list[TradeRecord],
    prices: dict[str, Decimal],
) -> DailyReport:
    positions = _build_positions(portfolio, prices)
    nav = _compute_nav(portfolio, prices)
    pnl = _compute_unrealized_pnl(portfolio, prices)
    benchmark_return = _compute_benchmark_return(prices)
    return DailyReport(
        date=report_date,
        nav=nav,
        pnl=pnl,
        benchmark_return=benchmark_return,
        trades=trades,
        positions=positions,
        citations=[],
    )
