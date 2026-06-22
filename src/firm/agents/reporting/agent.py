"""ReportingAgent — build a DailyReport and dispatch it via the ReportSink."""

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
        report = _build_report(inp.report_date, portfolio, trades)

        try:
            self._sink.send_daily_report(report)
        except Exception as exc:
            return ReportFailure(reason=f"sink error: {exc}")

        return ReportSent(report_date=inp.report_date)


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


def _build_positions(portfolio: Portfolio) -> list[PositionRecord]:
    return [
        PositionRecord(
            symbol=symbol,
            qty=float(holding.quantity),
            avg_cost=float(holding.avg_cost),
            current_price=0.0,
            unrealized_pnl=0.0,
        )
        for symbol, holding in portfolio.holdings.items()
        if holding.quantity > Decimal("0")
    ]


def _build_report(
    report_date: date,
    portfolio: Portfolio,
    trades: list[TradeRecord],
    prices: dict[str, Decimal] | None = None,
) -> DailyReport:
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
    if prices is None:
        return portfolio.cash
    try:
        return portfolio.nav(prices)
    except (ValueError, KeyError):
        return portfolio.cash
