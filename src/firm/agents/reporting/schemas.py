"""Reporting agent I/O schemas."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


class ReportingInput(BaseModel):
    cycle_id: UUID
    portfolio_id: UUID
    report_date: date
    correlation_id: str
    # Current market prices keyed by symbol (holdings + SPY for benchmark).
    # Fetched by make_reporting_node via ports.market_data so the agent itself
    # remains IO-free and deterministically testable.
    prices: dict[str, Decimal] = {}

    model_config = {"frozen": True}


class ReportSent(BaseModel):
    report_date: date

    model_config = {"frozen": True}


class ReportFailure(BaseModel):
    reason: str

    model_config = {"frozen": True}
