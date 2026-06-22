"""Reporting agent I/O schemas."""

from __future__ import annotations

from datetime import date
from uuid import UUID

from pydantic import BaseModel


class ReportingInput(BaseModel):
    cycle_id: UUID
    portfolio_id: UUID
    report_date: date
    correlation_id: str

    model_config = {"frozen": True}


class ReportSent(BaseModel):
    report_date: date

    model_config = {"frozen": True}


class ReportFailure(BaseModel):
    reason: str

    model_config = {"frozen": True}
