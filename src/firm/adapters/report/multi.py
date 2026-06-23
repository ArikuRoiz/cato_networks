"""MultiReportSink — composite ReportSink that fans out to multiple channels.

Design decisions
----------------
* ``send_daily_report`` fans out to *every* wrapped sink.  A failure in one
  sink is logged and skipped so that the remaining sinks still receive the report.
* ``send_hitl_request`` is routed exclusively to the designated HITL sink
  (``hitl_sink``).  Fan-out for HITL doesn't make sense — only one channel
  should gate approval.  If no ``hitl_sink`` is provided the first sink that is
  not ``ExcelReportSink`` is used; if none qualifies the first sink is used.
* ``send_alert`` fans out to every sink (best-effort, same failure isolation as
  ``send_daily_report``).
"""

from __future__ import annotations

import logging
from typing import Sequence

from firm.ports.report import ReportSink
from firm.ports.types import ApprovalResult, DailyReport, HITLRequest

logger = logging.getLogger(__name__)


class MultiReportSink:
    """Fan-out ReportSink that wraps one or more ReportSink implementations.

    Parameters
    ----------
    sinks:
        Ordered sequence of sinks.  At least one is required.
    hitl_sink:
        Sink to delegate HITL requests to.  Defaults to the first non-Excel
        sink, or the first sink when all are Excel-type.
    """

    def __init__(
        self,
        sinks: Sequence[ReportSink],
        hitl_sink: ReportSink | None = None,
    ) -> None:
        if not sinks:
            raise ValueError("MultiReportSink requires at least one sink")
        self._sinks = list(sinks)
        self._hitl_sink = hitl_sink or _choose_hitl_sink(self._sinks)

    # ------------------------------------------------------------------
    # ReportSink interface
    # ------------------------------------------------------------------

    def send_daily_report(self, report: DailyReport) -> None:
        """Fan out the daily report to every wrapped sink."""
        for sink in self._sinks:
            _call_safely(sink.send_daily_report, report, sink=sink, method="send_daily_report")

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Delegate the HITL request to the designated HITL sink."""
        return self._hitl_sink.send_hitl_request(req)

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Fan out the alert to every wrapped sink."""
        for sink in self._sinks:
            _call_safely(
                sink.send_alert,
                message,
                correlation_id,
                sink=sink,
                method="send_alert",
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _choose_hitl_sink(sinks: list[ReportSink]) -> ReportSink:
    """Pick the best HITL-capable sink from *sinks*.

    Preference: first sink whose class name does not contain "Excel".
    Fallback: first sink in the list.
    """
    for sink in sinks:
        if "Excel" not in type(sink).__name__:
            return sink
    return sinks[0]


def _call_safely(fn: object, *args: object, sink: ReportSink, method: str) -> None:
    """Invoke *fn* with *args*, logging (not re-raising) any exception."""
    try:
        fn(*args)  # type: ignore[operator]
    except Exception:
        logger.exception(
            "MultiReportSink: %s.%s raised — continuing with remaining sinks",
            type(sink).__name__,
            method,
        )
