from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from firm.domain.enums import ApprovalStatus
from firm.ports.types import ApprovalResult, DailyReport, HITLRequest


class FileReportSink:
    def __init__(self, output_dir: Path | str = "reports", hitl_auto_approve: bool = True) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hitl_auto_approve = hitl_auto_approve

    def send_daily_report(self, report: DailyReport) -> None:
        (self._dir / f"report_{report.date}.txt").write_text(_render_text(report), encoding="utf-8")

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        if not self._hitl_auto_approve:
            raise RuntimeError(
                f"FileReportSink: HITL request received but hitl_auto_approve=False "
                f"(correlation_id={req.correlation_id})"
            )
        self._append_alert(
            f"HITL auto-approved: {req.symbol} {req.side} {req.qty_str} notional={req.notional}",
            req.correlation_id,
        )
        return ApprovalResult(status=ApprovalStatus.APPROVED)

    def send_alert(self, message: str, correlation_id: str) -> None:
        self._append_alert(message, correlation_id)

    def _append_alert(self, message: str, correlation_id: str) -> None:
        ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
        with (self._dir / "alerts.log").open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{correlation_id[:8]}] {message}{os.linesep}")


def _render_text(report: DailyReport) -> str:
    pnl_sign = "+" if report.pnl >= 0 else ""
    bench_sign = "+" if report.benchmark_return >= 0 else ""
    lines: list[str] = [
        "=" * 60,
        f"  DAILY REPORT — {report.date}",
        "=" * 60,
        "",
        "PORTFOLIO SUMMARY",
        f"  NAV              ${report.nav:>14,.2f}",
        f"  PnL (day)        {pnl_sign}{report.pnl:>14,.2f}",
        f"  Benchmark return {bench_sign}{report.benchmark_return:>13.2%}",
        "",
    ]
    if report.trades:
        lines.append("TRADES")
        for t in report.trades:
            lines.append(
                f"  {t['side'].upper():4s} {t['symbol']:6s} {t['qty']:>8.2f} @ ${t['fill_price']:>9.2f}  [{t['status']}]"
            )
        lines.append("")
    if report.positions:
        lines.append("POSITIONS")
        for p in report.positions:
            pnl_str = (
                f"+{p['unrealized_pnl']:.2f}"
                if p["unrealized_pnl"] >= 0
                else f"{p['unrealized_pnl']:.2f}"
            )
            lines.append(
                f"  {p['symbol']:6s} {p['qty']:>8.2f} shares  cost ${p['avg_cost']:.2f}  upnl {pnl_str}"
            )
        lines.append("")
    if report.citations:
        lines.append("EVIDENCE CITED")
        for c in report.citations:
            lines.append(f"  [{c['symbol']}] {c['source_url']}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"
