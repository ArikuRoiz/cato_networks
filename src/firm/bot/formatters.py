"""Channel-agnostic message formatters for the Telegram bot.

All functions are pure: dict in → str out. No IO, no side effects.
Telegram-specific escaping is applied by the bot service layer, not here.

Formats
-------
- ``format_approval_card``   : rich HITL card (research context + trade line + buttons hint)
- ``format_decision_report`` : post-approval/rejection "what I did & why"
- ``format_portfolio_report``: NAV / P&L / cash snapshot
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_approval_card(req_dict: dict[str, Any]) -> str:
    """Return the human-readable approval card text for a HITL interrupt.

    *req_dict* is the serialised HITLRequest (plain dict from model_dump).
    """
    symbol = req_dict.get("symbol", "?")
    side = str(req_dict.get("side", "buy")).upper()
    qty = req_dict.get("qty_str", "?")
    notional = req_dict.get("notional", "?")
    reason = req_dict.get("reason", "")
    recommendation = req_dict.get("recommendation")
    conviction = req_dict.get("conviction")
    rationale = req_dict.get("rationale")
    bull_case = req_dict.get("bull_case")
    bear_case = req_dict.get("bear_case")

    notional_str = _fmt_notional(notional)
    conviction_str = _fmt_conviction(conviction)
    rec_str = _fmt_recommendation(recommendation)

    lines = [
        f"🚨 *Trade Approval Required — {symbol}*",
        "",
        f"📊 *Recommendation:* {rec_str}   *Conviction:* {conviction_str}",
    ]

    if rationale:
        lines += ["", f"💡 *Why:* {rationale}"]

    if bull_case:
        lines += ["", f"👍 *Pros:* {bull_case}"]

    if bear_case:
        lines += ["", f"👎 *Cons:* {bear_case}"]

    lines += [
        "",
        f"📋 *Trade:* {side} {qty} {symbol} ≈ ${notional_str}",
    ]

    if reason:
        lines += [f"⚖️ *Risk note:* {reason}"]

    lines += ["", "Tap *Approve* or *Reject* below."]
    return "\n".join(lines)


def format_decision_report(
    symbol: str,
    hitl_status: str,
    cycle_outcome: str | None,
    synthesis: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    approved_trade: dict[str, Any] | None,
    rejection_reason: str | None = None,
) -> str:
    """Return the post-decision "what I did & why" message.

    Covers both approved (filled) and rejected paths.
    """
    was_approved = hitl_status == "approved"
    was_filled = cycle_outcome == "filled"

    if was_approved and was_filled:
        return _format_fill_report(symbol, approved_trade, synthesis, verdict)
    if was_approved and not was_filled:
        return _format_approved_not_filled(symbol, cycle_outcome, synthesis)
    return _format_rejection_report(symbol, rejection_reason, synthesis)


def format_portfolio_report(
    nav: str,
    cash: str,
    pnl: str,
    holdings: list[dict[str, Any]],
) -> str:
    """Return a compact portfolio snapshot message."""
    lines = [
        "📈 *Portfolio Snapshot*",
        "",
        f"💰 *NAV:* ${nav}",
        f"🏦 *Cash:* ${cash}",
        f"📊 *P&L:* {pnl}",
    ]
    if holdings:
        lines += ["", "*Holdings:*"]
        for h in holdings:
            sym = h.get("symbol", "?")
            qty = h.get("qty", "?")
            lines.append(f"  • {sym}: {qty} shares")
    else:
        lines += ["", "_No open positions._"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers — each does exactly one formatting concern
# ---------------------------------------------------------------------------


def _fmt_notional(notional: Any) -> str:
    try:
        return f"{float(notional):,.2f}"
    except (TypeError, ValueError):
        return str(notional)


def _fmt_conviction(conviction: Any) -> str:
    if conviction is None:
        return "-"
    try:
        return f"{float(conviction) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(conviction)


def _fmt_recommendation(recommendation: Any) -> str:
    if not recommendation:
        return "-"
    return str(recommendation).replace("_", " ").title()


def _fmt_score(verdict: dict[str, Any] | None) -> str:
    if not verdict:
        return "-"
    score = verdict.get("coherence_score")
    alignment = verdict.get("alignment", "")
    if score is None:
        return "-"
    return f"{score}/5 {alignment}"


def _fmt_summary(synthesis: dict[str, Any] | None) -> str:
    if not synthesis:
        return ""
    return str(synthesis.get("executive_summary", ""))


def _format_fill_report(
    symbol: str,
    approved_trade: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
) -> str:
    trade = approved_trade or {}
    qty = trade.get("trade", {}).get("qty") if isinstance(trade.get("trade"), dict) else None
    price = (
        trade.get("trade", {}).get("requested_price")
        if isinstance(trade.get("trade"), dict)
        else None
    )

    fill_line = f"✅ *Approved → FILLED {qty or '?'} {symbol}"
    if price is not None:
        fill_line += f" @ ${price}"
    fill_line += "*"

    lines = [fill_line]

    summary = _fmt_summary(synthesis)
    if summary:
        lines += ["", f"📝 {summary}"]

    score = _fmt_score(verdict)
    if score != "-":
        lines += [f"⚖️ Judge: {score}"]

    return "\n".join(lines)


def _format_approved_not_filled(
    symbol: str,
    cycle_outcome: str | None,
    synthesis: dict[str, Any] | None,
) -> str:
    reason = cycle_outcome or "unknown"
    lines = [f"✅ *Approved* — trade not filled ({reason}) for {symbol}."]
    summary = _fmt_summary(synthesis)
    if summary:
        lines += ["", f"📝 {summary}"]
    return "\n".join(lines)


def _format_rejection_report(
    symbol: str,
    rejection_reason: str | None,
    synthesis: dict[str, Any] | None,
) -> str:
    reason_str = rejection_reason or "operator rejected"
    lines = [f"❌ *Rejected → no trade booked for {symbol}.*", f"_{reason_str}_"]
    summary = _fmt_summary(synthesis)
    if summary:
        lines += ["", f"📝 {summary}"]
    return "\n".join(lines)
