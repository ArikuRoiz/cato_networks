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

    lines += [""]
    if _is_real_trade(qty, notional):
        lines += [f"📋 *Trade:* {side} {qty} {symbol} ≈ ${notional_str}"]
    else:
        lines += [
            f"📋 *Proposed action:* {rec_str} — no trade to place "
            "(tap *Reject* to place a Buy/Sell instead)."
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
    hitl_decision: str | None = None,
    recommendation: str | None = None,
) -> str:
    """Return the post-decision "what I did & why" message.

    The label reflects the ACTUAL decision the human made, not merely
    approve-vs-reject: an approve, an override-buy, an override-sell, and an
    override-to-hold each read differently — because both an approve and an
    override resolve to ``hitl_status == "approved"`` internally, the
    ``hitl_decision`` is what disambiguates them here.
    """
    is_override = _is_override(hitl_decision)
    was_filled = cycle_outcome == "filled"

    if _is_hold_decision(hitl_decision):
        return _format_held_report(symbol, is_override, recommendation, synthesis)
    if was_filled:
        return _format_fill_report(
            symbol, approved_trade, synthesis, verdict, is_override, recommendation
        )
    if hitl_status == "approved":
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


def _is_real_trade(qty: Any, notional: Any) -> bool:
    """True only when there's an actual sized trade (qty > 0 and notional > 0)."""
    try:
        return float(qty) > 0 and float(notional) > 0
    except (TypeError, ValueError):
        return False


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


def _is_override(hitl_decision: str | None) -> bool:
    return bool(hitl_decision) and str(hitl_decision).startswith("override:")


def _is_hold_decision(hitl_decision: str | None) -> bool:
    return hitl_decision == "override:hold"


def _trade_qty_price(approved_trade: dict[str, Any] | None) -> tuple[Any, Any]:
    """Pull (qty, requested_price) out of a serialised approved_trade, if present."""
    trade = (approved_trade or {}).get("trade")
    if not isinstance(trade, dict):
        return None, None
    return trade.get("qty"), trade.get("requested_price")


def _format_fill_report(
    symbol: str,
    approved_trade: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    is_override: bool,
    recommendation: str | None,
) -> str:
    qty, price = _trade_qty_price(approved_trade)
    side = _fill_side(approved_trade)
    qty_str = qty if qty is not None else "?"
    price_suffix = f" @ ${price}" if price is not None else ""

    if is_override:
        emoji = "🔴" if side == "sell" else "🟢"
        verb = "Sold" if side == "sell" else "Bought"
        head = f"{emoji} *Override → {verb} {qty_str} {symbol}{price_suffix}*"
    else:
        head = f"✅ *Approved → FILLED {qty_str} {symbol}{price_suffix}*"

    lines = [head]
    if is_override and recommendation:
        lines.append(f"_You overrode the desk's {recommendation} recommendation._")

    summary = _fmt_summary(synthesis)
    if summary:
        lines += ["", f"📝 {summary}"]

    score = _fmt_score(verdict)
    if score != "-":
        lines += [f"⚖️ Judge: {score}"]

    return "\n".join(lines)


def _fill_side(approved_trade: dict[str, Any] | None) -> str:
    trade = (approved_trade or {}).get("trade")
    if isinstance(trade, dict):
        return str(trade.get("side", "buy")).lower()
    return "buy"


def _format_held_report(
    symbol: str,
    is_override: bool,
    recommendation: str | None,
    synthesis: dict[str, Any] | None,
) -> str:
    if is_override:
        head = f"⏸ *Held (your override) — no trade for {symbol}.*"
        lines = [head]
        if recommendation:
            lines.append(f"_You overrode the desk's {recommendation} recommendation._")
    else:
        lines = [f"⏸ *Held — no trade for {symbol}.*"]
    summary = _fmt_summary(synthesis)
    if summary:
        lines += ["", f"📝 {summary}"]
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
