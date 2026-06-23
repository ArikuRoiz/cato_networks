"""Human-in-the-loop decision model and the clean resume interface.

NEW HITL model (every cycle pauses for a human):
    The risk node calls ``interrupt()`` on every cycle.  The human responds with
    a structured *decision* that names the ACTION to take, not merely
    approve/reject:

      - ``approve``        — execute the recommended action (buy/sell as sized;
                             hold → no trade).
      - ``override:buy``   — size and execute a buy now.
      - ``override:sell``  — sell the existing position for the symbol.
      - ``override:hold``  — no trade.

    Legacy values (``approved`` / ``rejected`` / ``expired``) are still accepted
    so existing callers and durability tests keep working: ``approved`` →
    approve, ``rejected`` → override:hold, ``expired`` → reject-timeout.

The resume interface (:func:`resume_decision`) is the single entry point that
both the console ``firm run`` path and the Telegram bot use to continue an
interrupted graph thread.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from firm.domain.enums import HITLStatus


class HITLDecision(StrEnum):
    """A structured human decision carried back into the risk node on resume."""

    APPROVE = "approve"
    OVERRIDE_BUY = "override:buy"
    OVERRIDE_SELL = "override:sell"
    OVERRIDE_HOLD = "override:hold"
    EXPIRE = "expire"

    @property
    def is_override(self) -> bool:
        return self in (
            HITLDecision.OVERRIDE_BUY,
            HITLDecision.OVERRIDE_SELL,
            HITLDecision.OVERRIDE_HOLD,
        )

    @property
    def hitl_status(self) -> HITLStatus:
        """Map the decision to the durable approval status.

        approve / any override → APPROVED (a human acted on the proposal);
        expire → EXPIRED (timed out, fail-safe).
        """
        if self is HITLDecision.EXPIRE:
            return HITLStatus.EXPIRED
        return HITLStatus.APPROVED


# Aliases → structured decision.  Covers the four spec verbs that name the
# ACTION directly (approve / buy / sell / hold), the short single-letter console
# codes, and the legacy approved/rejected/expired strings (back-compat for old
# callers and durability tests).
_LEGACY_DECISIONS: dict[str, HITLDecision] = {
    # Spec verbs: name the action directly.
    "approve": HITLDecision.APPROVE,
    "buy": HITLDecision.OVERRIDE_BUY,
    "sell": HITLDecision.OVERRIDE_SELL,
    "hold": HITLDecision.OVERRIDE_HOLD,
    # Short console codes.
    "a": HITLDecision.APPROVE,
    "b": HITLDecision.OVERRIDE_BUY,
    "s": HITLDecision.OVERRIDE_SELL,
    "h": HITLDecision.OVERRIDE_HOLD,
    # Legacy status strings.
    "approved": HITLDecision.APPROVE,
    "rejected": HITLDecision.OVERRIDE_HOLD,
    "reject": HITLDecision.OVERRIDE_HOLD,
    "expired": HITLDecision.EXPIRE,
}


def parse_decision(raw: object) -> HITLDecision:
    """Coerce a resume value into a :class:`HITLDecision`.

    Accepts a ``HITLDecision``, one of its string values, a legacy string
    (``approved``/``rejected``/``expired``), or a ``{"decision": ...}`` mapping
    as produced by the durability tests.  Unknown values fail safe to EXPIRE.
    """
    if isinstance(raw, HITLDecision):
        return raw
    if isinstance(raw, dict):
        return parse_decision(raw.get("decision"))
    if isinstance(raw, str):
        text = raw.strip().lower()
        try:
            return HITLDecision(text)
        except ValueError:
            return _LEGACY_DECISIONS.get(text, HITLDecision.EXPIRE)
    return HITLDecision.EXPIRE


def resume_decision(
    graph: Any,  # CompiledStateGraph — kept Any to avoid heavy LangGraph import at module load
    thread_id: str,
    decision: HITLDecision | str,
) -> dict[str, Any]:
    """Resume an interrupted graph thread with a structured human *decision*.

    Single entry point shared by the console ``firm run`` path and the bot.
    Builds ``Command(resume=<decision>, update={"hitl_status": ...})`` and
    streams the graph to completion, returning the final state values.

    *decision* may be a :class:`HITLDecision` or any string the
    :func:`parse_decision` helper understands.
    """
    from langgraph.types import Command

    parsed = parse_decision(decision)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    cmd: Any = Command(
        resume=parsed.value,
        update={"hitl_status": parsed.hitl_status},
    )
    final_state: dict[str, Any] = {}
    for event in graph.stream(cmd, config=config, stream_mode="values"):
        final_state = event
    return final_state
