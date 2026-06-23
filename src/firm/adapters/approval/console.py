"""Console approval channel — interactive stdin prompt.

Implements :class:`firm.ports.approval.ApprovalChannel`. Every cycle pauses for
the operator, who picks the ACTION directly: approve the recommendation, or
override with buy / sell / hold. Unrecognised / aborted input fails safe to a
hold override (a recorded human decision, but no trade).
"""

from __future__ import annotations

from firm.orchestration.hitl import HITLDecision
from firm.ports.types import HITLRequest

_PROMPT_KEYS: dict[str, HITLDecision] = {
    "a": HITLDecision.APPROVE,
    "approve": HITLDecision.APPROVE,
    "b": HITLDecision.OVERRIDE_BUY,
    "buy": HITLDecision.OVERRIDE_BUY,
    "s": HITLDecision.OVERRIDE_SELL,
    "sell": HITLDecision.OVERRIDE_SELL,
    "h": HITLDecision.OVERRIDE_HOLD,
    "hold": HITLDecision.OVERRIDE_HOLD,
}


class ConsoleApprovalChannel:
    """Block on stdin for a structured HITL decision."""

    def request_decision(self, request: HITLRequest) -> HITLDecision:
        print(_format_card(request), flush=True)
        try:
            raw = input("Decision (a/b/s/h) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "h"
        return _PROMPT_KEYS.get(raw, HITLDecision.OVERRIDE_HOLD)


def _format_card(request: HITLRequest) -> str:
    """Render the console approval card for *request*."""
    recommendation = request.recommendation or "?"
    return (
        f"\n[HITL] Decision required for {request.symbol} (every cycle pauses)\n"
        f"  Recommendation: {recommendation}\n"
        f"  Proposed trade: {request.side.upper()} {request.qty_str} "
        f"@ notional ${request.notional}\n"
        f"  Reason: {request.reason}\n"
        "  Options: [a]pprove  [b]uy  [s]ell  [h]old"
    )
