"""Domain guardrails — defense-in-depth enforcement at the ledger boundary.

All classes are pure domain: zero IO imports, no framework dependencies.
Exceptions represent hard safety violations; result unions surface recoverable
outcomes to callers without exception flow.
"""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar, NamedTuple

from firm.domain.decisions import Approved, HITLRequired, Rejected
from firm.domain.portfolio import Portfolio
from firm.domain.risk import RiskPolicy
from firm.domain.trade import Trade

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class LimitExceeded(Exception):
    """Raised when a trade breaches a hard risk limit at the ledger boundary."""


class TokenBudgetExceeded(Exception):
    """Raised when a correlation ID has consumed more tokens than its budget."""


# ---------------------------------------------------------------------------
# InjectionDetected — result type, not exception
# ---------------------------------------------------------------------------


class InjectionDetected(NamedTuple):
    """Potential prompt-injection detected in scanned text."""

    pattern: str
    text_preview: str  # first 120 chars for audit, never the full payload


# ---------------------------------------------------------------------------
# TokenBudgetCircuitBreaker
# ---------------------------------------------------------------------------


class TokenBudgetCircuitBreaker:
    """Track cumulative LLM token consumption per decision cycle.

    Each cycle is identified by its *correlation_id*.  When the running total
    exceeds the budget ``check_budget`` raises ``TokenBudgetExceeded`` so the
    orchestration layer can halt the cycle (fail-safe).
    """

    def __init__(self) -> None:
        self._budgets: dict[str, int] = {}

    def record_tokens(self, correlation_id: str, tokens: int) -> None:
        """Add *tokens* to the running total for *correlation_id*."""
        self._budgets[correlation_id] = self._budgets.get(correlation_id, 0) + tokens

    def check_budget(
        self,
        correlation_id: str,
        limit: int = 50_000,
    ) -> None:
        """Raise ``TokenBudgetExceeded`` if the total for *correlation_id* > *limit*.

        The default limit of 50 000 tokens is an implementation choice.
        Callers that need a different cap pass *limit* explicitly.
        """
        total = self._budgets.get(correlation_id, 0)
        if total > limit:
            raise TokenBudgetExceeded(
                f"Correlation {correlation_id!r} has consumed {total} tokens, "
                f"exceeding the budget of {limit}."
            )

    def get_total(self, correlation_id: str) -> int:
        """Return the cumulative token count for *correlation_id* (0 if unseen)."""
        return self._budgets.get(correlation_id, 0)


# ---------------------------------------------------------------------------
# InjectionGuard
# ---------------------------------------------------------------------------

_PREVIEW_LEN = 120


class InjectionGuard:
    """Detect and neutralise prompt-injection attempts in retrieved text.

    Corpus text is *data*, never instructions.  Any chunk containing a known
    injection phrase is flagged before it is forwarded to an LLM.
    """

    INJECTION_PATTERNS: ClassVar[tuple[str, ...]] = (
        "ignore instructions",
        "ignore previous",
        "execute trade",
        "bypass",
        "jailbreak",
    )

    def scan(self, text: str) -> str | InjectionDetected:
        """Return *text* unchanged, or ``InjectionDetected`` if a pattern is found.

        The scan is case-insensitive so ``BYPASS`` and ``Bypass`` are both caught.
        """
        lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if pattern in lower:
                preview = text[:_PREVIEW_LEN]
                return InjectionDetected(pattern=pattern, text_preview=preview)
        return text

    def sanitize(self, text: str) -> str:
        """Return a version of *text* with all injection patterns redacted.

        Each match is replaced with ``[REDACTED]`` so the surrounding context
        is preserved for audit while the instruction surface is neutralised.
        """
        result = text
        for pattern in self.INJECTION_PATTERNS:
            result = _replace_case_insensitive(result, pattern, "[REDACTED]")
        return result


def _replace_case_insensitive(text: str, pattern: str, replacement: str) -> str:
    """Replace all case-insensitive occurrences of *pattern* with *replacement*."""
    lower = text.lower()
    result_parts: list[str] = []
    search_start = 0
    pattern_len = len(pattern)
    while True:
        idx = lower.find(pattern, search_start)
        if idx == -1:
            result_parts.append(text[search_start:])
            break
        result_parts.append(text[search_start:idx])
        result_parts.append(replacement)
        search_start = idx + pattern_len
    return "".join(result_parts)


# ---------------------------------------------------------------------------
# LedgerGuardrail
# ---------------------------------------------------------------------------


class LedgerGuardrail:
    """Last-resort enforcement of ``RiskPolicy`` at the ledger write boundary.

    This guardrail is defense-in-depth: the Risk agent checks limits first;
    this class checks them again at the moment of write so that no code path
    can bypass the policy — even with agent + human approval.
    """

    def __init__(self, risk: RiskPolicy) -> None:
        self._risk = risk

    def enforce_before_write(
        self,
        trade: Trade,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        start_of_day_nav: Decimal | None = None,
    ) -> None:
        """Raise ``LimitExceeded`` when the trade would violate ``RiskPolicy``.

        Accepts *Approved* silently.  Converts *HITLRequired* and *Rejected*
        to ``LimitExceeded`` so the ledger write is unconditionally blocked —
        no trade reaches the database without passing every policy check.

        Pass *start_of_day_nav* to enforce the -3 % daily-loss halt (a LOCKED
        DECISION).  When omitted the daily-halt check is skipped (useful in
        tests that focus on per-trade limits only).

        **HITL note** — this method re-runs ``check_trade`` from scratch.  A
        trade that previously triggered ``HITLRequired`` will still be blocked
        here even if a human approved it, because the original oversized
        proposal is resubmitted unchanged.  Use ``enforce_hitl_approved`` when
        the trade has been explicitly approved by a human operator.
        """
        result = self._risk.check_trade(trade, portfolio, prices, start_of_day_nav)
        match result:
            case Approved():
                return
            case Rejected(reason=reason) | HITLRequired(reason=reason):
                raise LimitExceeded(
                    f"Ledger guardrail blocked trade {trade.id} ({trade.symbol}): {reason}"
                )

    def enforce_hitl_approved(
        self,
        trade: Trade,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        start_of_day_nav: Decimal | None = None,
    ) -> None:
        """Enforce risk limits for a human-approved (HITL) trade.

        Unlike ``enforce_before_write``, this method accepts ``HITLRequired``
        as passing — a human operator has already reviewed and approved the
        oversized proposal.  Hard limits (``Rejected``, e.g. daily halt,
        max_trade_notional_pct) still apply; only the HITL soft-gate is
        bypassed.

        Args:
            trade: The approved-but-oversized trade to validate.
            portfolio: Current portfolio state for NAV computation.
            prices: Current market prices keyed by symbol.
            start_of_day_nav: Optional; enables daily-halt enforcement.
        """
        result = self._risk.check_trade(trade, portfolio, prices, start_of_day_nav)
        match result:
            case Approved() | HITLRequired():
                return  # Human approved — HITL soft-gate is already satisfied
            case Rejected(reason=reason):
                raise LimitExceeded(
                    f"Ledger guardrail blocked HITL-approved trade "
                    f"{trade.id} ({trade.symbol}): {reason}"
                )
