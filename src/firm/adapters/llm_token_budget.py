"""Token-budget circuit-breaker wrapper for the LLM port.

Composes with any LLM implementation (GracefulLLM, CassetteLLM, FakeLLM, …).
Before each completion call it checks whether the current decision cycle has
already exceeded its token budget; if so it returns an ``LLMError`` immediately
so the calling agent degrades to its Failure variant — no crash, no hang.

After every successful call it records ``input_tokens + output_tokens`` into
the ``TokenBudgetCircuitBreaker``, keyed by the correlation_id read from the
tracing context var.  When the breaker trips it also calls
``report_sink.send_alert(...)`` if a sink is provided.

Composition order in NodePorts construction:
    TokenBudgetLLM(inner=GracefulLLM(CassetteLLM(replay_path)))
"""

from __future__ import annotations

import logging

from firm.domain.guardrails import TokenBudgetCircuitBreaker
from firm.observability.tracing import get_correlation_id
from firm.ports.llm import LLM
from firm.ports.report import ReportSink
from firm.ports.types import LLMError, LLMMessage, LLMResponse, ToolDef, ToolExecutors

logger = logging.getLogger(__name__)


class TokenBudgetLLM:
    """LLM wrapper that enforces a per-cycle token budget.

    Before each call: checks the breaker for the current correlation_id.
    After a successful call: records consumed tokens back into the breaker.
    On breach: returns LLMError so the calling agent degrades gracefully.
    """

    def __init__(
        self,
        inner: LLM,
        breaker: TokenBudgetCircuitBreaker,
        budget: int,
        report_sink: ReportSink | None = None,
    ) -> None:
        self._inner = inner
        self._breaker = breaker
        self._budget = budget
        self._report_sink = report_sink

    # ------------------------------------------------------------------
    # LLM Protocol implementation
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        over_budget = self._check_budget_before_call()
        if over_budget is not None:
            return over_budget
        result = self._inner.complete(messages, model=model, max_tokens=max_tokens)
        self._record_if_response(result)
        return result

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDef],
        executors: ToolExecutors,
        *,
        model: str,
        max_tokens: int,
        max_rounds: int = 5,
    ) -> LLMResponse | LLMError:
        over_budget = self._check_budget_before_call()
        if over_budget is not None:
            return over_budget
        result = self._inner.complete_with_tools(
            messages,
            tools,
            executors,
            model=model,
            max_tokens=max_tokens,
            max_rounds=max_rounds,
        )
        self._record_if_response(result)
        return result

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        return self._inner.count_tokens(messages, model=model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_cid(self) -> str:
        return get_correlation_id()

    def _check_budget_before_call(self) -> LLMError | None:
        """Return an LLMError when the cycle is already over budget; else None."""
        cid = self._current_cid()
        total = self._breaker.get_total(cid)
        if total > self._budget:
            msg = f"token budget exceeded for {cid!r} ({total} > {self._budget})"
            logger.warning("TokenBudgetLLM: %s", msg)
            self._maybe_alert(cid, total)
            return LLMError(message=msg, retryable=False)
        return None

    def _record_if_response(self, result: LLMResponse | LLMError) -> None:
        """Record tokens from a successful response into the breaker."""
        if isinstance(result, LLMResponse):
            cid = self._current_cid()
            self._breaker.record_tokens(cid, result.input_tokens + result.output_tokens)

    def _maybe_alert(self, cid: str, total: int) -> None:
        """Send an alert via report_sink when the breaker trips, if available."""
        if self._report_sink is None:
            return
        try:
            self._report_sink.send_alert(
                message=(
                    f"Token budget exceeded for cycle {cid!r}: "
                    f"{total} tokens consumed (budget: {self._budget}). "
                    "Remaining LLM calls in this cycle will be rejected."
                ),
                correlation_id=cid,
            )
        except Exception:
            logger.exception("TokenBudgetLLM: failed to send alert for %s", cid)
