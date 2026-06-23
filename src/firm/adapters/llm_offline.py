"""Offline LLM selector and graceful cassette wrapper.

Public API
----------
GracefulLLM
    Wraps any LLM; converts ``CassetteNotFound`` into ``LLMError`` so agents
    degrade gracefully on a cassette miss — never crash, never hang.

build_offline_llm(cassette_path) -> LLM
    Single shared selector used by both the CLI demo and the eval harness.
    The ONLY path that touches the network is ``CASSETTE_MODE=record``.
    A bare ``ANTHROPIC_API_KEY`` never selects a live path.

Selection rules (evaluated top-to-bottom):
    1. ``CASSETTE_MODE=record``     → CassetteLLM(record) wrapping AnthropicLLM
    2. cassette exists and non-empty → GracefulLLM(CassetteLLM(replay))
    3. default                      → FakeLLM (offline, deterministic)
"""

from __future__ import annotations

import os
from pathlib import Path

from firm.adapters.llm_cassette import CassetteLLM, CassetteNotFound
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, LLMResponse, ToolDef, ToolExecutors

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class GracefulLLM(LLM):
    """Wraps an inner LLM and catches ``CassetteNotFound`` on replay miss.

    When the cassette has no entry for a request, the miss is silently
    converted to ``LLMError(retryable=False)`` so the calling agent returns
    its Failure variant instead of raising.  ``count_tokens`` falls back to a
    character-based estimate rather than raising.
    """

    def __init__(self, inner: LLM) -> None:
        self._inner = inner

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        try:
            return self._inner.complete(messages, model=model, max_tokens=max_tokens)
        except CassetteNotFound as exc:
            return LLMError(message=f"cassette miss: {exc.key[:16]}", retryable=False)

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
        try:
            return self._inner.complete_with_tools(
                messages,
                tools,
                executors,
                model=model,
                max_tokens=max_tokens,
                max_rounds=max_rounds,
            )
        except CassetteNotFound as exc:
            return LLMError(message=f"cassette miss: {exc.key[:16]}", retryable=False)

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        try:
            return self._inner.count_tokens(messages, model=model)
        except CassetteNotFound:
            return sum(len(m.content) for m in messages) // 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_offline_llm(cassette_path: Path) -> LLM:
    """Return the appropriate LLM for offline-first operation.

    Selection order (first match wins):

    1. ``CASSETTE_MODE=record`` — explicit opt-in to network; wraps
       ``AnthropicLLM`` in a recording ``CassetteLLM``.  The API key must be
       set; if absent a ``ValueError`` is raised so the misconfiguration is
       immediately visible.

    2. Cassette exists and is non-empty — offline replay via
       ``GracefulLLM(CassetteLLM(replay))``.  Misses degrade to ``LLMError``
       rather than crashing.

    3. Default — fully offline ``FakeLLM`` that returns ``"[]"`` for every
       call.  Used by CI and the demo when no cassette is present.

    A bare ``ANTHROPIC_API_KEY`` in the environment does **not** select a live
    path; only ``CASSETTE_MODE=record`` does.
    """
    cassette_mode = os.environ.get("CASSETTE_MODE", "")

    if cassette_mode == "record":
        return _build_record_llm(cassette_path)

    if _cassette_is_usable(cassette_path):
        return GracefulLLM(CassetteLLM(cassette_path=cassette_path, mode="replay"))

    return _build_fake_llm()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _cassette_is_usable(cassette_path: Path) -> bool:
    return cassette_path.exists() and cassette_path.stat().st_size > 0


def _build_record_llm(cassette_path: Path) -> CassetteLLM:
    """Return a recording CassetteLLM backed by AnthropicLLM.

    Raises ``ValueError`` when ``ANTHROPIC_API_KEY`` is absent so the
    misconfiguration surfaces immediately rather than at the first LLM call.
    """
    from firm.adapters.llm_anthropic import AnthropicLLM

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("CASSETTE_MODE=record requires ANTHROPIC_API_KEY to be set")
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    inner = AnthropicLLM(api_key=api_key)
    return CassetteLLM(cassette_path=cassette_path, mode="record", inner=inner)


def _build_fake_llm() -> LLM:
    from firm.adapters.fakes import FakeLLM

    canned = LLMResponse(
        content="[]",
        input_tokens=10,
        output_tokens=2,
        model="claude-haiku-4-5",
    )
    return FakeLLM(responses=[canned] * 500)
