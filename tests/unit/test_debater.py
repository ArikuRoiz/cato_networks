"""Unit tests for the DebaterAgent.

Regression coverage for the live-run defect where the debater silently
reported itself "unavailable" (empty case), leaving the recommendation to
rest only on technicals. The debate must always contribute a substantive
case — even when fundamentals are thin or the LLM hiccups.

All tests use the in-memory FakeLLM — no network.
"""

from __future__ import annotations

from firm.adapters.fakes import FakeLLM
from firm.agents.debater.agent import DebaterAgent
from firm.agents.debater.schemas import DebaterCase, DebaterInput
from firm.ports.types import LLMError, LLMResponse

_SYMBOL = "NVDA"
_CID = "11111111-1111-1111-1111-111111111111"


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, input_tokens=10, output_tokens=20, model="haiku")


def _technical_only_input(stance: str = "bull") -> DebaterInput:
    """Thin-evidence cycle: technicals present, no fundamental claims."""
    return DebaterInput(
        symbol=_SYMBOL,
        round_num=1,
        correlation_id=_CID,
        stance=stance,  # type: ignore[arg-type]
        evidence_summary="Research found no usable claims.",
        technical_summary="Bias: bullish | RSI: 62.0 | Price riding the upper Bollinger band.",
        opponent_history=[],
    )


def test_debater_produces_nonempty_case_with_only_technical_context() -> None:
    """A clean LLM JSON response yields a substantive case from technicals alone."""
    fake = FakeLLM(
        responses=[
            _response(
                '{"argument":"Momentum favours the upside given the bullish bias.",'
                '"key_points":["RSI at 62 shows strength","Price riding upper band"]}'
            )
        ]
    )
    agent = DebaterAgent(llm=fake, stance="bull")

    result = agent.run(_technical_only_input("bull"))

    assert isinstance(result, DebaterCase)
    assert result.argument.strip()
    assert result.key_points


def test_fenced_json_is_parsed_not_treated_as_failure() -> None:
    """Markdown-fenced JSON (a common Haiku habit) must not become an empty case."""
    fake = FakeLLM(
        responses=[
            _response(
                "```json\n"
                '{"argument":"The bull case stands on momentum.",'
                '"key_points":["bullish bias"]}\n'
                "```"
            )
        ]
    )
    agent = DebaterAgent(llm=fake, stance="bull")

    result = agent.run(_technical_only_input("bull"))

    assert isinstance(result, DebaterCase)
    assert result.argument == "The bull case stands on momentum."
    assert result.key_points == ["bullish bias"]


def test_llm_error_falls_back_to_technical_grounded_case() -> None:
    """A transient LLM error still yields a usable case built from the signals."""

    class _ErroringLLM:
        def complete(self, messages, *, model, max_tokens):  # type: ignore[no-untyped-def]
            return LLMError(message="rate limited", retryable=True)

    agent = DebaterAgent(llm=_ErroringLLM(), stance="bear")

    result = agent.run(_technical_only_input("bear"))

    assert isinstance(result, DebaterCase)
    assert result.argument.strip()
    assert result.key_points
    # Fallback quotes the supplied technical signal — no fabricated numbers.
    assert "RSI: 62.0" in result.argument


def test_empty_argument_json_degrades_to_fallback() -> None:
    """Parseable JSON with empty content degrades to the deterministic fallback."""
    fake = FakeLLM(responses=[_response('{"argument":"","key_points":[]}')])
    agent = DebaterAgent(llm=fake, stance="bull")

    result = agent.run(_technical_only_input("bull"))

    assert isinstance(result, DebaterCase)
    assert result.argument.strip()
    assert result.key_points
