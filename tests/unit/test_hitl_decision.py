"""Unit tests for the shared HITL resume interface (orchestration/hitl.py).

Covers ``parse_decision`` (the verb/short-code/legacy mapping and the unknown →
EXPIRE fail-safe) and ``resume_decision`` (the single graph-resume entry point
shared by the console and the bot), driving a fake graph so no LangGraph runtime
or network is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from firm.domain.enums import HITLStatus
from firm.orchestration.hitl import HITLDecision, parse_decision, resume_decision

# ---------------------------------------------------------------------------
# parse_decision — verbs, short codes, legacy strings, fail-safe
# ---------------------------------------------------------------------------


class TestParseDecision:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("approve", HITLDecision.APPROVE),
            ("buy", HITLDecision.OVERRIDE_BUY),
            ("sell", HITLDecision.OVERRIDE_SELL),
            ("hold", HITLDecision.OVERRIDE_HOLD),
        ],
    )
    def test_spec_verbs(self, raw: str, expected: HITLDecision) -> None:
        assert parse_decision(raw) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a", HITLDecision.APPROVE),
            ("b", HITLDecision.OVERRIDE_BUY),
            ("s", HITLDecision.OVERRIDE_SELL),
            ("h", HITLDecision.OVERRIDE_HOLD),
        ],
    )
    def test_short_codes(self, raw: str, expected: HITLDecision) -> None:
        assert parse_decision(raw) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("approved", HITLDecision.APPROVE),
            ("rejected", HITLDecision.OVERRIDE_HOLD),
            ("reject", HITLDecision.OVERRIDE_HOLD),
            ("expired", HITLDecision.EXPIRE),
        ],
    )
    def test_legacy_strings(self, raw: str, expected: HITLDecision) -> None:
        assert parse_decision(raw) is expected

    def test_canonical_enum_values_round_trip(self) -> None:
        assert parse_decision("override:buy") is HITLDecision.OVERRIDE_BUY
        assert parse_decision("expire") is HITLDecision.EXPIRE

    def test_case_and_whitespace_insensitive(self) -> None:
        assert parse_decision("  APPROVE  ") is HITLDecision.APPROVE

    def test_passthrough_of_existing_decision(self) -> None:
        assert parse_decision(HITLDecision.OVERRIDE_SELL) is HITLDecision.OVERRIDE_SELL

    def test_mapping_payload_unwrapped(self) -> None:
        assert parse_decision({"decision": "buy"}) is HITLDecision.OVERRIDE_BUY

    def test_unknown_string_fails_safe_to_expire(self) -> None:
        assert parse_decision("maybe") is HITLDecision.EXPIRE

    def test_none_fails_safe_to_expire(self) -> None:
        assert parse_decision(None) is HITLDecision.EXPIRE

    def test_unexpected_type_fails_safe_to_expire(self) -> None:
        assert parse_decision(42) is HITLDecision.EXPIRE


# ---------------------------------------------------------------------------
# HITLDecision predicates
# ---------------------------------------------------------------------------


class TestHITLDecisionPredicates:
    def test_overrides_are_overrides(self) -> None:
        for d in (
            HITLDecision.OVERRIDE_BUY,
            HITLDecision.OVERRIDE_SELL,
            HITLDecision.OVERRIDE_HOLD,
        ):
            assert d.is_override

    def test_approve_and_expire_are_not_overrides(self) -> None:
        assert not HITLDecision.APPROVE.is_override
        assert not HITLDecision.EXPIRE.is_override

    def test_hitl_status_mapping(self) -> None:
        assert HITLDecision.APPROVE.hitl_status == HITLStatus.APPROVED
        assert HITLDecision.OVERRIDE_BUY.hitl_status == HITLStatus.APPROVED
        assert HITLDecision.EXPIRE.hitl_status == HITLStatus.EXPIRED


# ---------------------------------------------------------------------------
# resume_decision — Command(resume=..., update=...) + stream-to-completion
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Captures the resume Command and replays a final-state stream."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.resume_arg: Any = None
        self.update_arg: Any = None
        self.config: Any = None

    def stream(self, cmd: Any, config: Any, stream_mode: str):  # type: ignore[no-untyped-def]
        self.resume_arg = cmd.resume
        self.update_arg = cmd.update
        self.config = config
        yield from self._events


class TestResumeDecision:
    def test_builds_command_with_decision_value_and_status(self) -> None:
        graph = _FakeGraph([{"cycle_outcome": "filled"}])
        final = resume_decision(graph, "tid-1", "buy")

        assert graph.resume_arg == HITLDecision.OVERRIDE_BUY.value
        assert graph.update_arg == {"hitl_status": HITLStatus.APPROVED}
        assert graph.config == {"configurable": {"thread_id": "tid-1"}}
        assert final == {"cycle_outcome": "filled"}

    def test_returns_last_streamed_state(self) -> None:
        graph = _FakeGraph([{"step": 1}, {"step": 2}, {"step": 3}])
        assert resume_decision(graph, "tid", "approve") == {"step": 3}

    def test_unknown_input_resumes_as_expire(self) -> None:
        graph = _FakeGraph([{}])
        resume_decision(graph, "tid", "garbage")
        assert graph.resume_arg == HITLDecision.EXPIRE.value
        assert graph.update_arg == {"hitl_status": HITLStatus.EXPIRED}

    def test_accepts_structured_decision_directly(self) -> None:
        graph = _FakeGraph([{}])
        resume_decision(graph, "tid", HITLDecision.OVERRIDE_SELL)
        assert graph.resume_arg == HITLDecision.OVERRIDE_SELL.value
