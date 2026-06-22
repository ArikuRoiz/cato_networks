"""Unit tests for the cassette LLM adapter (FIRM-10).

All tests are pure-Python — no IO beyond a tmp_path filesystem, no network.

Covers:
- record → replay round-trip produces identical responses
- inner LLM is never called during replay
- CassetteNotFound raised for unknown requests in replay mode
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from firm.adapters.fakes import FakeLLM
from firm.adapters.llm_cassette import CassetteLLM, CassetteNotFound
from firm.ports.types import LLMError, LLMMessage, LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(content: str, role: Literal["user", "assistant", "system"] = "user") -> LLMMessage:
    return LLMMessage(role=role, content=content)


def _response(content: str = "ok", model: str = "test-model") -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        model=model,
    )


def _fake_llm(*responses: LLMResponse) -> FakeLLM:
    return FakeLLM(responses=list(responses))


# ---------------------------------------------------------------------------
# test_cassette_record_replay
# ---------------------------------------------------------------------------


def test_cassette_record_replay(tmp_path: Path) -> None:
    """Record two completions then replay both and verify identical responses."""
    cassette = tmp_path / "test.jsonl"

    r1 = _response("answer one", model="haiku")
    r2 = _response("answer two", model="haiku")
    inner = _fake_llm(r1, r2)

    # --- record phase ---
    recorder = CassetteLLM(cassette, mode="record", inner=inner)
    msgs1 = [_msg("question one")]
    msgs2 = [_msg("question two")]

    recorded1 = recorder.complete(msgs1, model="haiku", max_tokens=100)
    recorded2 = recorder.complete(msgs2, model="haiku", max_tokens=100)

    assert isinstance(recorded1, LLMResponse)
    assert isinstance(recorded2, LLMResponse)
    assert recorded1.content == "answer one"
    assert recorded2.content == "answer two"

    # Cassette file must have exactly 2 JSONL lines.
    lines = [ln for ln in cassette.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2

    # --- replay phase ---
    replayer = CassetteLLM(cassette, mode="replay")
    replayed1 = replayer.complete(msgs1, model="haiku", max_tokens=100)
    replayed2 = replayer.complete(msgs2, model="haiku", max_tokens=100)

    assert isinstance(replayed1, LLMResponse)
    assert isinstance(replayed2, LLMResponse)
    assert replayed1 == recorded1
    assert replayed2 == recorded2


# ---------------------------------------------------------------------------
# test_cassette_no_network_in_replay
# ---------------------------------------------------------------------------


def test_cassette_no_network_in_replay(tmp_path: Path) -> None:
    """Inner LLM is never called (complete or count_tokens) while replaying."""
    cassette = tmp_path / "no_network.jsonl"
    msgs = [_msg("some prompt")]

    # Record both a complete() and a count_tokens() entry.
    inner_record = _fake_llm(_response("cached"))
    recorder = CassetteLLM(cassette, mode="record", inner=inner_record)
    recorder.complete(msgs, model="sonnet", max_tokens=50)
    recorder.count_tokens(msgs, model="sonnet")

    # Now replay — supply an inner that would raise if called.
    class _NeverCallLLM:
        def complete(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
            max_tokens: int,
        ) -> LLMResponse | LLMError:
            raise AssertionError("inner LLM must not be called during replay")

        def count_tokens(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
        ) -> int:
            raise AssertionError("inner LLM must not be called during replay")

    replayer = CassetteLLM(cassette, mode="replay", inner=_NeverCallLLM())
    result = replayer.complete(msgs, model="sonnet", max_tokens=50)
    token_count = replayer.count_tokens(msgs, model="sonnet")

    assert isinstance(result, LLMResponse)
    assert result.content == "cached"
    assert isinstance(token_count, int)


# ---------------------------------------------------------------------------
# test_cassette_not_found
# ---------------------------------------------------------------------------


def test_cassette_not_found(tmp_path: Path) -> None:
    """Replaying a request not in the cassette raises CassetteNotFound."""
    cassette = tmp_path / "sparse.jsonl"

    # Record a single entry.
    inner = _fake_llm(_response("known"))
    CassetteLLM(cassette, mode="record", inner=inner).complete(
        [_msg("known prompt")], model="haiku", max_tokens=50
    )

    # Replay a different prompt that was never recorded.
    replayer = CassetteLLM(cassette, mode="replay")
    with pytest.raises(CassetteNotFound) as exc_info:
        replayer.complete([_msg("unknown prompt")], model="haiku", max_tokens=50)

    assert exc_info.value.key  # key attribute is populated


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_cassette_record_requires_inner(tmp_path: Path) -> None:
    """Constructing CassetteLLM in record mode without inner raises ValueError."""
    with pytest.raises(ValueError, match="inner LLM"):
        CassetteLLM(tmp_path / "x.jsonl", mode="record")


def test_cassette_replay_empty_cassette_raises(tmp_path: Path) -> None:
    """Replaying from a non-existent (empty) cassette raises CassetteNotFound."""
    replayer = CassetteLLM(tmp_path / "empty.jsonl", mode="replay")
    with pytest.raises(CassetteNotFound):
        replayer.complete([_msg("anything")], model="haiku", max_tokens=50)


def test_cassette_records_llm_error(tmp_path: Path) -> None:
    """An LLMError from the inner LLM is recorded and replayed faithfully."""
    cassette = tmp_path / "errors.jsonl"
    error = LLMError(message="rate limited", retryable=True)

    class _ErrorLLM:
        def complete(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
            max_tokens: int,
        ) -> LLMResponse | LLMError:
            return error

        def count_tokens(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
        ) -> int:
            return 0

    recorder = CassetteLLM(cassette, mode="record", inner=_ErrorLLM())
    recorded = recorder.complete([_msg("q")], model="haiku", max_tokens=10)
    assert isinstance(recorded, LLMError)
    assert recorded.retryable is True

    replayer = CassetteLLM(cassette, mode="replay")
    replayed = replayer.complete([_msg("q")], model="haiku", max_tokens=10)
    assert isinstance(replayed, LLMError)
    assert replayed == error


def test_cassette_same_request_same_key(tmp_path: Path) -> None:
    """Identical requests with the same model produce the same cache key (idempotent)."""
    cassette = tmp_path / "idempotent.jsonl"
    msgs = [_msg("repeat me")]
    inner = _fake_llm(_response("once"))

    recorder = CassetteLLM(cassette, mode="record", inner=inner)
    recorder.complete(msgs, model="haiku", max_tokens=10)

    # Replaying twice with the exact same inputs must return the same entry.
    replayer = CassetteLLM(cassette, mode="replay")
    r1 = replayer.complete(msgs, model="haiku", max_tokens=10)
    r2 = replayer.complete(msgs, model="haiku", max_tokens=10)
    assert r1 == r2


# ---------------------------------------------------------------------------
# count_tokens record → replay round-trip (issues 3 & 4)
# ---------------------------------------------------------------------------


def test_cassette_count_tokens_record_replay(tmp_path: Path) -> None:
    """count_tokens round-trip: recorded value is served from cassette in replay."""
    cassette = tmp_path / "tokens.jsonl"
    msgs = [_msg("count these tokens")]
    inner = _fake_llm(_response("irrelevant"))  # complete() responses not used here

    recorder = CassetteLLM(cassette, mode="record", inner=inner)
    recorded_count = recorder.count_tokens(msgs, model="haiku")

    # The cassette file must contain a token_count entry.
    lines = [ln for ln in cassette.read_text().splitlines() if ln.strip()]
    assert any("token_count" in ln for ln in lines)

    replayer = CassetteLLM(cassette, mode="replay")
    replayed_count = replayer.count_tokens(msgs, model="haiku")

    assert replayed_count == recorded_count


def test_cassette_count_tokens_not_found_in_replay(tmp_path: Path) -> None:
    """count_tokens for a pair not in the cassette raises CassetteNotFound."""
    cassette = tmp_path / "sparse_tokens.jsonl"

    # Record one complete() entry only — no count_tokens entry written.
    inner = _fake_llm(_response("known"))
    CassetteLLM(cassette, mode="record", inner=inner).complete(
        [_msg("known prompt")], model="haiku", max_tokens=50
    )

    replayer = CassetteLLM(cassette, mode="replay")
    with pytest.raises(CassetteNotFound) as exc_info:
        replayer.count_tokens([_msg("unknown prompt")], model="haiku")

    assert exc_info.value.key  # key attribute is populated


def test_cassette_count_tokens_independent_of_error_entry(tmp_path: Path) -> None:
    """count_tokens in replay does not piggyback on error complete() entries.

    Previously, a complete() that recorded an LLMError would cause count_tokens
    for the same messages to silently fail with CassetteNotFound(key) and a
    confusing key.  With the dedicated token-count namespace, the failure is
    transparent: count_tokens simply has no entry in the cassette.
    """
    cassette = tmp_path / "error_entry.jsonl"
    error = LLMError(message="rate limited", retryable=True)

    class _ErrorLLM:
        def complete(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
            max_tokens: int,
        ) -> LLMResponse | LLMError:
            return error

        def count_tokens(
            self,
            messages: list[LLMMessage],
            *,
            model: str,
        ) -> int:
            return 42

    msgs = [_msg("fail me")]
    recorder = CassetteLLM(cassette, mode="record", inner=_ErrorLLM())
    recorder.complete(msgs, model="haiku", max_tokens=10)
    # No count_tokens call was recorded — replay must raise CassetteNotFound,
    # not return a stale or wrong value.
    replayer = CassetteLLM(cassette, mode="replay")
    with pytest.raises(CassetteNotFound) as exc_info:
        replayer.count_tokens(msgs, model="haiku")
    assert exc_info.value.key
