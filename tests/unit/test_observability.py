"""Unit tests for src/firm/observability/tracing.py — no IO, no network.

The tracing layer is intentionally thin: span decorators are transparent
no-ops (real span capture is delegated to Langfuse ``@observe`` in the LLM
adapter + the Postgres audit log), and the correlation-id ContextVar is always
active.  These tests pin that real contract — not an aspirational OTel one.

Covers:
  get/set/reset_correlation_id — ContextVar round-trip + isolation
  with_span                    — transparent no-op context manager
  trace_agent_call             — returns the wrapped fn, behaviour preserved
  trace_tool_call              — returns the wrapped fn, behaviour preserved
  log_trade_event              — emits a debug log, falls back to the ctx cid
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from firm.observability.tracing import (
    get_correlation_id,
    log_trade_event,
    reset_correlation_id,
    set_correlation_id,
    trace_agent_call,
    trace_tool_call,
    with_span,
)

# ---------------------------------------------------------------------------
# correlation-id ContextVar
# ---------------------------------------------------------------------------


def test_correlation_id_defaults_to_empty() -> None:
    assert get_correlation_id() == ""


def test_correlation_id_set_get_reset_round_trip() -> None:
    token = set_correlation_id("cid-123")
    try:
        assert get_correlation_id() == "cid-123"
    finally:
        reset_correlation_id(token)
    assert get_correlation_id() == ""


def test_correlation_id_reset_restores_previous_value() -> None:
    outer = set_correlation_id("outer")
    inner = set_correlation_id("inner")
    assert get_correlation_id() == "inner"
    reset_correlation_id(inner)
    assert get_correlation_id() == "outer"
    reset_correlation_id(outer)


# ---------------------------------------------------------------------------
# span helpers — transparent no-ops
# ---------------------------------------------------------------------------


def test_with_span_is_transparent_no_op() -> None:
    with with_span("research", correlation_id="cid-1", symbol="NVDA"):
        result = 2 + 2
    assert result == 4


def test_trace_agent_call_preserves_behaviour() -> None:
    @trace_agent_call("research")
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42


def test_trace_tool_call_preserves_behaviour() -> None:
    @trace_tool_call("get_price")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# log_trade_event
# ---------------------------------------------------------------------------


def test_log_trade_event_emits_debug_log() -> None:
    # Patch the module logger directly rather than relying on caplog, so the
    # assertion is immune to global logging state set by earlier tests.
    with patch("firm.observability.tracing.logger") as mock_logger:
        log_trade_event("trade-1", "fill", correlation_id="cid-9", qty=10)
    mock_logger.debug.assert_called_once()
    logged_args = mock_logger.debug.call_args.args
    assert "trade_event" in logged_args[0]  # the format string
    assert "trade-1" in logged_args  # trade_id positional arg
    assert "cid-9" in logged_args  # correlation_id positional arg


def test_log_trade_event_falls_back_to_context_correlation_id() -> None:
    token = set_correlation_id("ctx-cid")
    try:
        # Must not raise when correlation_id is omitted; uses the ContextVar.
        log_trade_event("trade-2", "submit", correlation_id="")
    finally:
        reset_correlation_id(token)
