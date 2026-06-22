"""Unit tests for src/firm/observability/ — no IO, no network.

Covers:
  setup_telemetry     — provider wiring (in-memory exporter)
  with_span           — attribute setting, error recording, no-op tracer
  trace_agent_call    — sync and async decorator behaviour
  trace_tool_call     — sync and async decorator behaviour
  log_trade_event     — event emission on current span
  get/set/reset_correlation_id — ContextVar round-trip
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from firm.observability.tracing import (
    _build_event_attrs,
    get_correlation_id,
    log_trade_event,
    reset_correlation_id,
    set_correlation_id,
    trace_agent_call,
    trace_tool_call,
    with_span,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_and_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def span_exporter() -> Generator[InMemorySpanExporter, None, None]:
    """Patch ``firm.observability.tracing.trace.get_tracer`` to use an
    InMemorySpanExporter, keeping all other ``trace.*`` calls intact."""
    provider, exporter = _make_provider_and_exporter()
    import firm.observability.tracing as tracing_mod

    real_trace = tracing_mod.trace

    # Only override get_tracer; delegate get_current_span to the real module.
    def fake_get_tracer(name: str, *args: Any, **kwargs: Any) -> Any:
        return provider.get_tracer(name, *args, **kwargs)

    with patch.object(real_trace, "get_tracer", side_effect=fake_get_tracer):
        yield exporter
        exporter.clear()


@pytest.fixture(autouse=True)
def _reset_correlation_id_var() -> Generator[None, None, None]:
    """Ensure correlation_id is empty before and after each test."""
    token = set_correlation_id("")
    yield
    reset_correlation_id(token)


# ---------------------------------------------------------------------------
# ContextVar helpers
# ---------------------------------------------------------------------------


class TestCorrelationIdVar:
    def test_default_is_empty_string(self) -> None:
        assert get_correlation_id() == ""

    def test_set_returns_token_and_updates_value(self) -> None:
        token = set_correlation_id("cycle-123")
        assert get_correlation_id() == "cycle-123"
        reset_correlation_id(token)

    def test_reset_restores_previous_value(self) -> None:
        token = set_correlation_id("cycle-abc")
        reset_correlation_id(token)
        assert get_correlation_id() == ""

    def test_nested_set_and_reset(self) -> None:
        outer = set_correlation_id("outer")
        inner = set_correlation_id("inner")
        assert get_correlation_id() == "inner"
        reset_correlation_id(inner)
        assert get_correlation_id() == "outer"
        reset_correlation_id(outer)
        assert get_correlation_id() == ""


# ---------------------------------------------------------------------------
# setup_telemetry
# ---------------------------------------------------------------------------


class TestSetupTelemetry:
    def test_sets_global_tracer_provider(self) -> None:
        from opentelemetry import trace

        from firm.observability.setup import setup_telemetry

        setup_telemetry("test-svc")
        assert isinstance(trace.get_tracer_provider(), TracerProvider)

    def test_langfuse_skipped_when_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry import trace

        from firm.observability.setup import setup_telemetry

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        setup_telemetry("test-no-langfuse")
        assert isinstance(trace.get_tracer_provider(), TracerProvider)

    def test_langfuse_activated_when_both_keys_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        mock_langfuse_cls = MagicMock()
        fake_langfuse_module = MagicMock()
        fake_langfuse_module.Langfuse = mock_langfuse_cls

        with patch.dict("sys.modules", {"langfuse": fake_langfuse_module}):
            from importlib import reload

            import firm.observability.setup as setup_mod

            reload(setup_mod)
            setup_mod.setup_telemetry("test-langfuse")

        mock_langfuse_cls.assert_called_once()
        _, call_kwargs = mock_langfuse_cls.call_args
        assert call_kwargs["public_key"] == "pk-test"
        assert call_kwargs["secret_key"] == "sk-test"
        assert "tracer_provider" in call_kwargs


# ---------------------------------------------------------------------------
# with_span
# ---------------------------------------------------------------------------


class TestWithSpan:
    def test_span_created_with_correct_name(self, span_exporter: InMemorySpanExporter) -> None:
        with with_span("test.span"):
            pass
        spans = span_exporter.get_finished_spans()
        assert any(s.name == "test.span" for s in spans)

    def test_correlation_id_set_on_span(self, span_exporter: InMemorySpanExporter) -> None:
        with with_span("test.cid", correlation_id="cid-999"):
            pass
        span = span_exporter.get_finished_spans()[-1]
        assert span.attributes["correlation_id"] == "cid-999"

    def test_correlation_id_falls_back_to_context_var(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        token = set_correlation_id("ctx-var-cid")
        with with_span("test.fallback"):
            pass
        reset_correlation_id(token)
        span = span_exporter.get_finished_spans()[-1]
        assert span.attributes["correlation_id"] == "ctx-var-cid"

    def test_extra_attrs_attached_as_strings(self, span_exporter: InMemorySpanExporter) -> None:
        with with_span("test.attrs", symbol="AAPL", qty=100):
            pass
        span = span_exporter.get_finished_spans()[-1]
        assert span.attributes["symbol"] == "AAPL"
        assert span.attributes["qty"] == "100"

    def test_exception_is_recorded_and_reraised(self, span_exporter: InMemorySpanExporter) -> None:
        with pytest.raises(ValueError, match="boom"):
            with with_span("test.error"):
                raise ValueError("boom")
        span = span_exporter.get_finished_spans()[-1]
        assert span.status.status_code.name == "ERROR"
        event_names = [e.name for e in span.events]
        assert "exception" in event_names

    def test_yields_span_object(self, span_exporter: InMemorySpanExporter) -> None:
        with with_span("test.yield") as span:
            assert span is not None

    def test_noop_tracer_survives_without_exporter(self) -> None:
        """with_span must not raise even with no configured exporter."""
        with with_span("test.noop"):
            pass


# ---------------------------------------------------------------------------
# _build_event_attrs helper
# ---------------------------------------------------------------------------


class TestBuildEventAttrs:
    def test_base_fields_present(self) -> None:
        result = _build_event_attrs("t-1", "filled", "cid-1", {})
        assert result == {"trade_id": "t-1", "action": "filled", "correlation_id": "cid-1"}

    def test_extra_values_stringified(self) -> None:
        result = _build_event_attrs("t-2", "rejected", "cid-2", {"qty": 50, "price": 1.5})
        assert result["qty"] == "50"
        assert result["price"] == "1.5"


# ---------------------------------------------------------------------------
# log_trade_event
# ---------------------------------------------------------------------------


class TestLogTradeEvent:
    def test_event_emitted_on_current_span(self, span_exporter: InMemorySpanExporter) -> None:
        with with_span("trade.parent"):
            log_trade_event("t-42", "filled", "cid-42", symbol="NVDA")
        span = span_exporter.get_finished_spans()[-1]
        event_names = [e.name for e in span.events]
        assert "trade.filled" in event_names

    def test_event_attributes_include_trade_id_and_action(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        with with_span("trade.parent2"):
            log_trade_event("t-99", "rejected", "cid-99")
        span = span_exporter.get_finished_spans()[-1]
        event = next(e for e in span.events if e.name == "trade.rejected")
        assert event.attributes["trade_id"] == "t-99"
        assert event.attributes["correlation_id"] == "cid-99"

    def test_correlation_id_falls_back_to_context_var(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        token = set_correlation_id("fallback-cid")
        with with_span("trade.parent3"):
            log_trade_event("t-77", "hitl_requested", "")
        reset_correlation_id(token)
        span = span_exporter.get_finished_spans()[-1]
        event = next(e for e in span.events if e.name == "trade.hitl_requested")
        assert event.attributes["correlation_id"] == "fallback-cid"


# ---------------------------------------------------------------------------
# trace_agent_call — sync
# ---------------------------------------------------------------------------


class TestTraceAgentCallSync:
    def test_span_created_for_sync_agent(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_agent_call("research")
        def run() -> str:
            return "result"

        assert run() == "result"
        span = span_exporter.get_finished_spans()[-1]
        assert span.name == "agent.research"
        assert span.attributes["agent"] == "research"

    def test_sync_exception_propagates(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_agent_call("pm")
        def run() -> None:
            raise RuntimeError("pm error")

        with pytest.raises(RuntimeError, match="pm error"):
            run()
        span = span_exporter.get_finished_spans()[-1]
        assert span.status.status_code.name == "ERROR"

    def test_args_and_kwargs_forwarded(self) -> None:
        @trace_agent_call("exec")
        def run(a: int, b: str = "x") -> str:
            return f"{a}-{b}"

        assert run(1, b="y") == "1-y"


# ---------------------------------------------------------------------------
# trace_agent_call — async
# ---------------------------------------------------------------------------


class TestTraceAgentCallAsync:
    async def test_span_created_for_async_agent(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_agent_call("risk")
        async def run() -> str:
            return "async-result"

        result = await run()
        assert result == "async-result"
        span = span_exporter.get_finished_spans()[-1]
        assert span.name == "agent.risk"

    async def test_async_exception_propagates(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_agent_call("reporting")
        async def run() -> None:
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await run()
        span = span_exporter.get_finished_spans()[-1]
        assert span.status.status_code.name == "ERROR"

    async def test_returns_awaitable_not_plain_value(self) -> None:
        """Wrapping an async function must not collapse the coroutine."""
        import inspect as _inspect

        @trace_agent_call("langgraph_node")
        async def run() -> int:
            return 42

        coro = run()
        assert _inspect.iscoroutine(coro)
        result = await coro
        assert result == 42


# ---------------------------------------------------------------------------
# trace_tool_call — sync
# ---------------------------------------------------------------------------


class TestTraceToolCallSync:
    def test_span_created_for_sync_tool(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_tool_call("get_bar")
        def get_bar(symbol: str) -> str:
            return symbol

        assert get_bar("AAPL") == "AAPL"
        span = span_exporter.get_finished_spans()[-1]
        assert span.name == "tool.get_bar"
        assert span.attributes["tool"] == "get_bar"

    def test_sync_tool_exception_propagates(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_tool_call("embed")
        def embed() -> None:
            raise OSError("embed failed")

        with pytest.raises(OSError, match="embed failed"):
            embed()
        span = span_exporter.get_finished_spans()[-1]
        assert span.status.status_code.name == "ERROR"


# ---------------------------------------------------------------------------
# trace_tool_call — async
# ---------------------------------------------------------------------------


class TestTraceToolCallAsync:
    async def test_span_created_for_async_tool(self, span_exporter: InMemorySpanExporter) -> None:
        @trace_tool_call("fetch_news")
        async def fetch_news(ticker: str) -> list[str]:
            return [ticker]

        result = await fetch_news("NVDA")
        assert result == ["NVDA"]
        span = span_exporter.get_finished_spans()[-1]
        assert span.name == "tool.fetch_news"

    async def test_async_tool_exception_propagates(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        @trace_tool_call("rag_query")
        async def rag_query() -> None:
            raise LookupError("not found")

        with pytest.raises(LookupError):
            await rag_query()
