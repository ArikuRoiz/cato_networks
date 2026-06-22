"""OpenTelemetry + Langfuse setup and correlation-id propagation across all agent spans.

Public surface
--------------
``setup_telemetry``      — call once at process start to wire the global TracerProvider.
``with_span``            — context manager: open a named span with correlation_id.
``trace_agent_call``     — decorator factory for agent ``run()`` methods.
``trace_tool_call``      — decorator factory for domain tool functions.
``log_trade_event``      — emit a structured OTel event on the current span.
``get_correlation_id``   — read the cycle id from the current context.
``set_correlation_id``   — set the cycle id (returns a reset Token).
``reset_correlation_id`` — restore the previous value via a Token.
"""

from firm.observability.setup import flush_telemetry, setup_telemetry
from firm.observability.tracing import (
    get_correlation_id,
    log_trade_event,
    reset_correlation_id,
    set_correlation_id,
    trace_agent_call,
    trace_tool_call,
    with_span,
)

__all__ = [
    "flush_telemetry",
    "get_correlation_id",
    "log_trade_event",
    "reset_correlation_id",
    "set_correlation_id",
    "setup_telemetry",
    "trace_agent_call",
    "trace_tool_call",
    "with_span",
]
