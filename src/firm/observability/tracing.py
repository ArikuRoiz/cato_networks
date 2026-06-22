"""Correlation-id propagation and span helpers for the AI Investment Firm.

Usage pattern
-------------
Every decision cycle is assigned a ``correlation_id`` (the ``DecisionCycle.id``
as a string).  Set it once at cycle entry::

    token = set_correlation_id(str(cycle.id))
    try:
        ...
    finally:
        reset_correlation_id(token)

Every span opened inside the cycle inherits the id automatically via
``get_correlation_id()``.  The ``trace TRADE=<id>`` Make target uses this
attribute to reconstruct the full audit log for a single trade.

Decorator helpers
-----------------
``@trace_agent_call("research")`` wraps ``run()`` with a span named
``agent.research``; ``@trace_tool_call("get_bar")`` wraps any function with
``tool.get_bar``.  Both pass through all positional/keyword arguments unchanged
and surface exceptions after recording them on the span.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correlation-id context variable
# ---------------------------------------------------------------------------

_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the correlation_id for the current async/thread context."""
    return _correlation_id_var.get()


def set_correlation_id(cid: str) -> Token[str]:
    """Set the correlation_id for the current context; returns a reset token."""
    return _correlation_id_var.set(cid)


def reset_correlation_id(token: Token[str]) -> None:
    """Restore the correlation_id to the value before the matching ``set`` call."""
    _correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# Span context manager
# ---------------------------------------------------------------------------


def _attach_attrs(span: Span, correlation_id: str, attrs: dict[str, Any]) -> None:
    """Set correlation_id and extra string attributes on *span*."""
    span.set_attribute("correlation_id", correlation_id)
    for key, value in attrs.items():
        span.set_attribute(key, str(value))


def _record_exception(span: Span, exc: Exception) -> None:
    """Mark *span* as error and record the exception details."""
    span.set_status(StatusCode.ERROR, str(exc))
    span.record_exception(exc)


@contextmanager
def with_span(
    name: str,
    *,
    correlation_id: str = "",
    **attrs: Any,
) -> Generator[Span, None, None]:
    """Open a span named *name*, attach correlation_id + extra attributes.

    The span is opened on the tracer named ``"firm"``; if no ``TracerProvider``
    has been configured (e.g. in plain unit tests) OTel falls back to a no-op
    tracer so no network calls occur.

    Yields:
        The active ``Span`` so callers can add events or set attributes.
    """
    tracer = trace.get_tracer("firm")
    effective_cid = correlation_id or get_correlation_id()
    with tracer.start_as_current_span(name) as span:
        _attach_attrs(span, effective_cid, attrs)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            raise


# ---------------------------------------------------------------------------
# Decorator factories
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])


def _make_span_wrapper(span_name: str, **span_attrs: str) -> Callable[[_F], _F]:
    """Build a sync-or-async decorator that wraps *func* in a span named *span_name*."""

    def decorator(func: _F) -> _F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with with_span(span_name, **span_attrs):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with with_span(span_name, **span_attrs):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def trace_agent_call(agent_name: str) -> Callable[[_F], _F]:
    """Return a decorator that wraps ``run()`` in a span named ``agent.<agent_name>``.

    Supports both sync and async callables (LangGraph nodes are often async).

    Args:
        agent_name: Short identifier, e.g. ``"research"`` or ``"portfolio_manager"``.
    """
    return _make_span_wrapper(f"agent.{agent_name}", agent=agent_name)


def trace_tool_call(tool_name: str) -> Callable[[_F], _F]:
    """Return a decorator that wraps any function in a span named ``tool.<tool_name>``.

    Supports both sync and async callables.

    Args:
        tool_name: Short identifier, e.g. ``"get_bar"`` or ``"embed_chunks"``.
    """
    return _make_span_wrapper(f"tool.{tool_name}", tool=tool_name)


# ---------------------------------------------------------------------------
# Trade-event helper
# ---------------------------------------------------------------------------


def _build_event_attrs(
    trade_id: str,
    action: str,
    correlation_id: str,
    extra: dict[str, Any],
) -> dict[str, str]:
    """Assemble OTel event attribute dict from trade lifecycle fields."""
    base: dict[str, str] = {
        "trade_id": trade_id,
        "action": action,
        "correlation_id": correlation_id,
    }
    base.update({k: str(v) for k, v in extra.items()})
    return base


def log_trade_event(
    trade_id: str,
    action: str,
    correlation_id: str,
    **attrs: Any,
) -> None:
    """Emit a structured OTel event on the current span for a trade lifecycle action.

    This is the anchor point for ``make trace TRADE=<id>``: every fill, reject,
    HITL pause, and error emits an event here.

    Args:
        trade_id: The ``Trade.id`` UUID as a string.
        action: A short past-tense verb, e.g. ``"filled"`` or ``"rejected"``.
        correlation_id: Falls back to the context-var value when empty.
        **attrs: Additional key/value pairs included in the event attributes.
    """
    effective_cid = correlation_id or get_correlation_id()
    event_attrs = _build_event_attrs(trade_id, action, effective_cid, attrs)
    span = trace.get_current_span()
    span.add_event(f"trade.{action}", attributes=event_attrs)
    logger.debug(
        "trade_event action=%s trade_id=%s correlation_id=%s extra=%s",
        action,
        trade_id,
        effective_cid,
        attrs,
    )
