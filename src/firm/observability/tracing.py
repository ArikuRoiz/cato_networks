"""Correlation-id propagation and span helpers.

Span decorators (``trace_agent_call``, ``trace_tool_call``, ``with_span``,
``log_trade_event``) are no-ops when Langfuse is not configured — the
real tracing is done via ``@observe`` in the LLM adapter and CLI cycle.
The correlation-id context variable is always active.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id_var.get()


def set_correlation_id(cid: str) -> Token[str]:
    return _correlation_id_var.set(cid)


def reset_correlation_id(token: Token[str]) -> None:
    _correlation_id_var.reset(token)


@contextmanager
def with_span(
    name: str,
    *,
    correlation_id: str = "",
    **attrs: Any,
) -> Generator[None, None, None]:
    """No-op span context manager — kept for API compatibility."""
    yield


def trace_agent_call[F: Callable[..., Any]](agent_name: str) -> Callable[[F], F]:
    return lambda func: func


def trace_tool_call[F: Callable[..., Any]](tool_name: str) -> Callable[[F], F]:
    return lambda func: func


def log_trade_event(
    trade_id: str,
    action: str,
    correlation_id: str,
    **attrs: Any,
) -> None:
    effective_cid = correlation_id or get_correlation_id()
    logger.debug(
        "trade_event action=%s trade_id=%s correlation_id=%s extra=%s",
        action,
        trade_id,
        effective_cid,
        attrs,
    )
