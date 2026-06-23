"""``firm demo`` — replay Oct 23 2024 (NVDA earnings day) against frozen data.

Also hosts the offline cycle-loop helpers (``_run_graph_loop`` and friends)
shared with ``firm dev``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from firm.cli.output import _emit, _summarise
from firm.constants import DEFAULT_INITIAL_CASH, DEFAULT_WATCHLIST

if TYPE_CHECKING:
    from firm.config.settings import RiskPolicyConfig

logger = logging.getLogger(__name__)


def _cmd_demo(args: argparse.Namespace) -> None:
    """Replay Oct 23 2024 (NVDA earnings day) end-to-end, print trace."""
    from firm.cli.output import _project_root
    from firm.composition import build_offline_pipeline  # deferred: heavy import

    root = _project_root()
    demo_date = datetime(2024, 10, 23, tzinfo=UTC)
    watchlist = list(DEFAULT_WATCHLIST)

    _emit(
        {
            "event": "demo_start",
            "date": demo_date.date().isoformat(),
            "watchlist": watchlist,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
    )

    pipeline = build_offline_pipeline(root, initial_cash=DEFAULT_INITIAL_CASH)
    _run_graph_loop(pipeline.graph, watchlist, demo_date)
    _emit({"event": "demo_done", "ts": datetime.now(tz=UTC).isoformat()})


def _emit_cycle_done(
    symbol: str,
    correlation_id: str,
    final_state: dict[str, Any],
) -> None:
    """Emit a cycle_done NDJSON record from *final_state*."""
    verdict = final_state.get("verdict") or {}
    synthesis = final_state.get("synthesis") or {}
    _emit(
        {
            "event": "cycle_done",
            "symbol": symbol,
            "correlation_id": correlation_id,
            "outcome": final_state.get("cycle_outcome", "unknown"),
            "evidence": _summarise(final_state.get("evidence")),
            "trade_proposal": _summarise(final_state.get("trade_proposal")),
            "research_plan": _summarise(final_state.get("research_plan")),
            "technical_signal": _summarise(final_state.get("technical_signal")),
            "synthesis_title": synthesis.get("title", "none"),
            "judge_score": verdict.get("coherence_score", "none"),
            "judge_alignment": verdict.get("alignment", "none"),
        }
    )


def _invoke_one_symbol(
    graph: Any,
    symbol: str,
    decision_ts: datetime,
) -> None:
    """Run one graph cycle for *symbol*, emitting cycle_start + cycle_done/error."""
    import uuid

    from firm.observability.tracing import reset_correlation_id, set_correlation_id

    correlation_id = str(uuid.uuid4())
    _emit(
        {
            "event": "cycle_start",
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
    )
    # Propagate the correlation_id into the tracing context var so that the
    # TokenBudgetLLM wrapper can key token consumption to this cycle.
    token = set_correlation_id(correlation_id)
    try:
        initial_state = {
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
        config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
        try:
            from langfuse.decorators import langfuse_context, observe  # deferred: heavy import

            @observe(name=f"decision_cycle.{symbol}")  # type: ignore[untyped-decorator]
            def _traced_cycle() -> None:
                langfuse_context.update_current_observation(
                    input={"symbol": symbol, "decision_ts": decision_ts.isoformat()},
                    metadata={"correlation_id": correlation_id},
                )
                try:
                    final_state: dict[str, Any] = graph.invoke(initial_state, config=config)
                    outcome = final_state.get("cycle_outcome", "unknown")
                    langfuse_context.update_current_observation(output={"outcome": outcome})
                    _emit_cycle_done(symbol, correlation_id, final_state)
                except Exception as exc:
                    langfuse_context.update_current_observation(
                        level="ERROR", status_message=str(exc)
                    )
                    _emit(
                        {
                            "event": "cycle_error",
                            "symbol": symbol,
                            "correlation_id": correlation_id,
                            "error": str(exc),
                        }
                    )

            _traced_cycle()
        except ImportError as exc:
            logger.warning("Langfuse unavailable, running untraced: %s", exc)
            try:
                final_state = graph.invoke(initial_state, config=config)
                _emit_cycle_done(symbol, correlation_id, final_state)
            except Exception as graph_exc:
                _emit(
                    {
                        "event": "cycle_error",
                        "symbol": symbol,
                        "correlation_id": correlation_id,
                        "error": str(graph_exc),
                    }
                )
    finally:
        reset_correlation_id(token)


def _run_graph_loop(
    graph: Any,
    watchlist: list[str],
    decision_ts: datetime,
) -> None:
    """Invoke *graph* once per symbol in *watchlist*, emitting NDJSON events."""
    for symbol in watchlist:
        _invoke_one_symbol(graph, symbol, decision_ts)


def _safe_load_risk_policy(root: Path) -> RiskPolicyConfig:
    """Load risk policy from YAML, returning defaults on failure."""
    from firm.config.settings import load_risk_policy_or_default

    return load_risk_policy_or_default(root / "config" / "risk_policy.yaml")
