"""``firm run`` — LIVE production run against real market data + Postgres."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from firm.cli.commands.demo import _emit_cycle_done
from firm.cli.output import _emit, _load_settings
from firm.constants import DEFAULT_INITIAL_CASH, DEFAULT_WATCHLIST

if TYPE_CHECKING:
    from firm.config.settings import Settings


def _cmd_run(args: argparse.Namespace) -> None:
    """Live production run: real market data + news + Postgres + live LLM.

    Prerequisites (not met in offline CI):
      - ``make up``   — Postgres + pgvector running in Docker.
      - ``make seed`` — migrations applied, tables exist.
      - ``.env``      — ANTHROPIC_API_KEY and DATABASE_URL set.
    """
    from firm.cli.output import _project_root
    from firm.composition import build_live_pipeline  # deferred: heavy import

    root = _project_root()
    tickers = _parse_tickers(args.tickers)
    lookback_days: int = args.lookback_days
    hitl_channel: str = getattr(args, "hitl", "auto")
    force_buy: bool = getattr(args, "force_buy", False)

    _emit(
        {
            "event": "run_start",
            "tickers": tickers,
            "lookback_days": lookback_days,
            "hitl_channel": hitl_channel,
            "force_buy": force_buy,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
    )

    settings = _load_settings()
    _validate_live_settings(settings)

    _ingest_live_news(tickers, lookback_days, settings)

    pipeline = build_live_pipeline(settings, root=root, initial_cash=DEFAULT_INITIAL_CASH)
    decision_ts = datetime.now(tz=UTC)
    _run_live_graph_loop(pipeline.graph, tickers, decision_ts, settings, hitl_channel, force_buy)

    _emit({"event": "run_done", "ts": datetime.now(tz=UTC).isoformat()})


def _parse_tickers(tickers_arg: str | None) -> list[str]:
    """Parse comma-separated tickers from CLI arg; fall back to default watchlist."""
    if not tickers_arg:
        return list(DEFAULT_WATCHLIST)
    return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]


def _validate_live_settings(settings: Settings) -> None:
    """Raise SystemExit with a clear message when required env vars are missing."""
    if not settings.has_anthropic_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to .env or export it before running 'firm run'."
        )


def _ingest_live_news(tickers: list[str], lookback_days: int, settings: Settings) -> None:
    """Fetch recent headlines for *tickers* and upsert them into pgvector.

    On failure (yfinance error, DB connectivity, etc.) a warning is emitted and
    the run continues — stale evidence is better than aborting the entire cycle.
    """
    import uuid

    import psycopg  # deferred: heavy import

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.agents.news_ingestion import NewsIngestionAgent
    from firm.agents.news_ingestion.schemas import NewsIngestionInput
    from firm.orchestration.checkpointer import _normalise_database_url

    _emit({"event": "news_ingestion", "status": "starting", "tickers": tickers})
    try:
        url = _normalise_database_url(settings.database_url)
        with psycopg.connect(url) as conn:
            store = PgvectorEvidenceStore(conn)
            agent = NewsIngestionAgent(evidence=store)
            inp = NewsIngestionInput(
                symbols=tickers,
                lookback_hours=lookback_days * 24,
                correlation_id=str(uuid.uuid4()),
            )
            result = agent.run(inp)
            conn.commit()
        _emit({"event": "news_ingestion", "status": "ok", "result": result.model_dump()})
    except Exception as exc:
        _emit({"event": "news_ingestion", "status": "warning", "error": str(exc)})


def _run_live_graph_loop(
    graph: Any,
    tickers: list[str],
    decision_ts: datetime,
    settings: Settings,
    hitl_channel: str = "auto",
    force_buy: bool = False,
) -> None:
    """Invoke the graph once per ticker, handling HITL interrupts.

    The HITL surface is selected by *hitl_channel* via the approval-channel
    registry (``console`` / ``telegram`` / ``slack`` / ``auto``); ``auto`` uses
    Telegram when configured, else console.
    """
    from firm.adapters.approval import build_approval_channel

    channel = build_approval_channel(hitl_channel, settings)
    for symbol in tickers:
        _invoke_live_symbol(graph, symbol, decision_ts, settings, channel, force_buy)


def _invoke_live_symbol(
    graph: Any,
    symbol: str,
    decision_ts: datetime,
    settings: Settings,
    channel: Any = None,
    force_buy: bool = False,
) -> None:
    """Stream the graph for one symbol; block on HITL interrupts.

    On a HITL interrupt the *channel* (an ``ApprovalChannel``) is asked for a
    structured decision, which is resumed through the shared ``resume_decision``
    entry point.  *force_buy* injects a synthetic BUY plan (demo override) so a
    trade > 5% NAV reliably fires the HITL interrupt.
    """
    import uuid

    from firm.observability.tracing import reset_correlation_id, set_correlation_id
    from firm.orchestration.hitl import resume_decision

    correlation_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    _emit(
        {
            "event": "cycle_start",
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
    )
    token = set_correlation_id(correlation_id)
    try:
        initial_state: dict[str, Any] = {
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
        if force_buy:
            initial_state["force_buy"] = True

        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        # Stream events; stop on the first interrupt or when the graph ends.
        final_state: dict[str, Any] = {}
        while True:
            for event in graph.stream(initial_state, config=config, stream_mode="values"):
                final_state = event
            # Check whether the graph halted on an interrupt by querying state.
            run_state = graph.get_state(config)
            interrupted = False
            interrupt_payload: dict[str, Any] = {}
            if run_state.next and run_state.tasks:
                for task in run_state.tasks:
                    if getattr(task, "interrupts", None):
                        interrupted = True
                        interrupt_payload = task.interrupts[0].value if task.interrupts else {}
                        break

            if not interrupted:
                break

            # Ask the configured approval channel for a structured decision,
            # then resume via the shared resume_decision interface.
            decision = _channel_decision(channel, interrupt_payload, correlation_id)
            final_state = resume_decision(graph, thread_id, decision)
            break  # single HITL per cycle

        _emit_cycle_done(symbol, correlation_id, final_state)
    except Exception as exc:
        _emit(
            {
                "event": "cycle_error",
                "symbol": symbol,
                "correlation_id": correlation_id,
                "error": str(exc),
            }
        )
    finally:
        reset_correlation_id(token)


def _channel_decision(
    channel: Any,
    interrupt_payload: dict[str, Any],
    correlation_id: str,
) -> Any:
    """Build a HITLRequest from the interrupt payload and ask *channel* to decide.

    Returns a structured ``HITLDecision`` (the channel maps timeout / undeliverable
    to ``EXPIRE`` — never auto-approve).
    """
    from firm.ports.types import HITLRequest

    req = HITLRequest.from_interrupt(interrupt_payload, correlation_id)
    decision = channel.request_decision(req)
    _emit(
        {
            "event": "hitl_decision",
            "correlation_id": correlation_id,
            "channel": type(channel).__name__,
            "decision": decision.value,
        }
    )
    return decision
