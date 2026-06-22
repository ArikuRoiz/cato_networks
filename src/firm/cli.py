"""Command-line entry point for the AI Investment Firm.

Commands
--------
seed   -- Run Alembic migrations, load frozen bar CSVs into the frozen-data
          directory, and embed the news corpus into pgvector.
demo   -- Run one full replay day (Oct 23 2024, NVDA earnings day) against
          frozen data with recorded LLM responses. Prints structured trace
          to stdout in NDJSON format.
dev    -- Start the scheduler + event listener in a foreground loop against
          frozen data. Use for local development with hot-reload.
trace  -- Print the audit log for a single trade from the database,
          identified by --trade-id (UUID). Outputs NDJSON to stdout.

All commands read DATABASE_URL (and other config) from the environment.
No secrets are hard-coded here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firm.adapters.fakes import FakeLLM
    from firm.adapters.llm_cassette import CassetteLLM
    from firm.config.settings import RiskPolicyConfig, Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root directory (three levels above this file)."""
    return Path(__file__).parent.parent.parent


def _load_dotenv() -> None:
    """Load .env from the project root into os.environ if not already set."""
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _emit(record: dict[str, Any]) -> None:
    """Print a single JSON record to stdout followed by a newline."""
    print(json.dumps(record), flush=True)


def _load_settings() -> Settings:
    """Load application settings from the environment."""
    from firm.config.settings import load_settings

    return load_settings()


# ---------------------------------------------------------------------------
# seed command
# ---------------------------------------------------------------------------


def _cmd_seed(args: argparse.Namespace) -> None:
    """Run migrations, load frozen bar CSVs, embed news corpus."""
    settings = _load_settings()
    root = _project_root()

    _emit({"step": "migrations", "status": "starting"})
    _run_migrations(settings.database_url)
    _emit({"step": "migrations", "status": "ok"})

    _check_bars(root / "data" / "bars")

    corpus_path = root / "data" / "news" / "corpus.json"
    _emit({"step": "corpus", "status": "starting", "path": str(corpus_path)})
    if not corpus_path.exists():
        _emit(
            {
                "step": "corpus",
                "status": "warning",
                "message": "corpus.json not found; skipping embedding",
            }
        )
    else:
        count = _embed_corpus(corpus_path, settings.database_url)
        _emit({"step": "corpus", "status": "ok", "articles_embedded": count})

    _emit({"step": "seed", "status": "done"})


def _check_bars(bars_dir: Path) -> None:
    """Emit a status event for the frozen bar CSV files."""
    _emit({"step": "bars", "status": "checking", "dir": str(bars_dir)})
    bar_files = list(bars_dir.glob("*.csv"))
    if not bar_files:
        _emit({"step": "bars", "status": "warning", "message": "no CSV files found in data/bars/"})
    else:
        _emit({"step": "bars", "status": "ok", "files": [f.name for f in bar_files]})


def _run_migrations(database_url: str) -> None:
    """Run Alembic migrations programmatically."""
    from alembic import command
    from alembic.config import Config

    root = _project_root()
    alembic_cfg = Config(str(root / "migrations" / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")


def _embed_corpus(corpus_path: Path, database_url: str) -> int:
    """Parse corpus.json and upsert articles into pgvector via PgvectorEvidenceStore.

    Requires a live Postgres connection identified by *database_url*.
    Returns the number of articles processed.
    """
    import psycopg

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.orchestration.checkpointer import _normalise_database_url

    docs = _parse_news_docs(corpus_path)
    url = _normalise_database_url(database_url)
    with psycopg.connect(url) as conn:
        store = PgvectorEvidenceStore(conn)
        store.migrate()
        for doc in docs:
            store.embed_and_store(doc)
        conn.commit()
    return len(docs)


def _parse_news_docs(corpus_path: Path) -> list[Any]:
    """Read corpus.json and return a list of NewsDoc objects."""
    import json as _json

    from firm.ports.types import NewsDoc

    raw: list[dict[str, Any]] = _json.loads(corpus_path.read_text(encoding="utf-8"))
    return [
        NewsDoc(
            symbol=item["symbol"],
            text=item["text"],
            source_url=item["source_url"],
            published_at=datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")),
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# demo command
# ---------------------------------------------------------------------------


def _cmd_demo(args: argparse.Namespace) -> None:
    """Replay Oct 23 2024 (NVDA earnings day) end-to-end, print trace."""
    from decimal import Decimal

    root = _project_root()
    demo_date = datetime(2024, 10, 23, tzinfo=UTC)
    watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"]

    _emit(
        {
            "event": "demo_start",
            "date": demo_date.date().isoformat(),
            "watchlist": watchlist,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
    )

    graph, _portfolio, _portfolio_id = _build_pipeline(root, Decimal("100000"))
    _run_graph_loop(graph, watchlist, demo_date)
    _emit({"event": "demo_done", "ts": datetime.now(tz=UTC).isoformat()})


def _build_domain_objects(
    risk_policy_config: RiskPolicyConfig,
    initial_cash: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Construct Portfolio, RiskPolicy, LedgerGuardrail, InjectionGuard, FakeLedger.

    Returns (portfolio, portfolio_id, domain_policy, guardrail, ledger).
    """
    import uuid
    from decimal import Decimal

    from eval.replay import _FakeLedger
    from firm.domain import Portfolio, RiskPolicy
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail

    portfolio_id = uuid.uuid4()
    portfolio = Portfolio(cash=Decimal(str(initial_cash)))
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(risk_policy_config.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(risk_policy_config.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(risk_policy_config.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(risk_policy_config.hitl_threshold_pct)),
    )
    guardrail = LedgerGuardrail(domain_policy)
    injection_guard = InjectionGuard()
    ledger = _FakeLedger(portfolio, portfolio_id)
    return portfolio, portfolio_id, guardrail, injection_guard, ledger


def _build_evidence_store(corpus_path: Path) -> Any:
    """Load corpus docs into a FakeEvidenceStore and return it."""
    from firm.adapters.fakes import FakeEvidenceStore

    store = FakeEvidenceStore()
    for doc in _load_corpus_docs(corpus_path):
        store.embed_and_store(doc)
    return store


def _build_agents(
    root: Path,
    risk_policy_config: RiskPolicyConfig,
    guardrail: Any,
    injection_guard: Any,
    ledger: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Construct the five pipeline agents. Returns (research, pm, risk, execution, reporting)."""
    from firm.adapters.fakes import FakeReportSink
    from firm.adapters.market_data_frozen import FrozenMarketData
    from firm.agents.execution import ExecutionAgent
    from firm.agents.portfolio_manager import PortfolioManagerAgent
    from firm.agents.reporting import ReportingAgent
    from firm.agents.research import ResearchAgent
    from firm.agents.risk import RiskAgent

    market_data = FrozenMarketData(root / "data" / "bars")
    evidence_store = _build_evidence_store(root / "data" / "news" / "corpus.json")
    llm = _build_demo_llm(root / "data" / "cassettes" / "eval.jsonl")

    return (
        ResearchAgent(evidence=evidence_store, llm=llm, injection_guard=injection_guard),
        PortfolioManagerAgent(market_data=market_data, risk=risk_policy_config, llm=None),
        RiskAgent(risk=risk_policy_config),
        ExecutionAgent(ledger=ledger, guardrail=guardrail),
        ReportingAgent(report_sink=FakeReportSink(), ledger=ledger),
    )


def _build_pipeline(root: Path, initial_cash: Any) -> tuple[Any, Any, Any]:
    """Wire all agents + LangGraph graph for demo and dev commands.

    Returns (graph, portfolio, portfolio_id).
    """
    from langgraph.checkpoint.memory import MemorySaver

    from firm.adapters.market_data_frozen import FrozenMarketData
    from firm.adapters.report import FileReportSink
    from firm.orchestration.graph import build_graph
    from firm.orchestration.nodes import NodePorts

    risk_policy_config = _safe_load_risk_policy(root)
    portfolio, portfolio_id, guardrail, injection_guard, ledger = _build_domain_objects(
        risk_policy_config, initial_cash
    )
    market_data = FrozenMarketData(root / "data" / "bars")
    evidence_store = _build_evidence_store(root / "data" / "news" / "corpus.json")
    llm = _build_demo_llm(root / "data" / "cassettes" / "eval.jsonl")
    reports_dir = root / "reports"

    ports = NodePorts(
        evidence=evidence_store,
        llm=llm,
        market_data=market_data,
        ledger=ledger,
        report_sink=FileReportSink(output_dir=reports_dir),
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=risk_policy_config,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
    )
    graph = build_graph(
        checkpointer=MemorySaver(),
        risk_policy=risk_policy_config,
        ports=ports,
    )
    return graph, portfolio, portfolio_id


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

    correlation_id = str(uuid.uuid4())
    _emit(
        {
            "event": "cycle_start",
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
    )
    initial_state = {
        "symbol": symbol,
        "decision_ts": decision_ts.isoformat(),
        "correlation_id": correlation_id,
    }
    config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        from langfuse.decorators import langfuse_context, observe

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
                langfuse_context.update_current_observation(level="ERROR", status_message=str(exc))
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
        import logging

        logging.getLogger(__name__).warning("Langfuse unavailable, running untraced: %s", exc)
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


def _run_graph_loop(
    graph: Any,
    watchlist: list[str],
    decision_ts: datetime,
) -> None:
    """Invoke *graph* once per symbol in *watchlist*, emitting NDJSON events."""
    for symbol in watchlist:
        _invoke_one_symbol(graph, symbol, decision_ts)


def _load_corpus_docs(corpus_path: Path) -> list[Any]:
    """Load corpus.json into NewsDoc objects; return empty list if missing."""
    if not corpus_path.exists():
        return []
    return _parse_news_docs(corpus_path)


def _build_demo_llm(cassette_path: Path) -> CassetteLLM | FakeLLM:
    """Return CassetteLLM in record mode when an API key is set, else FakeLLM.

    - CASSETTE_MODE=replay: force replay (fails on miss; useful for CI with a full cassette)
    - CASSETTE_MODE=record (or ANTHROPIC_API_KEY set): record live calls into the cassette
    - default (no API key, no CASSETTE_MODE): FakeLLM so demo runs offline without errors
    """
    from firm.adapters.fakes import FakeLLM
    from firm.adapters.llm_cassette import CassetteLLM
    from firm.ports.types import LLMResponse

    explicit_mode = os.environ.get("CASSETTE_MODE", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if explicit_mode == "replay" and cassette_path.exists() and cassette_path.stat().st_size > 0:
        return CassetteLLM(cassette_path=cassette_path, mode="replay")

    if api_key or explicit_mode == "record":
        from firm.adapters.llm_anthropic import AnthropicLLM

        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        inner = AnthropicLLM(api_key=api_key)
        return CassetteLLM(cassette_path=cassette_path, mode="record", inner=inner)

    canned = LLMResponse(content="[]", input_tokens=10, output_tokens=2, model="claude-haiku-4-5")
    return FakeLLM(responses=[canned] * 500)


def _safe_load_risk_policy(root: Path) -> RiskPolicyConfig:
    """Load risk policy from YAML, returning defaults on failure."""
    from firm.config.settings import RiskPolicyConfig, load_risk_policy

    policy_path = root / "config" / "risk_policy.yaml"
    if policy_path.exists():
        try:
            return load_risk_policy(policy_path)
        except Exception:
            pass
    return RiskPolicyConfig(
        max_trade_notional_pct=0.10,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=0.05,
        buy_threshold=0.05,
        sell_threshold=-0.05,
        momentum_weight=0.6,
        sentiment_weight=0.4,
        momentum_lookback_days=5,
        max_events_per_symbol_per_hour=3,
        event_relevance_threshold=0.7,
        slippage_bps=5,
        commission_per_share=0.005,
        token_budget_per_cycle=50000,
    )


def _summarise(value: Any) -> str:
    """Return a short human-readable summary of an agent result."""
    if value is None:
        return "none"
    if isinstance(value, dict):
        keys = list(value.keys())[:4]
        return f"dict({', '.join(keys)})"
    return type(value).__name__


# ---------------------------------------------------------------------------
# dev command
# ---------------------------------------------------------------------------


def _cmd_dev(args: argparse.Namespace) -> None:
    """Start the scheduler + event listener in a foreground loop.

    Fires a decision cycle for each watchlist symbol on a 30-second polling
    interval using frozen data and the fake/cassette LLM.  Press Ctrl+C to
    stop.
    """
    import time
    from decimal import Decimal

    root = _project_root()
    watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"]
    poll_interval_s = 30

    _emit({"event": "dev_start", "watchlist": watchlist, "poll_interval_s": poll_interval_s})

    graph, _portfolio, _portfolio_id = _build_pipeline(root, Decimal("100000"))
    cycle_count = 0
    try:
        while True:
            now = datetime.now(tz=UTC)
            _emit({"event": "tick", "ts": now.isoformat(), "cycle": cycle_count})
            _run_graph_loop(graph, watchlist, now)
            cycle_count += 1
            _emit({"event": "sleeping", "seconds": poll_interval_s})
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        _emit({"event": "dev_stop", "cycles_run": cycle_count})


# ---------------------------------------------------------------------------
# trace command
# ---------------------------------------------------------------------------


def _cmd_trace(args: argparse.Namespace) -> None:
    """Reconstruct the audit log for a single trade by trade_id.

    Queries the audit_log table for all entries whose correlation_id matches
    the decision cycle that contains the given trade_id. Emits NDJSON to
    stdout (one event per line).
    """
    from firm.observability.tracing import (
        get_correlation_id,
        reset_correlation_id,
        set_correlation_id,
    )

    trade_id: str = args.trade_id
    token = set_correlation_id(trade_id)
    try:
        active_cid = get_correlation_id()
        _emit({"correlation_id": active_cid, "trade_id": trade_id})
        settings = _load_settings()
        entries = _query_audit_log(trade_id, settings.database_url)
        _emit_audit_entries(trade_id, active_cid, entries)
    finally:
        reset_correlation_id(token)


def _emit_audit_entries(
    trade_id: str,
    correlation_id: str,
    entries: list[dict[str, Any]],
) -> None:
    """Emit NDJSON audit entries, or a not-found record when *entries* is empty."""
    if not entries:
        _emit(
            {
                "status": "not_found",
                "message": (
                    f"No audit entries found for trade {trade_id}. "
                    "Ensure the trade exists and DATABASE_URL is correct."
                ),
            }
        )
        return
    for entry in entries:
        _emit(entry)
    _emit(
        {
            "status": "ok",
            "trade_id": trade_id,
            "correlation_id": correlation_id,
            "entry_count": len(entries),
        }
    )


def _fetch_audit_rows(url: str, trade_id: str) -> list[Any]:
    """Execute the audit_log SQL query and return raw dict rows."""
    import psycopg
    import psycopg.rows

    with psycopg.connect(url, row_factory=psycopg.rows.dict_row) as conn:  # pyright: ignore[reportArgumentType]
        return conn.execute(
            """
            SELECT
                al.id::text          AS id,
                al.correlation_id::text AS correlation_id,
                al.actor,
                al.action,
                al.payload,
                al.ts::text          AS ts
            FROM audit_log al
            JOIN trades t ON t.cycle_id = al.correlation_id
            WHERE t.id = %s::uuid
            ORDER BY al.ts
            """,
            (trade_id,),
        ).fetchall()


def _query_audit_log(trade_id: str, database_url: str) -> list[dict[str, Any]]:
    """Return audit entries for *trade_id*; empty list on any DB error."""
    try:
        from firm.orchestration.checkpointer import _normalise_database_url

        rows = _fetch_audit_rows(_normalise_database_url(database_url), trade_id)
        return [dict(row) for row in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="firm",
        description="The AI Investment Firm — multi-agent paper-trading desk",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_seed_subcommand(sub)
    _add_demo_subcommand(sub)
    _add_dev_subcommand(sub)
    _add_trace_subcommand(sub)
    return parser


def _add_seed_subcommand(sub: Any) -> None:
    """Register the 'seed' subcommand."""
    sub.add_parser(
        "seed",
        help=(
            "Run Alembic migrations, verify frozen bar CSVs, and embed the news corpus "
            "into the evidence store."
        ),
    )


def _add_demo_subcommand(sub: Any) -> None:
    """Register the 'demo' subcommand."""
    sub.add_parser(
        "demo",
        help=(
            "Replay Oct 23 2024 (NVDA earnings day) end-to-end against frozen data "
            "and recorded LLM responses. Prints a structured trace to stdout."
        ),
    )


def _add_dev_subcommand(sub: Any) -> None:
    """Register the 'dev' subcommand."""
    sub.add_parser(
        "dev",
        help=(
            "Start the scheduler + event listener in a foreground loop against frozen data. "
            "Press Ctrl+C to stop."
        ),
    )


def _add_trace_subcommand(sub: Any) -> None:
    """Register the 'trace' subcommand."""
    trace_p = sub.add_parser(
        "trace",
        help="Print the full audit log for one trade, identified by --trade-id.",
    )
    trace_p.add_argument(
        "--trade-id",
        required=True,
        metavar="UUID",
        help="UUID of the trade whose audit log should be printed.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point invoked by pyproject.toml [project.scripts] and python -m firm.cli."""
    _load_dotenv()
    from firm.observability import setup_telemetry

    setup_telemetry()
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "seed": _cmd_seed,
        "demo": _cmd_demo,
        "dev": _cmd_dev,
        "trace": _cmd_trace,
    }
    dispatch[args.command](args)
    from firm.observability import flush_telemetry

    flush_telemetry()


if __name__ == "__main__":
    main()
