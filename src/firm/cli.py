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
run    -- LIVE production run: pull real market data + news for the last N
          days, run the 11-node graph against Postgres, and write a daily
          report. Requires Docker (make up), make seed, and a .env with
          ANTHROPIC_API_KEY + DATABASE_URL.
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

from firm.ports.llm import LLM

if TYPE_CHECKING:
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

    from firm.persistence.db_url import to_sqlalchemy_url

    root = _project_root()
    alembic_cfg = Config(str(root / "migrations" / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", to_sqlalchemy_url(database_url))
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
) -> tuple[Any, Any, Any, Any]:
    """Construct the pipeline agents. Returns (research, risk, execution, reporting).

    The Portfolio Manager agent has been dissolved — sizing is now handled by
    the deterministic ``size_position`` tool inside the graph's ``pm`` node.
    """
    from firm.adapters.fakes import FakeReportSink
    from firm.agents.execution import ExecutionAgent
    from firm.agents.reporting import ReportingAgent
    from firm.agents.research import ResearchAgent
    from firm.agents.risk import RiskAgent

    evidence_store = _build_evidence_store(root / "data" / "news" / "corpus.json")
    llm = _build_demo_llm(root / "data" / "cassettes" / "eval.jsonl")

    return (
        ResearchAgent(evidence=evidence_store, llm=llm, injection_guard=injection_guard),
        RiskAgent(risk=risk_policy_config),
        ExecutionAgent(ledger=ledger, guardrail=guardrail),
        ReportingAgent(report_sink=FakeReportSink(), ledger=ledger),
    )


def _build_pipeline(root: Path, initial_cash: Any) -> tuple[Any, Any, Any]:
    """Wire all agents + LangGraph graph for demo and dev commands.

    Returns (graph, portfolio, portfolio_id).
    """
    from langgraph.checkpoint.memory import MemorySaver

    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.adapters.market_data_frozen import FrozenMarketData
    from firm.adapters.report import ExcelReportSink, MultiReportSink, SlackReportSink
    from firm.domain.guardrails import TokenBudgetCircuitBreaker
    from firm.orchestration.graph import build_graph
    from firm.orchestration.nodes import NodePorts
    from firm.services.calendar import NYSECalendar

    risk_policy_config = _safe_load_risk_policy(root)
    portfolio, portfolio_id, guardrail, injection_guard, ledger = _build_domain_objects(
        risk_policy_config, initial_cash
    )
    market_data = FrozenMarketData(root / "data" / "bars")
    evidence_store = _build_evidence_store(root / "data" / "news" / "corpus.json")
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # SlackReportSink degrades gracefully: without SLACK_BOT_TOKEN it logs the
    # Block Kit payload instead of calling the API (no network, no crash).
    report_sink = MultiReportSink(
        sinks=[
            ExcelReportSink(output_dir=reports_dir),
            SlackReportSink(channel=os.getenv("SLACK_CHANNEL", "#trading-desk")),
        ]
    )

    raw_llm = _build_demo_llm(root / "data" / "cassettes" / "eval.jsonl")
    llm = TokenBudgetLLM(
        inner=raw_llm,
        breaker=TokenBudgetCircuitBreaker(),
        budget=risk_policy_config.token_budget_per_cycle,
        report_sink=report_sink,
    )

    ports = NodePorts(
        evidence=evidence_store,
        llm=llm,
        market_data=market_data,
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=risk_policy_config,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        calendar=NYSECalendar(),
    )
    graph = build_graph(checkpointer=MemorySaver(), ports=ports)
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


def _load_corpus_docs(corpus_path: Path) -> list[Any]:
    """Load corpus.json into NewsDoc objects; return empty list if missing."""
    if not corpus_path.exists():
        return []
    return _parse_news_docs(corpus_path)


def _build_demo_llm(cassette_path: Path) -> LLM:
    """Return the appropriate offline LLM for the demo command.

    Delegates to ``build_offline_llm`` — see that function for selection rules.
    A bare ``ANTHROPIC_API_KEY`` does NOT trigger live calls; only
    ``CASSETTE_MODE=record`` does (explicit opt-in).
    """
    from firm.adapters.llm_offline import build_offline_llm

    return build_offline_llm(cassette_path)


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
# run command (LIVE production)
# ---------------------------------------------------------------------------

# Default watchlist mirrors the demo watchlist.
_DEFAULT_WATCHLIST: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"]
_DEFAULT_LOOKBACK_DAYS: int = 7


def _cmd_run(args: argparse.Namespace) -> None:
    """Live production run: real market data + news + Postgres + live LLM.

    Prerequisites (not met in offline CI):
      - ``make up``   — Postgres + pgvector running in Docker.
      - ``make seed`` — migrations applied, tables exist.
      - ``.env``      — ANTHROPIC_API_KEY and DATABASE_URL set.
    """
    from decimal import Decimal

    root = _project_root()
    tickers = _parse_tickers(args.tickers)
    lookback_days: int = args.lookback_days

    _emit(
        {
            "event": "run_start",
            "tickers": tickers,
            "lookback_days": lookback_days,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
    )

    settings = _load_settings()
    _validate_live_settings(settings)

    _ingest_live_news(tickers, lookback_days, settings)

    graph, _portfolio, _portfolio_id = _build_live_pipeline(root, Decimal("100000"), settings)
    decision_ts = datetime.now(tz=UTC)
    _run_live_graph_loop(graph, tickers, decision_ts, settings)

    _emit({"event": "run_done", "ts": datetime.now(tz=UTC).isoformat()})


def _parse_tickers(tickers_arg: str | None) -> list[str]:
    """Parse comma-separated tickers from CLI arg; fall back to default watchlist."""
    if not tickers_arg:
        return _DEFAULT_WATCHLIST
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

    import psycopg

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


def _build_live_pipeline(
    root: Path,
    initial_cash: Any,
    settings: Settings,
) -> tuple[Any, Any, Any]:
    """Wire all live adapters + LangGraph graph for the production run command.

    Returns (graph, portfolio, portfolio_id).
    """
    import psycopg

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.adapters.llm_anthropic import AnthropicLLM
    from firm.adapters.llm_offline import GracefulLLM
    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.adapters.market_data_live import LiveMarketData
    from firm.adapters.report import ExcelReportSink, MultiReportSink, SlackReportSink
    from firm.domain.guardrails import TokenBudgetCircuitBreaker
    from firm.orchestration.checkpointer import _normalise_database_url, setup_checkpointer
    from firm.orchestration.graph import build_graph
    from firm.orchestration.nodes import NodePorts
    from firm.services.calendar import NYSECalendar

    risk_policy_config = _safe_load_risk_policy(root)
    portfolio, portfolio_id, guardrail, injection_guard, ledger = _build_live_domain_objects(
        risk_policy_config, initial_cash, settings
    )

    live_url = _normalise_database_url(settings.database_url)
    evidence_conn = psycopg.connect(live_url)  # kept open for the run lifetime
    evidence_store = PgvectorEvidenceStore(evidence_conn)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_sink = MultiReportSink(
        sinks=[
            ExcelReportSink(output_dir=reports_dir),
            SlackReportSink(channel=settings.slack_channel),
        ]
    )

    raw_llm = GracefulLLM(AnthropicLLM(api_key=settings.anthropic_api_key))
    llm = TokenBudgetLLM(
        inner=raw_llm,
        breaker=TokenBudgetCircuitBreaker(),
        budget=risk_policy_config.token_budget_per_cycle,
        report_sink=report_sink,
    )

    # PostgresSaver gives durable HITL: graph state survives process restarts.
    pg_conn = psycopg.connect(  # pyright: ignore[reportArgumentType]
        live_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=__import__("psycopg.rows", fromlist=["dict_row"]).dict_row,
    )
    checkpointer = setup_checkpointer(pg_conn)

    ports = NodePorts(
        evidence=evidence_store,
        llm=llm,
        market_data=LiveMarketData(),
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=risk_policy_config,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        calendar=NYSECalendar(),
    )
    graph = build_graph(checkpointer=checkpointer, ports=ports)
    return graph, portfolio, portfolio_id


def _build_live_domain_objects(
    risk_policy_config: RiskPolicyConfig,
    initial_cash: Any,
    settings: Settings,
) -> tuple[Any, Any, Any, Any, Any]:
    """Construct Portfolio, RiskPolicy, guards, and the real Postgres LedgerRepository.

    Returns (portfolio, portfolio_id, guardrail, injection_guard, ledger).
    """
    import uuid
    from decimal import Decimal

    from sqlalchemy import create_engine

    from firm.domain import Portfolio, RiskPolicy
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
    from firm.persistence.db_url import to_sqlalchemy_url
    from firm.persistence.ledger import LedgerRepository

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
    engine = create_engine(to_sqlalchemy_url(settings.database_url))
    ledger = LedgerRepository(engine)
    return portfolio, portfolio_id, guardrail, injection_guard, ledger


def _run_live_graph_loop(
    graph: Any,
    tickers: list[str],
    decision_ts: datetime,
    settings: Settings,
) -> None:
    """Invoke the graph once per ticker, handling HITL interrupts via console input."""
    for symbol in tickers:
        _invoke_live_symbol(graph, symbol, decision_ts, settings)


def _invoke_live_symbol(
    graph: Any,
    symbol: str,
    decision_ts: datetime,
    settings: Settings,
) -> None:
    """Stream the graph for one symbol; block on HITL interrupts for console approval."""
    import uuid

    from langgraph.types import Command

    from firm.observability.tracing import reset_correlation_id, set_correlation_id

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
        initial_state = {
            "symbol": symbol,
            "decision_ts": decision_ts.isoformat(),
            "correlation_id": correlation_id,
        }
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        # Stream events; stop on the first interrupt or when the graph ends.
        final_state: dict[str, Any] = {}
        while True:
            interrupted = False
            interrupt_payload: dict[str, Any] = {}
            for event in graph.stream(initial_state, config=config, stream_mode="values"):
                final_state = event
            # Check whether the graph halted on an interrupt by querying state.
            run_state = graph.get_state(config)
            if run_state.next and run_state.tasks:
                for task in run_state.tasks:
                    if getattr(task, "interrupts", None):
                        interrupted = True
                        interrupt_payload = task.interrupts[0].value if task.interrupts else {}
                        break

            if not interrupted:
                break

            # Console HITL: block until the operator responds.
            resume_value, hitl_status = _console_hitl_prompt(symbol, interrupt_payload)
            # Resume the graph with the operator decision.
            # Command is typed generically; cast to Any so we can pass it
            # to graph.stream() which accepts Any input after an interrupt.
            resume_cmd: Any = Command(
                resume=resume_value,
                update={"hitl_status": hitl_status},
            )
            # Re-invoke with the command directly (LangGraph resumes from checkpoint).
            for event in graph.stream(resume_cmd, config=config, stream_mode="values"):
                final_state = event
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


def _console_hitl_prompt(
    symbol: str,
    payload: dict[str, Any],
) -> tuple[str, str]:
    """Block on console input for a HITL decision; return (resume_value, hitl_status).

    Accepted responses: approve / a, reject / r, edit <qty> / e <qty>.
    Any unrecognised input defaults to rejection for safety.
    """

    proposal = payload.get("trade_proposal", {})
    print(
        f"\n[HITL] Trade requires human approval for {symbol}\n"
        f"  Proposal: {proposal}\n"
        "  Options: [a]pprove  [r]eject  [e]dit <qty>",
        flush=True,
    )
    try:
        raw = input("Decision > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raw = "r"

    return _parse_hitl_response(raw)


def _parse_hitl_response(raw: str) -> tuple[str, str]:
    """Parse a raw HITL response string into (resume_value, hitl_status).

    Defaults to rejection on any unrecognised input so that ambiguity is safe.
    """
    from firm.domain.enums import HITLStatus

    if raw in {"a", "approve"}:
        return ("approved", HITLStatus.APPROVED)
    if raw in {"r", "reject"}:
        return ("rejected", HITLStatus.REJECTED)
    if raw.startswith(("e ", "edit ")):
        qty_str = raw.split(maxsplit=1)[1] if " " in raw else ""
        if qty_str.replace(".", "").isdigit():
            return (f"edit:{qty_str}", HITLStatus.APPROVED)
    return ("rejected", HITLStatus.REJECTED)


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
# web command (browser dashboard)
# ---------------------------------------------------------------------------


def _cmd_web(args: argparse.Namespace) -> None:
    """Start the FastAPI dashboard server via uvicorn."""
    from firm.web.server import run_server

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


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
    _add_run_subcommand(sub)
    _add_trace_subcommand(sub)
    _add_web_subcommand(sub)
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


def _add_run_subcommand(sub: Any) -> None:
    """Register the 'run' subcommand (live production mode)."""
    run_p = sub.add_parser(
        "run",
        help=(
            "LIVE production run: fetch real market data + news, run the 11-node graph "
            "against Postgres, and write a daily report. "
            "Requires: make up, make seed, ANTHROPIC_API_KEY in .env."
        ),
    )
    run_p.add_argument(
        "--tickers",
        default=None,
        metavar="TICKER,TICKER,...",
        help=(
            f"Comma-separated list of tickers to analyse (default: {','.join(_DEFAULT_WATCHLIST)})."
        ),
    )
    run_p.add_argument(
        "--lookback-days",
        type=int,
        default=_DEFAULT_LOOKBACK_DAYS,
        dest="lookback_days",
        metavar="N",
        help=f"Number of calendar days of market data + news to pull (default: {_DEFAULT_LOOKBACK_DAYS}).",
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


def _add_web_subcommand(sub: Any) -> None:
    """Register the 'web' subcommand (browser dashboard)."""
    web_p = sub.add_parser(
        "web",
        help=(
            "Start the FastAPI dashboard on http://localhost:8000. "
            "Requires DATABASE_URL; ANTHROPIC_API_KEY enables HITL endpoints."
        ),
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="TCP port to listen on (default: 8000).",
    )
    web_p.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Host to bind (default: 0.0.0.0).",
    )
    web_p.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn hot-reload (dev mode).",
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
        "run": _cmd_run,
        "trace": _cmd_trace,
        "web": _cmd_web,
    }
    dispatch[args.command](args)
    from firm.observability import flush_telemetry

    flush_telemetry()


if __name__ == "__main__":
    main()
