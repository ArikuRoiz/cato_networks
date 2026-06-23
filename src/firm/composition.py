"""Composition root — the single place that assembles the decision pipeline.

Every runtime (the ``firm`` CLI, the eval replay harness, and the FastAPI web
backend) builds the *same* eleven-node graph wired to the *same* ``NodePorts``.
Historically each of those three call sites grew its own copy of the wiring;
this module collapses them into two public builders:

  - :func:`build_offline_pipeline` — frozen market data, a corpus-loaded
    :class:`FakeEvidenceStore`, the offline LLM, the in-memory
    :class:`InMemoryLedger`, and a ``MemorySaver`` checkpointer.  Used by
    ``firm demo`` / ``firm dev`` and by the eval replay harness.
  - :func:`build_live_pipeline` — :class:`LiveMarketData`, a pgvector evidence
    store, the live ``AnthropicLLM`` (wrapped with the graceful + token-budget
    decorators), a Postgres-backed :class:`LedgerRepository`, and a
    ``PostgresSaver`` checkpointer.  Used by ``firm run`` / ``firm bot`` and the
    web dashboard.

Both return a typed :class:`Pipeline` so callers consume named fields rather
than positional tuples.  The private sub-builders (domain objects, ports,
evidence store, in-memory ledger) live here too, so no caller has to know how
the pieces fit together.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from firm.constants import DEFAULT_INITIAL_CASH
from firm.domain import Holding, Lot, Portfolio, RiskPolicy, Trade, TradeStatus
from firm.persistence.ledger import CycleAuditRecord

if TYPE_CHECKING:
    from firm.config.settings import RiskPolicyConfig, Settings
    from firm.orchestration.nodes import NodePorts
    from firm.ports.report import ReportSink
    from firm.ports.types import NewsDoc


def _project_root() -> Path:
    """Return the repository root (three levels above this file)."""
    return Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline:
    """A fully wired, ready-to-invoke decision pipeline.

    ``graph``, ``portfolio`` and ``portfolio_id`` are populated for both the
    offline and live builders.  ``checkpointer``, ``engine`` and ``ledger`` are
    only populated by :func:`build_live_pipeline` — the web backend needs to
    reach the checkpointer and engine independently of the graph; offline
    callers leave them ``None``.
    """

    graph: Any  # CompiledStateGraph (kept loose to avoid a heavy import here)
    portfolio: Portfolio
    portfolio_id: uuid.UUID
    checkpointer: Any = None  # PostgresSaver (live only)
    engine: Any = None  # SQLAlchemy Engine (live only)
    ledger: Any = None  # LedgerRepository (live only)


# ---------------------------------------------------------------------------
# Risk policy + domain objects
# ---------------------------------------------------------------------------


def _load_risk_policy(root: Path) -> RiskPolicyConfig:
    """Load the risk policy YAML, falling back to the built-in defaults."""
    from firm.config.settings import load_risk_policy_or_default

    return load_risk_policy_or_default(root / "config" / "risk_policy.yaml")


def _domain_policy(config: RiskPolicyConfig) -> RiskPolicy:
    """Project the typed config into the domain ``RiskPolicy`` value object."""
    return RiskPolicy(
        max_trade_notional_pct=Decimal(str(config.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(config.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(config.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(config.hitl_threshold_pct)),
    )


def _build_guards(config: RiskPolicyConfig) -> tuple[Any, Any]:
    """Return ``(LedgerGuardrail, InjectionGuard)`` for the given policy."""
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail

    return LedgerGuardrail(_domain_policy(config)), InjectionGuard()


# ---------------------------------------------------------------------------
# Evidence store (offline)
# ---------------------------------------------------------------------------


def _parse_corpus(corpus_path: Path) -> list[NewsDoc]:
    """Parse ``corpus.json`` into ``NewsDoc`` objects; empty when absent."""
    from firm.ports.types import NewsDoc

    if not corpus_path.exists():
        return []
    raw: list[dict[str, Any]] = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [
        NewsDoc(
            symbol=item["symbol"],
            text=item["text"],
            source_url=item["source_url"],
            published_at=datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")),
        )
        for item in raw
    ]


def _build_fake_evidence_store(corpus_path: Path) -> Any:
    """Load corpus docs into a ``FakeEvidenceStore`` and return it."""
    from firm.adapters.fakes import FakeEvidenceStore

    store = FakeEvidenceStore()
    for doc in _parse_corpus(corpus_path):
        store.embed_and_store(doc)
    return store


# ---------------------------------------------------------------------------
# Report sink + LLM wrapping
# ---------------------------------------------------------------------------


def _build_multi_report_sink(root: Path, slack_channel: str) -> ReportSink:
    """Build the standard Excel + Slack multi-sink used by demo and live runs."""
    from firm.adapters.report import ExcelReportSink, MultiReportSink, SlackReportSink

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return MultiReportSink(
        sinks=[
            ExcelReportSink(output_dir=reports_dir),
            SlackReportSink(channel=slack_channel),
        ]
    )


def _wrap_token_budget(raw_llm: Any, config: RiskPolicyConfig, report_sink: ReportSink) -> Any:
    """Wrap *raw_llm* in the token-budget circuit breaker decorator."""
    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.domain.guardrails import TokenBudgetCircuitBreaker

    return TokenBudgetLLM(
        inner=raw_llm,
        breaker=TokenBudgetCircuitBreaker(),
        budget=config.token_budget_per_cycle,
        report_sink=report_sink,
    )


# ---------------------------------------------------------------------------
# Offline pipeline (demo / dev / eval)
# ---------------------------------------------------------------------------


def build_offline_pipeline(
    root: Path | None = None,
    *,
    initial_cash: Decimal = DEFAULT_INITIAL_CASH,
    report_sink: ReportSink | None = None,
    cassette_path: Path | None = None,
) -> Pipeline:
    """Assemble the fully offline pipeline against frozen data.

    Wires :class:`FrozenMarketData`, a corpus-loaded :class:`FakeEvidenceStore`,
    the offline LLM (``build_offline_llm``), the in-memory :class:`InMemoryLedger`,
    and a ``MemorySaver`` checkpointer.

    *report_sink* lets the eval harness substitute its auto-approving
    ``FakeReportSink``; when omitted the standard Excel + Slack multi-sink is
    used (the demo / dev behaviour).  ``SlackReportSink`` degrades gracefully
    without ``SLACK_BOT_TOKEN`` — it logs the Block Kit payload instead of
    calling the API.

    *cassette_path* selects the LLM cassette; when omitted it defaults to
    ``data/cassettes/eval.jsonl`` (the demo path).  A path that does not exist
    deterministically selects the offline ``FakeLLM`` — the eval relies on this.
    """
    import os

    from langgraph.checkpoint.memory import MemorySaver

    from firm.adapters.llm_offline import build_offline_llm
    from firm.adapters.market_data_frozen import FrozenMarketData

    root = root or _project_root()
    config = _load_risk_policy(root)
    guardrail, injection_guard = _build_guards(config)

    portfolio_id = uuid.uuid4()
    portfolio = Portfolio(cash=Decimal(str(initial_cash)))
    ledger = InMemoryLedger(portfolio, portfolio_id)

    market_data = FrozenMarketData(root / "data" / "bars")
    evidence_store = _build_fake_evidence_store(root / "data" / "news" / "corpus.json")

    if report_sink is None:
        report_sink = _build_multi_report_sink(root, os.getenv("SLACK_CHANNEL", "#trading-desk"))

    raw_llm = build_offline_llm(cassette_path or root / "data" / "cassettes" / "eval.jsonl")
    llm = _wrap_token_budget(raw_llm, config, report_sink)

    graph = _compile_graph(
        checkpointer=MemorySaver(),
        evidence=evidence_store,
        llm=llm,
        market_data=market_data,
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        config=config,
        portfolio=portfolio,
        portfolio_id=portfolio_id,
    )
    return Pipeline(graph=graph, portfolio=portfolio, portfolio_id=portfolio_id)


# ---------------------------------------------------------------------------
# Live pipeline (run / bot / web)
# ---------------------------------------------------------------------------


def build_live_pipeline(
    settings: Settings,
    *,
    root: Path | None = None,
    initial_cash: Decimal = DEFAULT_INITIAL_CASH,
) -> Pipeline:
    """Assemble the live production pipeline against Postgres + real services.

    Wires :class:`LiveMarketData`, a pgvector evidence store, the live
    ``AnthropicLLM`` (wrapped with the graceful + token-budget decorators), the
    Postgres-backed :class:`LedgerRepository`, the Excel + Slack report sink, and
    a durable ``PostgresSaver`` checkpointer.

    The ledger uses the stable ``FIRM_PORTFOLIO_ID`` so portfolio state
    accumulates across restarts; ``ensure_portfolio`` creates the row on first
    use and ``get_portfolio`` loads the persisted state so NAV / sizing reflect
    real cash + holdings.  The returned :class:`Pipeline` exposes ``checkpointer``,
    ``engine`` and ``ledger`` so the web backend can inspect interrupts and run
    the pending-run registry.
    """
    import psycopg

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.adapters.llm_anthropic import AnthropicLLM
    from firm.adapters.llm_offline import GracefulLLM
    from firm.adapters.market_data_live import LiveMarketData
    from firm.orchestration.checkpointer import _normalise_database_url, setup_checkpointer

    root = root or _project_root()
    config = _load_risk_policy(root)
    guardrail, injection_guard = _build_guards(config)

    engine, ledger, portfolio, portfolio_id = _build_live_ledger(settings, initial_cash)

    live_url = _normalise_database_url(settings.database_url)
    evidence_conn = psycopg.connect(live_url)  # kept open for the run lifetime
    evidence_store = PgvectorEvidenceStore(evidence_conn)

    report_sink = _build_multi_report_sink(root, settings.slack_channel)

    raw_llm = GracefulLLM(AnthropicLLM(api_key=settings.anthropic_api_key))
    llm = _wrap_token_budget(raw_llm, config, report_sink)

    # PostgresSaver gives durable HITL: graph state survives process restarts.
    pg_conn = psycopg.connect(  # pyright: ignore[reportArgumentType]
        live_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=__import__("psycopg.rows", fromlist=["dict_row"]).dict_row,
    )
    checkpointer = setup_checkpointer(pg_conn)

    graph = _compile_graph(
        checkpointer=checkpointer,
        evidence=evidence_store,
        llm=llm,
        market_data=LiveMarketData(),
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        config=config,
        portfolio=portfolio,
        portfolio_id=portfolio_id,
    )
    return Pipeline(
        graph=graph,
        portfolio=portfolio,
        portfolio_id=portfolio_id,
        checkpointer=checkpointer,
        engine=engine,
        ledger=ledger,
    )


def _build_live_ledger(
    settings: Settings,
    initial_cash: Decimal,
) -> tuple[Any, Any, Portfolio, uuid.UUID]:
    """Build the Postgres ledger and load the persisted firm portfolio.

    Returns ``(engine, ledger, portfolio, FIRM_PORTFOLIO_ID)``.  Idempotently
    ensures the portfolio row exists before loading it.
    """
    from sqlalchemy import create_engine

    from firm.persistence.db_url import to_sqlalchemy_url
    from firm.persistence.ledger import FIRM_PORTFOLIO_ID, LedgerRepository

    engine = create_engine(to_sqlalchemy_url(settings.database_url))
    ledger = LedgerRepository(engine)
    ledger.ensure_portfolio(FIRM_PORTFOLIO_ID, Decimal(str(initial_cash)))
    portfolio = ledger.get_portfolio(FIRM_PORTFOLIO_ID)
    return engine, ledger, portfolio, FIRM_PORTFOLIO_ID


# ---------------------------------------------------------------------------
# Shared graph compilation
# ---------------------------------------------------------------------------


def _compile_graph(
    *,
    checkpointer: Any,
    evidence: Any,
    llm: Any,
    market_data: Any,
    ledger: Any,
    report_sink: Any,
    guardrail: Any,
    injection_guard: Any,
    config: RiskPolicyConfig,
    portfolio: Portfolio,
    portfolio_id: uuid.UUID,
) -> Any:
    """Wire ``NodePorts`` and compile the production graph."""
    from firm.orchestration.graph import build_graph
    from firm.orchestration.nodes import NodePorts
    from firm.services.calendar import NYSECalendar

    ports: NodePorts = NodePorts(
        evidence=evidence,
        llm=llm,
        market_data=market_data,
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=config,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        calendar=NYSECalendar(),
    )
    return build_graph(checkpointer=checkpointer, ports=ports)


# ---------------------------------------------------------------------------
# In-memory ledger (offline demo + eval — no Postgres required)
# ---------------------------------------------------------------------------


class _ApprovalRecord:
    """In-memory record of a HITL decision captured by :class:`InMemoryLedger`."""

    __slots__ = (
        "correlation_id",
        "decided_at",
        "decided_by",
        "edited_qty",
        "original_notional",
        "original_qty",
        "status",
        "trade_id",
    )

    def __init__(
        self,
        *,
        correlation_id: uuid.UUID,
        trade_id: uuid.UUID,
        status: str,
        original_notional: Decimal,
        original_qty: Decimal,
        edited_qty: Decimal | None,
        decided_at: datetime,
        decided_by: str,
    ) -> None:
        self.correlation_id = correlation_id
        self.trade_id = trade_id
        self.status = status
        self.original_notional = original_notional
        self.original_qty = original_qty
        self.edited_qty = edited_qty
        self.decided_at = decided_at
        self.decided_by = decided_by


class InMemoryLedger:
    """Minimal in-memory ledger for offline demo and eval runs.

    Tracks the portfolio in memory.  No Postgres, no ACID — only correct for a
    single-threaded run.  ``buy()`` / ``sell()`` return a ``Trade`` with FILLED
    status to satisfy the structural contract of ``LedgerRepository``;
    ``record_approval()`` and ``record_cycle()`` append to in-memory lists so the
    HITL recording path can be exercised without a database.
    """

    def __init__(self, portfolio: Portfolio, portfolio_id: uuid.UUID) -> None:
        self._portfolio = portfolio
        self._portfolio_id = portfolio_id
        self._trades: list[Trade] = []
        self.approvals: list[_ApprovalRecord] = []
        self.cycles: list[CycleAuditRecord] = []

    def ensure_portfolio(self, portfolio_id: uuid.UUID, starting_cash: Decimal) -> None:
        """No-op: the portfolio is constructed before the ledger wraps it."""

    def get_portfolio(self, portfolio_id: uuid.UUID) -> Portfolio:
        return self._portfolio

    def get_trades_for_cycle(self, cycle_id: uuid.UUID) -> list[Trade]:
        return [t for t in self._trades if t.cycle_id == cycle_id]

    def buy(
        self,
        trade: Trade,
        portfolio_id: uuid.UUID,
        opened_at: datetime | None = None,
    ) -> Trade:
        """Debit cash and add a holding; return the filled Trade."""
        filled = trade.model_copy(update={"status": TradeStatus.FILLED})
        notional = trade.qty * (trade.fill_price or trade.requested_price)
        commission = trade.commission or Decimal("0")
        self._portfolio.cash -= notional + commission
        _apply_buy(self._portfolio, trade, opened_at)
        self._trades.append(filled)
        return filled

    def sell(self, trade: Trade, portfolio_id: uuid.UUID) -> Trade:
        """Credit cash and reduce holding; return the filled Trade."""
        filled = trade.model_copy(update={"status": TradeStatus.FILLED})
        notional = trade.qty * (trade.fill_price or trade.requested_price)
        commission = trade.commission or Decimal("0")
        self._portfolio.cash += notional - commission
        _apply_sell(self._portfolio, trade)
        self._trades.append(filled)
        return filled

    def record_approval(
        self,
        *,
        correlation_id: uuid.UUID,
        trade_id: uuid.UUID,
        status: str,
        original_notional: Decimal,
        original_qty: Decimal,
        edited_qty: Decimal | None = None,
        decided_at: datetime | None = None,
        decided_by: str = "risk_committee",
    ) -> None:
        """Capture a HITL decision in memory for offline eval and testing."""
        ts = decided_at if decided_at is not None else datetime.now(tz=UTC)
        self.approvals.append(
            _ApprovalRecord(
                correlation_id=correlation_id,
                trade_id=trade_id,
                status=status,
                original_notional=original_notional,
                original_qty=original_qty,
                edited_qty=edited_qty,
                decided_at=ts,
                decided_by=decided_by,
            )
        )

    def record_cycle(self, record: CycleAuditRecord) -> None:
        """Capture a decision cycle record in memory for offline eval and testing."""
        self.cycles.append(record)


def _apply_buy(
    portfolio: Portfolio,
    trade: Trade,
    opened_at: datetime | None = None,
) -> None:
    """Update in-memory portfolio after a buy — no FIFO lot tracking."""
    fill_price = trade.fill_price or trade.requested_price
    ts = opened_at if opened_at is not None else datetime.now(tz=UTC)
    holding = portfolio.holdings.get(trade.symbol)
    lot = Lot(symbol=trade.symbol, qty=trade.qty, cost=fill_price, opened_at=ts)
    if holding is None:
        portfolio.holdings[trade.symbol] = Holding(symbol=trade.symbol, lots=[lot])
    else:
        holding.lots.append(lot)


def _apply_sell(portfolio: Portfolio, trade: Trade) -> None:
    """Update in-memory portfolio after a sell — remove quantity FIFO."""
    holding = portfolio.holdings.get(trade.symbol)
    if holding is None:
        return
    remaining = trade.qty
    for lot in list(holding.lots):
        if remaining <= Decimal("0"):
            break
        if lot.qty <= remaining:
            remaining -= lot.qty
            lot.qty = Decimal("0")
        else:
            lot.qty -= remaining
            remaining = Decimal("0")
