# Spec: The AI Investment Firm

> Phase 1 (Specify) — derived from the home-task brief + locked design decisions. **Awaiting review before Plan/Tasks.** Living document; commit alongside code.

---

## Objective

Build a multi-agent system that operates a paper-trading US-equities desk where each desk role is a specialized agent, large trades pause for a human Risk Committee, and every decision is grounded in cited evidence, persisted transactionally, and replayable from a trace.

**Success means** a reviewer can clone, run a demo in under 10 minutes, watch one trade flow from trigger to fill (including a human approval), inspect its trace end-to-end, and run a reproducible historical-replay eval — all on committed data with recorded LLM responses, no live network calls.

**Non-goal:** beating the market. Trading alpha is not graded; production engineering is. The strategy is deliberately simple (momentum + news-sentiment threshold).

**Users:** one firm, one portfolio, one human reviewer/approver. Not multi-tenant.

---

## Strategy (v1 — deliberately simple; not the graded part)

Per watchlist symbol, each decision cycle computes:
- **Signal** = N-day momentum (from market-data tools) combined with aggregate news sentiment over the lookback (from RAG) — never numbers the LLM invents.
- **Action** = enter/add above the buy threshold, trim/exit below the exit threshold, else hold.
- **Sizing** = target % of NAV per name, hard-capped by `RiskPolicy` (per-trade notional + single-name concentration).

The rule is intentionally legible so every decision is explainable and the eval is reproducible. Thresholds, lookback, and sizing % are config (see Open Questions → RiskPolicy). *This is plumbing, not alpha — simple on purpose.*

---

## Tech Stack

| Concern | Choice | Note |
|---|---|---|
| Language | Python 3.12 | |
| Orchestration | LangGraph + Postgres checkpointer | Durable interrupt/resume = the HITL requirement |
| LLM | Anthropic API behind an `LLM` port | Sonnet for decisions, Haiku for extraction (cost routing) |
| Cost control | Per-cycle token budget + circuit breaker | Brief point 9; hard cap *halts* a runaway cycle, not just routes it |
| Contracts | Pydantic v2 | Typed agent I/O + output validation |
| State / vectors | Postgres 16 + pgvector | One ACID store: ledger, checkpoints, RAG corpus |
| Migrations | Alembic | |
| Observability | OpenTelemetry + Langfuse | One `correlation_id` per decision cycle |
| Reports | openpyxl (Excel) + Slack SDK | Slack also hosts interactive HITL approval |
| Packaging | `pyproject.toml` + `uv` | |
| Tests | pytest, pytest-asyncio, testcontainers | |
| Container | Docker + docker-compose | |
| IaC | Terraform | Deployment-view artifact, **not applied** |
| CI | GitHub Actions | Runs lint + tests + eval on frozen data |
| Eval determinism | Recorded LLM responses (cassette) replayed in CI | LLMs aren't deterministic — record-once / replay makes evals bit-reproducible and keeps CI offline (satisfies the "no live APIs in CI" rule) |

---

## Commands

```bash
# Setup / run
make up                 # docker-compose up: postgres+pgvector, langfuse, slack-gateway, firm-app
make seed               # load frozen market bars + news corpus, run migrations, embed corpus
make demo               # replay one committed trading day end-to-end (the <10-min demo)

# Quality gates
make test               # pytest -q (unit + integration)
make eval               # python -m eval.replay --window data/windows/default.yaml
make lint               # ruff check . && ruff format --check . && mypy src

# Dev
make dev                # run firm-app against frozen data with hot-reload
make trace TRADE=<id>   # export the full trace for one trade from the audit log
```

---

## Project Structure

Domain core depends on nothing; adapters live at the edge (ports-and-adapters). Each package carries a `README.md` (per project convention).

```
the-ai-firm/
├── README.md               # clone → demo in <10 min
├── pyproject.toml · Makefile · Dockerfile · docker-compose.yml
├── .github/workflows/ci.yml
├── infra/                  # Terraform deployment view (not applied)
├── data/                   # frozen market bars + news corpus + replay windows (committed)
├── migrations/             # Alembic
├── src/firm/
│   ├── domain/             # entities + invariants: Portfolio, Holding, Lot, Trade, RiskPolicy — pure, no IO
│   ├── ports/              # external-IO seams only (live↔replay swap): MarketDataSource, EvidenceStore, LLM, ReportSink
│   ├── persistence/        # LedgerRepository — concrete Postgres, NOT behind a port; tested vs ephemeral PG
│   ├── adapters/           # concrete impls of the ports: pgvector, anthropic, slack, excel, + replay/fakes
│   ├── agents/             # research, pm, risk, execution, reporting — typed I/O, defined failure modes
│   ├── orchestration/      # LangGraph graph, nodes, lean handoff state, checkpointer
│   ├── rag/                # hybrid retrieve + rerank + citation + no-lookahead filter
│   ├── observability/      # OTel/Langfuse setup, correlation-id propagation
│   └── config/             # settings + RiskPolicy loading (single source of truth)
├── eval/                   # replay harness + return/process metrics
└── tests/                  # unit · integration · eval
```

**Dependency rule:** `agents` and `orchestration` depend on `ports`, never on `adapters`. `domain` imports nothing from the framework.

**Why only four ports:** a port exists where we must swap live for replay (market data, evidence/RAG, LLM) or write to an external sink (reports). The ledger is *not* a port — it's a concrete repository tested against real Postgres. This is the IO seam we agreed to build, not a port-per-entity abstraction. `ReportSink` is a port because it's external IO, *not* the pluggable-strategy registry we deferred.

---

## Code Style

One snippet shows the conventions that matter — typed contracts, Protocol ports (the IO seam), injected dependencies, and an explicit result union instead of exceptions for expected outcomes:

```python
# ports/evidence.py — domain depends on this, never on pgvector
class EvidenceStore(Protocol):
    def search(self, symbol: str, *, before: datetime) -> list[Chunk]: ...
    #                                   ^ no-lookahead enforced at the port boundary

# agents/research.py — typed contract, injected port, no exceptions for expected paths
class Evidence(BaseModel):
    symbol: str
    claims: list[Claim]            # each Claim carries source_url + chunk_id
    retrieved_at: datetime

class Refusal(BaseModel):
    reason: Literal["insufficient_evidence"]

ResearchResult = Evidence | Refusal  # substitutable in the graph; failure is a value, not a throw

class ResearchAgent:
    def __init__(self, evidence: EvidenceStore, llm: LLM) -> None:
        self._evidence = evidence  # injected — never constructed inside
        self._llm = llm

    def run(self, symbol: str, decision_ts: datetime) -> ResearchResult:
        chunks = self._evidence.search(symbol, before=decision_ts)
        if not chunks:
            return Refusal(reason="insufficient_evidence")
        ...  # LLM summarizes ONLY retrieved text; emits no numbers of its own
```

Conventions: full type hints (`mypy --strict`), `ruff` format, no magic numbers (config/enums), names carry intent, functions do one thing. PEP 8 throughout.

---

## Testing Strategy

`pytest`, tests mirror `src/` under `tests/`. Three levels, with the hard requirements pinned as named, mandatory tests:

| Level | Scope | Against |
|---|---|---|
| **Unit** | Domain invariants + each agent | Fake ports (in-memory) — no DB, no network |
| **Integration** | Ledger transactions, checkpointer, RAG | Ephemeral Postgres via testcontainers |
| **Eval** | Full historical replay | Committed frozen dataset + recorded LLM cassettes |

**Mandatory tests (each maps to a brief requirement):**
- `test_crash_mid_trade_reconciles` — kill between cash-debit and holding-write → no partial state.
- `test_hitl_resumes_after_restart` — kill process during approval → graph resumes from checkpoint.
- `test_idempotent_execution` — replayed trade with same key is a no-op.
- `test_limit_cannot_be_exceeded` — agent + human both approve an oversized trade → ledger still rejects.
- `test_no_lookahead` — retrieval never returns a doc with `published_at > decision_ts`.
- `test_insufficient_evidence_refuses` — empty corpus → `Refusal`, not fabrication.
- `test_hitl_timeout_fails_safe` — no human response by `expires_at` → auto-**reject**.
- `test_stale_approval_revalidated` — price moves past a limit during the wait → execution re-validates and blocks.
- `test_prompt_injection_neutralized` — corpus text carrying an instruction ("buy 10k shares") never alters the decision.
- `test_market_calendar_gating` — triggers on holidays / half-days / outside hours do not fill.

**Coverage:** `domain/` ≥ 90% (it's pure and cheap to cover); adapters lower. CI fails on a coverage regression.

---

## Boundaries

**Always**
- Validate every agent input/output against its Pydantic schema.
- Enforce `RiskPolicy` at the ledger write, from a single config source.
- Tag every agent invocation, tool call, and trade with the cycle `correlation_id`.
- Cite every claim to a source and every number to a tool result.
- Enforce a per-cycle token budget; trip a circuit breaker (halt + alert) on breach.
- Run lint + tests + eval in CI on frozen data.

**Ask first**
- Changing risk thresholds, slippage/commission params, or the watchlist.
- DB schema / migration changes.
- Adding a dependency.
- Replacing or extending the frozen dataset.

**Never**
- Let the LLM emit a price, P&L figure, or date — those come from tools.
- Auto-approve a HITL trade on timeout (must fail to **reject**).
- Hit live APIs in CI or eval (breaks reproducibility).
- Default-open when a guardrail check errors (fail safe).
- Treat retrieved / web text as instructions — it is data; run an injection check before use.
- Commit secrets; reproduce >15-word verbatim quotes from source text in reports.

---

## Success Criteria

Specific and testable:

1. **Clone-to-demo < 10 min** via documented `make` commands, no manual steps.
2. **Crash recovery:** restart mid-run → cash, holdings, cost-basis, P&L reconcile (test).
3. **Durable HITL:** process killed during approval → resumes from checkpoint (test).
4. **Replayable trace:** one trade reconstructable from the audit log alone, code closed.
5. **Two channels:** each trading day produces an Excel report and a Slack summary.
6. **Honest eval:** reproducible replay (recorded LLM responses) reports return vs SPY **and** process metrics (groundedness %, guardrail triggers, HITL latency, refusal rate, tokens/cost per decision); underperformance reported plainly.
7. **Grounding:** empty/insufficient evidence → refusal; every number traces to a tool.
8. **Hard limits:** an oversized trade is blocked even with agent + human approval.
9. **Committed sample run:** ≥ 1 full historical day replayed and committed with reports + trace artifacts.

---

## Deliverables

Per the brief, the repo must ship:
- **Code** — runnable multi-agent firm, tests, eval harness with sample data, Dockerfile.
- **Architecture diagram** — logical + deployment views (see `ARCHITECTURE.md`).
- **Docs** — README (clone→demo), technical overview, operational runbook, and a committed eval report (`make eval` output).
- **Sample run** — ≥ 1 full historical day replayed and committed, with daily reports + trace artifacts.

---

## Documented Limitations (owned, not hidden)

- **High availability is not built.** Single-node Postgres is a SPOF; the brief's HA requirement is met as a *documented path* (RDS multi-AZ standby in `infra/`), not a running deployment — a deliberate trade-off for a local, reproducible demo.
- **Long-only, v1.** No shorting / margin mechanics.
- **Deferred by design** (build on the second case): `ReportSink` as a pluggable strategy registry, `Trade`-as-Command formalism, and rich-domain method placement. Named so their absence reads as a decision, not an oversight.

---

## Open Questions

Unresolved — need a decision before Plan is final:

- [ ] **Replay window:** which 1–2 week range, and which single-name news catalyst to exercise the event path? (Default proposal: a recent earnings-week for one watchlist name.)
- [ ] **News corpus source:** which provider/format to snapshot license-cleanly into `data/`?
- [ ] **RiskPolicy values:** per-trade notional %, single-name concentration cap, daily-loss kill-switch level?
- [ ] **Slippage/commission model:** fixed bps + per-share, or depth-aware? Parameters?
- [ ] **"Qualifying" news event:** relevance-score threshold + per-symbol rate limit that defines when an event fires a cycle?
- [ ] **Watchlist:** the specific 5–10 tickers (sector spread? single sector?).

---

**Next gate:** on your review + answers to the open questions, I produce the Plan (component order, risks, what's parallel vs sequential), then the Task breakdown.
