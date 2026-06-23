# The AI Investment Firm — Architecture

> Runtime: **local Docker** · Cadence: **scheduled checkpoints + news-event triggers** · Watchlist: **~5–10 tickers**
>
> **Target architecture; build tracked in `docs/REFACTOR_TICKETS.md`.**

---

## Problem

A multi-agent paper-trading desk: agents research, debate, decide, size, and execute trades with realistic fills. Large trades pause for human approval. Every decision is grounded in cited evidence, persisted transactionally, and replayable from a trace.

**The graded subject is production engineering, not trading alpha.**

---

## Functional Requirements

| ID | Capability |
|---|---|
| FR-1 | Paper portfolio with realistic fills (slippage, commission), persistent across restarts |
| FR-2 | Decision cycles on schedule (open/midday/close) and on qualifying news events |
| FR-3 | Multi-agent pipeline with typed Pydantic I/O contracts and defined failure modes |
| FR-4 | RAG-grounded decisions with citations; refuse when evidence is insufficient |
| FR-5 | Trades over threshold route to human approval; graph state survives the wait |
| FR-6 | Structured trace per agent invocation, tool call, and trade; replayable end-to-end |
| FR-7 | Daily reports via Excel + Slack |
| FR-8 | Reproducible historical replay eval (vs SPY); runs offline in CI via recorded LLM cassettes |

---

## Pipeline

```
research + technical (parallel)
        → debate (bull ⇄ bear × N rounds)
        → Research Manager (decide direction + conviction)   [SOLE decider — LLM]
        → size_position tool (deterministic sizing) + check_risk
        → [RISK GUARDRAIL]
             ↓ > 5% NAV
             HITL interrupt → human approve / edit / reject   (RECORDED)
             ↓ approved
        → Execution (atomic ledger write)
        → Reporting agent (memo + Excel / Slack)
        → Judge (independent coherence audit, recorded)
        → END
```

`research` and `technical` run **in parallel** from START. Both feed the `debate` node.
`Research Manager` is the **sole** direction-decision agent; `size_position` converts its
recommendation + conviction into a deterministic share quantity capped by `RiskPolicy`.
The `RISK GUARDRAIL` re-validates before every ledger write — the LLM cannot skip it.

### Agent classification

| Node | Pattern | LLM? |
|---|---|---|
| `research` | **Tool-using agent** — calls `search_news` (offline/online) and `fetch_live_news` (prod); grounds cited claims | haiku |
| `technical` | **Tool-using agent** — calls `price_indicators`; produces structured bias signal | haiku |
| `debater` | **LLM agent, two roles** — one `DebaterAgent` class runs bull ⇄ bear turns × N rounds | haiku |
| `research_manager` | **LLM agent — SOLE decider** — adjudicates debate; outputs direction (strong_buy … strong_sell) + conviction (0–1) | sonnet |
| `size_position` | **Deterministic tool** — conviction × NAV → share qty; capped by `RiskPolicy` (≤ 10% NAV) | — |
| `check_risk` | **Deterministic tool** — advisory wrapper around `RiskPolicy.check_trade` | — |
| `risk_guardrail` | **Mandatory gate** — re-validates RiskPolicy before any ledger write; routes > 5% NAV to HITL; LLM cannot bypass | — |
| `execution` | **Pure function** — atomic ledger commit (cash + FIFO lot + audit + idempotency key) | — |
| `reporting` | **LLM agent** — writes investment memo; builds Excel + Slack report via `make_report` tool | sonnet |
| `judge` | **LLM agent, standalone auditor** — scores full-cycle coherence 1–5; verdict recorded; feeds eval process-quality metrics | sonnet |

**Tool-using agents** (`research`, `technical`) call `LLM.complete_with_tools()`: the LLM controls which tools to call, issues multiple calls if needed, and decides when it has enough information before producing its final response. All other LLM agents use the single-shot `LLM.complete()`.

### Portfolio Manager is NOT an agent

`PortfolioManagerAgent` is dissolved. Sizing math lives in the deterministic `size_position` tool:
`size_position(recommendation, conviction, nav, price, policy) → qty`. There is exactly **one**
direction-decision maker (Research Manager). The sizing step never flips a buy/sell into Hold for
directional reasons — only because qty rounds to 0 or RiskPolicy caps it.

### Mandatory guardrails (deterministic — LLM cannot skip)

- **Risk guardrail** — runs in every pipeline path before any ledger write; re-validates against
  `RiskPolicy`; routes > 5% NAV to HITL even if the Manager already self-checked.
- **Execution** — single ACID transaction keyed by `idempotency_key`; the only thing that moves money.
- **Cross-cutting:** injection scan on retrieved text · token-budget circuit breaker · output-schema validation.

### HITL is a feedback loop

Every human approve / edit / reject is **recorded** (audit log + trace via `ApprovalRow`) so
override rate and HITL latency are measurable process metrics and decisions feed future training.

---

## Ports & Adapters

Agents depend on **protocol interfaces (ports)**, never on drivers. Each port has a live and a replay adapter.

```
ports/
  llm.py           LLM — complete() + complete_with_tools()
  evidence.py      EvidenceStore — search() + embed_and_store()
  market_data.py   MarketDataSource — get_bar() / get_bars()
  report.py        ReportSink — send_daily_report() / send_hitl_request() / send_alert()

adapters/
  llm_anthropic.py     Live — full tool-calling loop via Anthropic API
  llm_cassette.py      Replay — record once, serve from JSONL; supports tool-loop responses
  fakes.py             In-memory fakes for unit tests (FakeLLM, FakeMarketData, …)
  evidence_pgvector.py Live — pgvector similarity search
  market_data_live.py  Live — OHLCV feed
  market_data_frozen.py Replay — frozen Parquet bars
  report/
    file.py    FileReportSink — text file + alerts.log
    slack.py   SlackReportSink — Block Kit messages
    excel.py   ExcelReportSink — openpyxl workbook
```

`ReportSink` rendering (text formatting, Block Kit builders, Excel sheet writers) is separated from delivery: the render functions sit at the top of each file and are callable independently of the sink class.

**The ledger is a concrete repository, not a port.** Tested against real Postgres; abstracting it buys nothing.

---

## LLM Port — Two Completion Modes

```python
# Single-shot (all agents except research + technical)
llm.complete(messages, model="haiku", max_tokens=512) -> LLMResponse | LLMError

# Tool-calling loop (research, technical)
llm.complete_with_tools(
    messages, tools=[ToolDef(...)], executors={"search_news": fn},
    model="haiku", max_tokens=1024, max_rounds=6,
) -> LLMResponse | LLMError
```

`complete_with_tools` runs the Anthropic tool-use loop internally: LLM calls a tool → executor runs → result fed back → repeat until the LLM stops requesting tools or `max_rounds` is reached. The final text response is returned as a regular `LLMResponse`. `FakeLLM` and `CassetteLLM` implement the method: fake skips tool execution; cassette records/replays the final response keyed on the tool names.

---

## Source Layout

```
src/firm/
  agents/
    base.py                  BaseAgent[InputT, OutputT] ABC
    research/                Tool-using: search_news (+ fetch_live_news in prod), Evidence | Refusal
    technical/               Tool-using: price_indicators tool, TechnicalSignal
    debater/                 LLM: one DebaterAgent class; runs bull ⇄ bear turns × N rounds
    research_manager/        LLM: SOLE decider — adjudicates debate → direction + conviction
    reporting/               LLM: writes investment memo + dispatches via make_report tool
    judge/                   LLM: standalone independent auditor — coherence score 1–5, recorded
  tools/
    search_news.py           Retrieves cited news chunks from the evidence store
    fetch_live_news.py       Production: fetches live news, appends to corpus
    price_indicators.py      RSI / MACD / Bollinger from market-data adapter
    size_position.py         Deterministic sizing: conviction × NAV → qty, capped by RiskPolicy
    check_risk.py            Advisory wrapper: RiskPolicy.check_trade (pre-sizing self-validation)
    make_report.py           Build + dispatch DailyReport to FileReportSink / SlackReportSink / ExcelReportSink
    ledger_commit.py         Atomic ledger write — the only thing that moves money
  orchestration/
    state.py     GraphState TypedDict — JSON-serialisable envelope
    graph.py     build_graph(checkpointer, risk_policy, ports) → CompiledStateGraph
    nodes.py     make_*_node factories; NodePorts DI container
    checkpointer.py  Postgres checkpointer setup
  domain/
    enums.py     StrEnum constants (TradeSide, TechnicalBias, CycleOutcome, …)
    trade.py     Trade, Portfolio, Holding, Lot domain entities
    guardrails.py InjectionGuard, LedgerGuardrail, TokenBudgetCircuitBreaker, OutputSchemaValidator
  strategy/
    indicators.py    compute_indicators (RSI/MACD/Bollinger — pure math)
  persistence/
    ledger.py    LedgerRepository — single ACID boundary, slippage applied internally
    models.py    ApprovalRow (HITL recording), audit log rows
  ports/         Protocol interfaces + value types
  adapters/      Concrete implementations (see Ports & Adapters section)
  utils/
    json.py      parse_json_dict
    uuid.py      str_to_uuid
  config/        RiskPolicyConfig, watchlist
  cli.py         Click CLI: run / backtest / ingest
```

### Three environments

| Tier | LLM | Market data | News |
|---|---|---|---|
| **Historic replay (offline / CI)** | Cassette | Frozen Parquet | Frozen corpus |
| **Agents on offline input** | FakeLLM / cassette | FakeMarketData | Frozen corpus |
| **Production (live)** | Anthropic API | yfinance | `fetch_live_news` → appended to corpus |

Live findings are appended to the corpus so each future offline eval replay is progressively richer.

---

## Data Stores

- **Postgres** — ledger, trades, decisions, audit log, LangGraph checkpoints. One ACID boundary: trade write and graph checkpoint commit together.
- **pgvector** — news corpus embeddings. At ~5–10K chunks, no dedicated vector DB needed; `published_at` filter enforces the no-lookahead invariant at the SQL layer.
- **Frozen files (Parquet/CSV, committed)** — market data + news corpus for offline eval. CI never calls a live feed.
- **No Redis** — no hot-key, no cache-coherency issue at 10 symbols.

---

## Dataflow — One Trade, Trigger to Fill

```
Trigger
  → research node:  LLM calls search_news × N [+ fetch_live_news in prod] → Evidence (cited)
  → technical node: LLM calls price_indicators() → TechnicalSignal              (parallel)
  → debater: DebaterAgent(bull) + DebaterAgent(bear) alternate × MAX_ROUNDS
  → research_manager: adjudicates debate → direction (strong_buy…strong_sell) + conviction (0–1)
  → size_position(recommendation, conviction, nav, price, policy) → TradeProposal | Hold  [deterministic]
  → check_risk(trade, portfolio, policy) → advisory validation
  → RISK GUARDRAIL: mandatory re-validation; > 5% NAV → HITL interrupt
      (HITL: checkpoint → Slack request → await human approve/edit/reject → ApprovalRow recorded → resume)
  → execution: ledger_commit (cash debit + FIFO lot + audit row)  [single ACID txn, idempotency key]
  → reporting: LLM writes memo + make_report → FileReportSink / SlackReportSink / ExcelReportSink
  → judge: independent LLM scores full-cycle coherence 1–5 → Verdict recorded
```

**Three correctness-critical invariants:**
1. `execution` / `ledger_commit` is a single ACID transaction keyed by `idempotency_key` — retried execution is a no-op.
2. `RISK GUARDRAIL` re-validates at execution time — a stale human approval against a moved price is caught.
3. `Research Manager` is the only component that decides direction — `size_position` never re-derives a signal.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Tool-using agents for research + technical | LLM controls its own information gathering; multiple search queries produce better evidence than a single hardcoded query |
| Single decision-maker (Research Manager) | Dissolving Portfolio Manager removes the two-signal conflict; the Judge no longer needs to flag PM-vs-Manager incoherence |
| `size_position` is deterministic, not an agent | Sizing math (conviction × NAV, RiskPolicy caps) has a correct answer; no LLM judgment needed; testable in isolation |
| One `DebaterAgent` class, two roles | Eliminates `bull_researcher` / `bear_researcher` duplication; stance is a constructor arg; schemas merge into one `CycleSnapshot` |
| Tools layer (`src/firm/tools/`) | Deterministic capabilities are importable + unit-testable without a graph; agents call tools via `complete_with_tools`; test fakes stub at tool level |
| Risk guardrail is mandatory and LLM-unreachable | Defense-in-depth: the Manager may self-check risk; the guardrail re-validates unconditionally before ledger write |
| HITL is a recorded feedback loop | Every approve/edit/reject persisted as `ApprovalRow` → override rate + latency are measurable process metrics |
| Reporting is an LLM agent, not a pure function | It writes a cited investment memo (graded requirement); dispatching the report (Excel + Slack) is via `make_report` tool |
| Judge is standalone | It audits the whole cycle including the memo, so it must not be the agent that wrote it; its 1–5 score feeds eval process-quality metrics |
| Pipeline graph over supervisor | Workflow is deterministic; supervisor adds routing nondeterminism that fights replayability (FR-6) |
| Single Postgres for ledger + checkpoints | One ACID boundary; trade write and checkpoint commit together — durability by construction |
| LLM port with two modes | `complete()` for single-shot agents, `complete_with_tools()` for tool agents; both cassette-recordable so eval stays offline |
| Adapters in `report/` subdirectory | Rendering (text/blocks/cells) separated from delivery; render functions are top-level and independently callable |
| StrEnum for all domain string literals | Eliminates raw string comparisons; values compare equal to plain strings so LangGraph GraphState (str fields) stays compatible |

---

## Non-Functional Targets

| Axis | Target |
|---|---|
| Full cycle latency | p95 < 30s (excluding human HITL wait) |
| Cycles/day | ~23 nominal, hard cap 50 |
| LLM calls/cycle | ~10–15 (tool-using agents may issue multiple calls) |
| Token budget | Hard cap 50K/cycle; ~1–2M/day |
| Consistency | Serializable for the ledger |
| Durability | Zero loss for ledger, audit log, checkpoints |
| Availability | Single-node; graceful restart + checkpoint resume |
