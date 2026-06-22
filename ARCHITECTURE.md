# The AI Investment Firm — Architecture

> Runtime: **local Docker** · Cadence: **scheduled checkpoints + news-event triggers** · Watchlist: **~5–10 tickers**

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

## Pipeline — 11-Node LangGraph Graph

```
START → research ──┐
START → technical ─┴→ debate_bull → debate_bear ─┬→ debate_bull  (loop ≤ MAX_ROUNDS)
                                                  └→ research_manager → pm → risk
                     → (approved) execution → reporting → synthesis → judge → END
                     → (rejected)            reporting → synthesis → judge → END
```

`research` and `technical` run **in parallel** from START. `debate_bull` fans in from both.  
After the debate loop, `research_manager` adjudicates and produces a `ResearchPlan` that feeds `pm`.

### Agent classification

| Node | Pattern | LLM? |
|---|---|---|
| `research` | **Tool-using agent** — calls `search_news` autonomously, multiple queries | haiku |
| `technical` | **Tool-using agent** — calls `get_price_and_indicators`, produces structured signal | haiku |
| `debate_bull` | LLM agent — single-shot bullish argument | haiku |
| `debate_bear` | LLM agent — single-shot bearish rebuttal | haiku |
| `research_manager` | LLM agent — adjudicates debate, outputs `ResearchPlan` | sonnet |
| `pm` | Deterministic — weighted signal formula, sizing rules; indirect LLM via `derive_sentiment` | indirect |
| `risk` | **Pure function** — rule-based policy check, no LLM | — |
| `execution` | **Pure function** — ledger I/O + guardrail, no LLM | — |
| `reporting` | **Pure function** — build + dispatch `DailyReport`, no LLM | — |
| `synthesis` | LLM agent — writes investment memo | sonnet |
| `judge` | LLM agent — scores decision-cycle coherence | sonnet |

**Tool-using agents** (`research`, `technical`) call `LLM.complete_with_tools()`: the LLM controls which tools to call, issues multiple calls if needed, and decides when it has enough information before producing its final response. All other LLM agents use the single-shot `LLM.complete()`.

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
    research/                Tool-using: search_news tool, Evidence | Refusal
    technical/               Tool-using: get_price_and_indicators tool, TechnicalSignal
    bull_researcher/         LLM: bullish argument
    bear_researcher/         LLM: bearish rebuttal
    research_manager/        LLM: adjudication → ResearchPlan
    portfolio_manager/       Deterministic: weighted signal + sizing
    risk/                    Pure function: policy check → ApprovedTrade | HITLRequired | Rejected
    execution/               Pure function: guardrail + ledger write → Fill
    reporting/               Pure function: build DailyReport + dispatch
    synthesis/               LLM: investment memo
    judge/                   LLM: coherence score
  orchestration/
    state.py     GraphState TypedDict — JSON-serialisable envelope
    graph.py     build_graph(checkpointer, risk_policy, ports) → CompiledStateGraph
    nodes.py     make_*_node factories; NodePorts DI container
    checkpointer.py  Postgres checkpointer setup
  domain/
    enums.py     StrEnum constants (TradeSide, TechnicalBias, CycleOutcome, …)
    trade.py     Trade, Portfolio, Holding, Lot domain entities
    guardrails.py InjectionGuard, LedgerGuardrail
  strategy/
    signals.py       derive_sentiment, technical_score, floor_qty
    indicators.py    compute_indicators (RSI/MACD/Bollinger — pure math)
  persistence/
    ledger.py    LedgerRepository — single ACID boundary, slippage applied internally
  ports/         Protocol interfaces + value types
  adapters/      Concrete implementations
  utils/
    json.py      parse_json_dict
    uuid.py      str_to_uuid
  config/        RiskPolicyConfig, watchlist
  cli.py         Click CLI: run / backtest / ingest
```

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
  → research node: LLM calls search_news(query) × N → synthesises Evidence
  → technical node: LLM calls get_price_and_indicators() → TechnicalSignal
  → debate_bull / debate_bear (× MAX_ROUNDS)
  → research_manager: adjudicates → ResearchPlan
  → pm: momentum + sentiment + technical → TradeProposal | Hold
  → risk: policy check → ApprovedTrade | HITLRequired | Rejected
      (if HITL: checkpoint → Slack → await human → resume)
  → execution: guardrail + ledger.buy()/sell() → Fill  [single ACID txn]
  → reporting: DailyReport → FileReportSink / SlackReportSink / ExcelReportSink
  → synthesis: LLM writes investment memo
  → judge: LLM scores coherence → Verdict
```

**Two correctness-critical moments:**
1. `execution` is a single ACID transaction keyed by `idempotency_key` — retried execution is a no-op.
2. `risk` re-validates limits at execution time — a stale human approval against a moved price is caught.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Tool-using agents for research + technical | LLM controls its own information gathering; multiple search queries produce better evidence than a single hardcoded query |
| Pure functions for risk, execution, reporting | No LLM involved; the graph owns the sequencing — not the model |
| Pipeline graph over supervisor | Workflow is deterministic; supervisor adds routing nondeterminism that fights replayability (FR-6) |
| Debate loop (bull + bear + manager) | Forces adversarial evidence gathering before PM; ResearchPlan conviction score feeds signal weighting |
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
