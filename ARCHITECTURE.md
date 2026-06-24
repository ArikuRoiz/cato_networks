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

## Pipeline

```
research + technical (parallel)
        → debate (bull ⇄ bear × N rounds)
        → Research Manager (decide direction + conviction)   [SOLE decider — LLM]
        → pm (deterministic sizing via size_position tool)
        → risk (RISK GUARDRAIL + HITL)
             ↓ every cycle (hitl_mode="always")
             HITL interrupt → human Approve / Reject→(Buy|Sell|Hold)   (RECORDED)
             ↓ approved / overridden
        → Execution (atomic ledger write, NYSE calendar-gated)
        → Reporting agent (Excel + Slack dispatch)
        → Synthesis (LLM investment memo)
        → Judge (independent coherence audit, recorded)
        → END
```

`research` and `technical` run **in parallel** from START. Both feed the `debate_bull` node.
`Research Manager` is the **sole** direction-decision agent; the deterministic `pm` node calls
`size_position` to convert recommendation + conviction into a share quantity capped by `RiskPolicy`.
The `risk` node re-validates before every ledger write — the LLM cannot skip it.

### Agent classification

| Node | Pattern | LLM? |
|---|---|---|
| `research` | **Tool-using agent** — calls `search_news` closure (inline in agent); grounds cited claims | haiku |
| `technical` | **Tool-using agent** — calls `price_indicators` closure; produces structured bias signal | haiku |
| `debate_bull` / `debate_bear` | **LLM agent, two roles** — one `DebaterAgent` class instantiated per stance; runs bull ⇄ bear turns × N rounds | haiku |
| `research_manager` | **LLM agent — SOLE decider** — adjudicates debate; outputs direction (strong_buy … strong_sell) + conviction (0–1) | sonnet |
| `pm` | **Deterministic node** — calls `size_position` tool; conviction × NAV → share qty; capped by `RiskPolicy` (≤ 10% NAV) | — |
| `risk` | **Mandatory gate** — wraps `RiskAgent`; re-validates RiskPolicy; interrupts every cycle for human Approve/override (`hitl_mode="always"`); hard limits still enforced; LLM cannot bypass | — |
| `execution` | **Pure function** — atomic ledger commit (cash + FIFO lot + audit + idempotency key); NYSE calendar-gated | — |
| `reporting` | **Agent** — dispatches Excel + Slack report with real NAV/P&L; numbers come from ledger + live prices | — |
| `synthesis` | **LLM agent** — writes the cited investment memo for the full cycle | sonnet |
| `judge` | **LLM agent, standalone auditor** — scores full-cycle coherence 1–5; verdict recorded; feeds eval process-quality metrics | sonnet |

**Tool-using agents** (`research`, `technical`) call `LLM.complete_with_tools()`: the LLM controls which tools to call, issues multiple calls if needed, and decides when it has enough information before producing its final response. All other LLM agents use the single-shot `LLM.complete()`.

### Portfolio Manager is not an LLM agent

`PortfolioManagerAgent` (the LLM-based PM) is dissolved. Sizing math lives in the deterministic
`size_position` tool called from the `pm` node:
`size_position(recommendation, conviction, nav, price, policy) → qty`. The `portfolio_manager/`
package remains as a schema source (`TradeProposal`, `Hold`) shared by the pm and risk nodes.
There is exactly **one** direction-decision maker (Research Manager). The sizing step never flips
a buy/sell into Hold for directional reasons — only because qty rounds to 0 or RiskPolicy caps it.

### Mandatory guardrails (deterministic — LLM cannot skip)

- **Risk guardrail (`risk` node)** — runs in every pipeline path before any ledger write; re-validates
  against `RiskPolicy`; under `hitl_mode="always"` it interrupts every cycle for human approval (the
  `"threshold"` mode routes only > 5% NAV) even if the Manager already self-checked. Hard limits are
  enforced at execution regardless of the human's decision.
- **Execution** — single ACID transaction keyed by `idempotency_key`; the only thing that moves money;
  NYSE calendar-gated (`CycleOutcome.REJECTED_MARKET_CLOSED` when market is closed).
- **Cross-cutting:** injection scan on retrieved text · token-budget circuit breaker (`TokenBudgetLLM`) ·
  output-schema validation.

### HITL — human approves every cycle, with override

The default policy is **human-approves-every-cycle, with override** (`RiskPolicyConfig.hitl_mode`
defaults to `"always"`; a `"threshold"` mode that only pauses above 5% NAV still exists). The
`risk` node fires a LangGraph `interrupt()` and checkpoints state, then the human either:

- **Approves** — execute the desk's recommended action (buy/sell as sized, or hold = no trade); or
- **Rejects → picks an alternative action** — Buy, Sell, or Hold, overriding the desk.

`firm.orchestration.hitl.resume_decision(graph, thread_id, decision)` carries the structured
`HITLDecision` (`APPROVE`, `OVERRIDE_BUY`, `OVERRIDE_SELL`, `OVERRIDE_HOLD`, `EXPIRE`) back into the
risk node via `Command(resume=…, update={"hitl_status": …})`. An override **rewrites the cycle's
`trade_proposal`** (`_proposal_from_approved` / `_override_hold_proposal` in `nodes.py`) so the
synthesis memo, report, and judge all describe **what actually executed** — an override-buy never
yields a stale "Hold / no position change" memo.

**Hard limits still enforce at the execution guardrail even after approval.**
`LedgerGuardrail.enforce_hitl_approved` bypasses only the *soft* HITL gate (`HITLRequired`); a hard
`RiskPolicy` breach (`Rejected`) still raises `LimitExceeded` and the trade fails.

Every decision is recorded (audit log + `ApprovalRow`) so override rate and HITL latency are
measurable process metrics. The **approval channel is a pluggable skill**: Telegram is the shipped
operator channel; Slack / email / SMS are drop-in adapters behind the same `resume_decision` core.

### Live run + Telegram operator service

Beyond the offline demo there are two live entry points (both build the same live pipeline —
yfinance market data + yfinance news ingestion + live Anthropic + Postgres + PostgresSaver):

- **`firm run --tickers … --lookback-days N [--hitl console|telegram|auto]`** — live one-shot. Pulls
  N days of real bars + news (`NewsIngestionAgent` via yfinance, upserted to pgvector), runs the graph
  once per ticker against the live Postgres ledger, and reports via Excel + Slack. The risk node
  interrupts each cycle; the operator approves/overrides on the console or via a single Telegram card.
- **`firm bot`** (`make bot`) — persistent Telegram operator service. The operator types `/run <ticker>`
  (or a bare ticker) → a rich approval card (recommendation + 💡 Why / 👍 Pros / 👎 Cons) → tap
  **Approve**, or **Reject → Buy / Sell / Hold** alternatives → `resume_decision` executes → a
  "what I did & why" message + run report come back to the chat. It uses a single `getUpdates`
  long-poll (**no webhook**) and is durable via the PostgresSaver checkpoint. See
  [`docs/telegram_flow.md`](docs/telegram_flow.md).

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
  llm_offline.py       build_offline_llm() + GracefulLLM; CASSETTE_MODE=record to network
  llm_token_budget.py  TokenBudgetLLM — circuit-breaker wrapping any LLM port
  fakes.py             In-memory fakes for unit tests (FakeLLM, FakeMarketData, …)
  embeddings.py        SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim) — real embeddings
  evidence_pgvector.py Live — pgvector similarity search using SentenceTransformerEmbedder
  market_data_frozen.py Replay — frozen CSV bars (offline demo/eval)
  market_data_live.py   Live — yfinance OHLCV (production)
  telegram.py           TelegramHITL + BotService — getUpdates long-poll, inline-keyboard approval
  report/
    file.py    FileReportSink — text file + alerts.log
    slack.py   SlackReportSink — Block Kit messages (Approve/Reject/Edit buttons)
    excel.py   ExcelReportSink — openpyxl workbook
    multi.py   MultiReportSink — fan-out to ExcelReportSink + SlackReportSink
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
    research/                Tool-using: search_news closure, Evidence | Refusal
    technical/               Tool-using: price_indicators closure, TechnicalSignal
    debater/                 LLM: one DebaterAgent class (stance arg); DebaterCase | DebaterFailure
    research_manager/        LLM: SOLE decider — adjudicates debate → direction + conviction
    reporting/               Dispatches Excel + Slack report; real NAV/P&L from ledger + prices
    synthesis/               LLM: writes the cited investment memo for the cycle
    execution/               Atomic ledger commit; NYSE calendar-gated
    judge/                   LLM: standalone independent auditor — coherence score 1–5, recorded
    portfolio_manager/       Schemas only: TradeProposal, Hold (used by pm + risk nodes)
  tools/
    size_position.py         Deterministic sizing: conviction × NAV → qty, capped by RiskPolicy
  orchestration/
    state.py     GraphState TypedDict — JSON-serialisable envelope
    graph.py     build_graph(checkpointer, ports) → CompiledStateGraph
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
  cli.py         argparse CLI: run (live) / bot (Telegram) / demo / dev / seed / trace / web
  bot/           Telegram operator service (getUpdates loop, approval cards, formatters)
```

### Three environments

| Environment | LLM | Market data | News | Persistence |
|---|---|---|---|---|
| **(1) Offline historic replay** (`firm demo` / `firm dev`, eval/CI) | Cassette via `build_offline_llm` (FakeLLM fallback) | `FrozenMarketData` (committed CSV bars) | Frozen corpus (`FakeEvidenceStore`) | In-memory (`MemorySaver`, `_FakeLedger`) |
| **(2) Offline agent tests** (pytest) | `FakeLLM` | `FakeMarketData` | `FakeEvidenceStore` | In-memory |
| **(3) Live production** (`firm run` / `firm bot`, web `/api/run`) | live Anthropic (`AnthropicLLM`, graceful + token-budget wrapped) | `LiveMarketData` (yfinance) | `NewsIngestionAgent` (yfinance) → pgvector | Postgres ledger + `PostgresSaver` checkpoints |

The demo/eval are fully offline; only `CASSETTE_MODE=record` touches the network there. Live
production uses real embeddings (`SentenceTransformerEmbedder`, all-MiniLM-L6-v2, 384-dim) in
pgvector and a stable `FIRM_PORTFOLIO_ID` (`ensure_portfolio`) so portfolio state and filled trades
persist across runs. Every decision cycle writes `decision_cycles` + `audit_log` for any outcome.
All entry points run the same `firm.orchestration.graph.build_graph`.

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
  → research node:  LLM calls search_news closure × N → Evidence (cited)
  → technical node: LLM calls price_indicators closure → TechnicalSignal    (parallel)
  → debate_bull / debate_bear: DebaterAgent(bull) + DebaterAgent(bear) alternate × MAX_ROUNDS
  → research_manager: adjudicates debate → direction (strong_buy…strong_sell) + conviction (0–1)
  → pm node: size_position(recommendation, conviction, nav, price, policy) → TradeProposal | Hold
  → risk node: RiskAgent.check_trade; interrupt for human approval (every cycle under hitl_mode="always")
      (HITL: checkpoint → approval card (Telegram/console/Slack) → await Approve / Reject→Buy|Sell|Hold
       → override rewrites trade_proposal → ApprovalRow recorded → resume_decision)
  → execution: atomic ledger commit  [single ACID txn, idempotency key; NYSE calendar-gated]
  → reporting: dispatches DailyReport to MultiReportSink → ExcelReportSink + SlackReportSink
  → synthesis: LLM writes cited investment memo
  → judge: independent LLM scores full-cycle coherence 1–5 → Verdict recorded
```

**Three correctness-critical invariants:**
1. `execution` is a single ACID transaction keyed by `idempotency_key` — retried execution is a no-op.
2. The `risk` node re-validates against `RiskPolicy` after every HITL resume — a stale approval against a moved price is caught.
3. `Research Manager` is the only component that decides direction — the `pm` node / `size_position` never re-derives a signal.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Tool-using agents for research + technical | LLM controls its own information gathering; multiple search queries produce better evidence than a single hardcoded query |
| Single decision-maker (Research Manager) | Dissolving Portfolio Manager removes the two-signal conflict; the Judge no longer needs to flag PM-vs-Manager incoherence |
| `size_position` is deterministic, not an agent | Sizing math (conviction × NAV, RiskPolicy caps) has a correct answer; no LLM judgment needed; testable in isolation |
| One `DebaterAgent` class, two roles | Eliminates duplication; stance is a constructor arg; output schema unified as `DebaterCase | DebaterFailure` |
| Tools layer (`src/firm/tools/`) | Deterministic capabilities (`size_position`) are importable + unit-testable without a graph; tool closures for LLM agents live inline in each agent |
| Risk guardrail is mandatory and LLM-unreachable | Defense-in-depth: the Manager may self-check risk; the guardrail re-validates unconditionally before ledger write |
| HITL = approve-every-cycle with override | `hitl_mode="always"` interrupts every cycle; human Approves or Rejects→picks Buy/Sell/Hold; an override rewrites `trade_proposal` so downstream nodes describe the executed action. Every decision persisted as `ApprovalRow` → override rate + latency are measurable. Approval channel is a pluggable skill (Telegram shipped) |
| Reporting dispatches; synthesis writes the memo | `reporting` fetches prices + ledger and sends the structured report (Excel + Slack); `synthesis` (LLM) writes the cited investment memo |
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
