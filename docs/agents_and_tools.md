# Agents and Tools

## At a glance

The pipeline has **6 real LLM agents** and a **deterministic tools layer**. No agent is
a disguised service. One agent decides direction; everything else is deterministic math.

**Real LLM agents:**

| Agent | Node | Model | Output |
|---|---|---|---|
| Research | `research` | haiku | `Evidence \| Refusal` |
| Technical | `technical` | haiku | `TechnicalSignal \| TechnicalUnavailable` |
| Debater (bull role) | `debate_bull` | haiku | `DebaterCase \| DebaterFailure` |
| Debater (bear role) | `debate_bear` | haiku | `DebaterCase \| DebaterFailure` |
| Research Manager | `research_manager` | sonnet | `ResearchPlan \| ResearchManagerFailure` |
| Synthesis | `synthesis` | sonnet | `SynthesisReport \| SynthesisFailure` |
| Judge | `judge` | sonnet | `Verdict \| JudgeFailure` |

**Deterministic tools layer** (no LLM — the LLM cannot skip or reorder these):

| Tool | Where defined | Purpose |
|---|---|---|
| `size_position` | `src/firm/tools/size_position.py` | Sizing math: conviction × NAV → qty, capped by RiskPolicy |
| `search_news` | inline closure in `research` agent | `EvidenceStore.search` + injection scan; called by the LLM tool loop |
| `price_indicators` | inline closure in `technical` agent | `MarketDataSource.get_bars` + `compute_indicators`; called by the LLM tool loop |

**Mandatory deterministic gates** (cross-cutting — the LLM cannot skip them):

| Gate | Position | Action |
|---|---|---|
| Risk node (`risk`) | After `pm` node, before every ledger write | Re-validates against RiskPolicy; interrupts every cycle for human approval (`hitl_mode="always"`); hard limits enforced at execution even after approval |
| Execution node | After risk node | Atomic ledger commit (the only thing that moves money); NYSE calendar-gated |
| Injection scan | Inside `search_news` closure | Filters every retrieved chunk before the LLM sees it |
| Token-budget circuit breaker | Cross-cutting (`TokenBudgetLLM`) | Halts the pipeline if token budget is exhausted |
| Output-schema validation | Cross-cutting | Validates every agent output against its typed schema |

**Pipeline:**

```
research + technical (parallel)
        → debate_bull → debate_bear (×N rounds, ONE DebaterAgent class, two stances)
        → research_manager (decide direction + conviction)   [SOLE decider — LLM]
        → pm (deterministic sizing via size_position tool)
        → risk → HITL interrupt (every cycle) → human Approve / Reject→(Buy|Sell|Hold)  (RECORDED)
        → execution (atomic ledger write, NYSE calendar-gated; hard RiskPolicy limits enforced)
        → reporting (Excel + Slack dispatch, real NAV/P&L)
        → synthesis (LLM investment memo)
        → judge (independent coherence audit, recorded)
        → END
```

`research` and `technical` fan out in parallel from START. `debate_bull` waits for both.
The Debater is **one class** (`DebaterAgent`) instantiated with `stance="bull"` or `stance="bear"`;
both nodes share the same code and produce `DebaterCase | DebaterFailure`.

---

# Tool-using LLM agents

These call `LLM.complete_with_tools()`. The LLM decides which tools to call, issues as many
calls as it needs, and produces its final response when done. No tool query is hardcoded.

## research → `research_node`

- **Responsibility:** Retrieve news evidence for a symbol and synthesize cited factual claims (the RAG agent).
- **I/O:** `ResearchInput` → `Evidence | Refusal`
- **What it does:**
  1. Defines a `search_news` tool whose executor calls `EvidenceStore.search()` and filters every chunk through `InjectionGuard.scan()`.
  2. Runs an LLM tool loop (`complete_with_tools`, haiku, up to 6 rounds) — the LLM autonomously issues queries for different aspects (earnings, guidance, risks…).
  3. Each safe chunk is added to a `chunk_registry` (chunk_id → source_url) used later for citations.
  4. `_parse_claims()` parses the final JSON response into `Claim` objects with citation URLs.
  5. Returns `Evidence(symbol, claims, retrieved_at)`.
- **Code/deps:** `EvidenceStore.search`, `LLM.complete_with_tools`, `InjectionGuard.scan`, `ToolDef`, private `_parse_claims`, closure `chunk_registry`.
- **Failure modes:** `Refusal(LLM_ERROR_RETRYABLE | LLM_ERROR_NON_RETRYABLE | INSUFFICIENT_EVIDENCE)` — the last when no safe chunks were retrieved.

### Tool: `search_news`
```
input:  query  string  required  e.g. "NVDA earnings revenue guidance Q3"
        k      int      optional  chunks to retrieve, 1–10, default 5
returns formatted text of safe chunks, each tagged with chunk_id
```
**Why it matters:** the LLM chooses what to search for, calls multiple times, and stops when it
has enough — rather than one hardcoded query.

In **production**, the agent also has access to `fetch_live_news` (the wired-up successor to the
orphaned `news_ingestion` agent), which pulls recent Yahoo Finance headlines via `yfinance` and
upserts them into the pgvector corpus before the search loop begins.

## technical → `technical_node`

- **Responsibility:** Compute technical indicators and produce a structured bias signal with support/resistance.
- **I/O:** `TechnicalInput` → `TechnicalSignal | TechnicalUnavailable`
- **What it does:**
  1. Defines a `price_indicators` tool whose executor calls `MarketDataSource.get_bars()` for the LLM-requested lookback (max 90 days).
  2. Passes bars to `compute_indicators()` (RSI, MACD, Bollinger, avg volume); stores `indicators_snapshot` + `last_close` in closures.
  3. Runs the tool loop (`complete_with_tools`, haiku, ≤3 rounds); the LLM returns JSON `headline/body/bias/key_support/key_resistance`.
  4. `_parse_signal()` merges LLM prose with the **deterministically computed** indicator numbers and derives `MACDCross` from the histogram sign.
- **Code/deps:** `MarketDataSource.get_bars`, `LLM.complete_with_tools`, `compute_indicators` (strategy), `parse_json_dict`, `MACDCross`/`TechnicalBias` enums, private `_parse_signal`.
- **Failure modes:** `TechnicalUnavailable` — `llm_error`, `insufficient price history` (<14 bars or tool never called), or `llm returned invalid JSON`.

### Tool: `price_indicators`
```
input:  lookback_days  int  optional  days of history, default 40, max 90
returns formatted text: RSI, MACD, Bollinger, BB position, avg volume, last close
```

---

# Single-shot LLM agents

These call `LLM.complete()` once with a structured prompt and parse the JSON response.

## Debater → `debate_bull` / `debate_bear`  (ONE class, two roles)

- **Responsibility:** Argue the strongest **upside** (bull) or **downside** (bear) case, rebutting
  the other side's most recent argument. Implemented as **one `DebaterAgent` class** parameterised
  by `stance` (`"bull"` or `"bear"`); the two nodes share the same code.
- **I/O:** `DebaterInput` → `DebaterCase | DebaterFailure`
- **What it does:**
  1. `_build_messages()` injects `evidence_summary`, `technical_summary`, and — if the opponent's history is non-empty — their last argument to rebut.
  2. `LLM.complete` (haiku, 768 tokens), single shot.
  3. `parse_json_dict()` → `argument` + `key_points`.
- **Code/deps:** `LLM.complete`, `parse_json_dict`, private `_build_messages` (in `agent.py`).
- **Failure modes:** `DebaterFailure` — `llm_error: …` or `non-object JSON`.

## research_manager → `research_manager_node`

- **Responsibility:** **SOLE decision agent.** Adjudicate the bull/bear debate into one actionable
  direction + conviction. No other component may override or re-derive direction.
- **I/O:** `ResearchManagerInput` → `ResearchPlan | ResearchManagerFailure`
- **What it does:**
  1. `_format_debate()` interleaves bull/bear history into a round-by-round transcript.
  2. `LLM.complete` (**sonnet**, 512 tokens).
  3. `_parse()` clamps `conviction` to [0,1] and maps `recommendation` to the `Recommendation` enum (defaults `HOLD` on invalid).
  4. Returns `ResearchPlan(recommendation, conviction, bull_summary, bear_summary, rationale)`.
- **Code/deps:** `LLM.complete`, `Recommendation` enum (domain), `parse_json_dict`, private `_format_debate`/`_build_messages`/`_parse`.
- **Failure modes:** `ResearchManagerFailure` — `llm_error`, `non-object JSON`, or `parse_error: <exc>`.

## reporting → `reporting_node`

- **Responsibility:** Fetch prices + ledger state, assemble a `DailyReport` with real NAV/P&L, and
  dispatch it through ≥2 channels (Excel + Slack). No LLM — purely deterministic.
- **I/O:** `ReportingInput` → `ReportSent | ReportFailure`
- **What it does:**
  1. Fetches current-bar close prices for all held symbols + SPY benchmark from `ports.market_data`.
  2. Calls `ReportingAgent.run()`: reads the portfolio + cycle trades from `LedgerRepository`,
     computes NAV = cash + Σ(qty × price), unrealised P&L, and SPY day-over-day benchmark return.
  3. Dispatches `DailyReport` through `MultiReportSink` → `ExcelReportSink` + `SlackReportSink`.
- **Code/deps:** `LedgerRepository`, `MultiReportSink`, `MarketDataSource` (via `ports.market_data`).
- **Failure modes:** `ReportFailure` — ledger unavailable; sink dispatch error (degraded gracefully — does not overwrite `cycle_outcome`).

## synthesis → `synthesis_node`

- **Responsibility:** Write the cited investment memo for the full decision cycle — prose summary
  integrating evidence, technicals, the debate, and the execution outcome.
- **I/O:** `SynthesisInput` → `SynthesisReport | SynthesisFailure`
- **What it does:**
  1. `_cycle_format` helpers build readable one-liners from each raw signal dict.
  2. `LLM.complete` (**sonnet**), single shot.
  3. `_parse_report()` extracts `title`, `executive_summary`, `rationale`, and `risk_factors`.
- **Code/deps:** `LLM.complete`, `_cycle_format` helpers, `parse_json_dict`.
- **Failure modes:** `SynthesisFailure` — `llm_error`, `non-object JSON`.

## judge → `judge_node`

- **Responsibility:** Independent LLM-as-judge auditor — score the whole cycle for coherence
  (evidence ↔ TA ↔ decision ↔ memo) on a 1–5 scale. The verdict is recorded and feeds the
  eval's process-quality metrics. Must run **after** `synthesis` so it can grade the memo; must
  be a **separate agent** from the one that wrote the memo.
- **I/O:** `JudgeInput` → `Verdict | JudgeFailure`
- **What it does:**
  1. Per-signal `_*_line()` extractors build readable one-liners from each raw dict (None-safe).
  2. `_build_messages()` assembles a system prompt + a 1–5 scoring rubric.
  3. `LLM.complete` (**sonnet**, 512 tokens).
  4. `_parse_verdict()` clamps `coherence_score` to [1,5], validates `alignment ∈ {aligned, partial, misaligned}`, returns `flags` + `reasoning`.
- **Code/deps:** `LLM.complete`, `VerdictAlignment` enum, private `_*_line`/`_build_messages`/`_parse_verdict`.
- **Failure modes:** `JudgeFailure` — `llm_error`, `invalid JSON`, or `non-object JSON`.

---

# Deterministic tools layer

These are pure functions (or thin wrappers around domain/port calls) with no LLM involvement.
The graph wires them as steps; the LLM **cannot** choose to skip them.

## size_position  (`src/firm/tools/size_position.py`)

- **Signature:** `size_position(recommendation, conviction, nav, price, max_trade_notional_pct) -> Decimal`
- **Called by:** the `pm` node (deterministic, no LLM).
- **Logic:** conviction scales the target notional (conviction=1.0 → `max_trade_notional_pct` of NAV);
  floored to whole shares. Direction comes from the Research Manager's `recommendation` —
  `size_position` **never** re-derives direction.
- **Why deterministic:** guarantees no hallucinated quantities; all numbers are arithmetic.

## search_news — inline closure in the `research` agent

See the research agent section above. The closure is defined inside `make_research_node`
and passed as an executor to `LLM.complete_with_tools`.

## price_indicators — inline closure in the `technical` agent

See the technical agent section above. The closure is defined inside `make_technical_node`.

---

# LLM port — two completion modes

```python
# Single-shot: one request, one response
llm.complete(messages, model="haiku", max_tokens=512) -> LLMResponse | LLMError

# Tool loop: LLM calls tools until it produces a final text response
llm.complete_with_tools(
    messages,
    tools=[ToolDef(name, description, input_schema)],
    executors={"tool_name": callable},
    model="haiku", max_tokens=1024, max_rounds=6,
) -> LLMResponse | LLMError
```

`complete_with_tools` in `AnthropicLLM`: call the API with tools → if `stop_reason == "tool_use"`,
execute each tool via its executor and feed `tool_result` blocks back → repeat until `end_turn`
or `max_rounds` → return final text.

- **`CassetteLLM`** records/replays the final response keyed on model + messages + tool names — fully offline for `make eval`/CI.
- **`FakeLLM.complete_with_tools`** calls each executor once with `{"query": "test", "k": 5}` so side effects (e.g. populating the chunk registry) occur before returning the queued response. See Limitations — this uses wrong arg names for `price_indicators` (which takes `lookback_days`, not `k`).

---

# Node registry / DI

`build_graph(checkpointer, ports)` builds nodes from a registry — adding a node is one line.
The node registry in `src/firm/orchestration/graph.py`:

```python
_NODE_FACTORIES = {
    "research":          make_research_node,
    "technical":         make_technical_node,
    "debate_bull":       make_bull_node,      # DebaterAgent(stance="bull")
    "debate_bear":       make_bear_node,      # DebaterAgent(stance="bear")
    "research_manager":  make_research_manager_node,
    "pm":                make_pm_node,        # deterministic sizing via size_position
    "risk":              make_risk_node,      # HITL gate
    "execution":         make_execution_node,
    "reporting":         make_reporting_node,
    "synthesis":         make_synthesis_node,
    "judge":             make_judge_node,
}
```

Every factory is `(ports: NodePorts) -> Callable`. `NodePorts` is the single DI container holding
all external dependencies (LLM, market data, evidence store, ledger, report sink, guardrails,
risk policy, portfolio, calendar). Both `cli.py` and `eval/replay.py` call `build_graph`.

---

# Live path, bot, and HITL channel abstraction

The same `build_graph` runs in three environments (see `ARCHITECTURE.md` → Three environments):
offline historic replay (frozen bars + cassette LLM), offline agent tests (fakes), and **live
production**. In live production:

- **Market data:** `LiveMarketData` (yfinance OHLCV) replaces `FrozenMarketData`.
- **News:** `NewsIngestionAgent` (`agents/news_ingestion/`) is **wired as the live news fetch** —
  it pulls recent Yahoo Finance headlines via `yfinance` and upserts them into the pgvector corpus
  (real `SentenceTransformerEmbedder` embeddings) before the graph runs.
- **LLM:** live Anthropic (`AnthropicLLM`), graceful + `TokenBudgetLLM`-wrapped.
- **Persistence:** Postgres ledger (stable `FIRM_PORTFOLIO_ID`) + `PostgresSaver` checkpoints.

**HITL channel is a pluggable skill.** The risk node only raises `interrupt()`; the channel that
shows the approval card and resumes the graph is swappable behind the shared
`firm.orchestration.hitl.resume_decision(graph, thread_id, decision)` core:

- **`firm run --hitl console`** — interactive stdin prompt (`a`/`b`/`s`/`h`).
- **`firm run --hitl telegram`** — single Telegram approval card (one-shot).
- **`firm bot`** (`make bot`) — persistent Telegram operator service: `/run <ticker>` → approval
  card (💡 Why / 👍 Pros / 👎 Cons) → Approve, or Reject → Buy / Sell / Hold override → resume →
  report back. See `docs/telegram_flow.md`.
- Slack / email / SMS are drop-in adapters behind the same core.

---

# Limitations worth knowing

1. **`FakeLLM.complete_with_tools` uses hardcoded arg names.** Calls executors with
   `{"query":"test","k":5}` — the wrong key for `price_indicators` (which takes `lookback_days`).
   Also wraps executors in `except Exception: pass`, masking real bugs. (Offline test path only.)
2. **Observability spans are thin.** Tracing setup exists; trace-replay leans primarily on the
   Postgres audit log (`audit_log` + `decision_cycles`), which is the authoritative replay source.
