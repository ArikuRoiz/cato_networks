# Agents and Tools

> **This document describes the TARGET design** agreed in `docs/PROJECT_UNDERSTANDING.md`.
> The build work to reach this target is tracked in `docs/REFACTOR_TICKETS.md`.
> The current code implements an earlier shape (11 nodes including `portfolio_manager`,
> `bull_researcher`, `bear_researcher`, `synthesis`); where the current code differs this
> document reflects the target, not the current state.

---

## At a glance

The target pipeline has **6 real LLM agents** and a **deterministic tools layer**. No agent is
a disguised service. One agent decides direction; everything else is deterministic math.

**Real LLM agents:**

| Agent | Node | Model | Output |
|---|---|---|---|
| Research | `research_node` | haiku | `Evidence \| Refusal` |
| Technical | `technical_node` | haiku | `TechnicalSignal \| TechnicalUnavailable` |
| Debater (bull role) | `debate_bull` | haiku | `BullCase \| BullFailure` |
| Debater (bear role) | `debate_bear` | haiku | `BearCase \| BearFailure` |
| Research Manager | `research_manager_node` | sonnet | `ResearchPlan \| ResearchManagerFailure` |
| Reporting | `reporting_node` | sonnet | `ReportSent \| ReportFailure` |
| Judge | `judge_node` | sonnet | `Verdict \| JudgeFailure` |

**Deterministic tools layer** (no LLM — the LLM cannot skip or reorder these):

| Tool | Replaces | Purpose |
|---|---|---|
| `size_position` | `portfolio_manager` agent | Sizing math: conviction × NAV → qty, capped by RiskPolicy |
| `check_risk` | pre-execution self-check | Wraps `RiskPolicy.check_trade`; advisory before the mandatory gate |
| `search_news` | (was inline in research) | `EvidenceStore.search` + injection scan |
| `fetch_live_news` | `news_ingestion` (orphaned) | yfinance pull → `embed_and_store`; production only |
| `price_indicators` | (was inline in technical) | `MarketDataSource.get_bars` + `compute_indicators` |
| `make_report` | (was inline in reporting) | Assembles `DailyReport`, dispatches Excel + Slack |
| `ledger_commit` | (was inline in execution) | Atomic ACID write (cash + lot + trade + audit) |

**Mandatory deterministic gates** (cross-cutting — the LLM cannot skip them):

| Gate | Position | Action |
|---|---|---|
| Risk guardrail | After `size_position`/`check_risk`, before every ledger write | Re-validates against RiskPolicy; routes > 5% NAV to HITL interrupt |
| Execution | After risk guardrail | Atomic ledger commit (the only thing that moves money) |
| Injection scan | Inside `search_news` | Filters every retrieved chunk before the LLM sees it |
| Token-budget circuit breaker | Cross-cutting | Halts the pipeline if token budget is exhausted |
| Output-schema validation | Cross-cutting | Validates every agent output against its typed schema |

**Pipeline:**

```
research + technical (parallel)
        → debate (bull ⇄ bear ×N rounds, ONE DebaterAgent class)
        → Research Manager (decide direction + conviction)   [SOLE decider — LLM]
        → size_position (deterministic) + check_risk
        → [RISK GUARDRAIL] →(> 5% NAV)→ HITL interrupt → human approve/edit/reject  (RECORDED)
        → Execution (atomic ledger write)
        → Reporting agent (investment memo + Excel/Slack via make_report)
        → Judge (independent coherence audit, recorded)
        → END
```

`research` and `technical` fan out in parallel from START. `debate_bull` waits for both.
The Debater is **one class** instantiated in two roles — no separate `bull_researcher` /
`bear_researcher` classes.

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
*(Previously named `get_price_and_indicators` in the current code; renamed to `price_indicators`
to match the tools-layer naming convention in the target.)*

---

# Single-shot LLM agents

These call `LLM.complete()` once with a structured prompt and parse the JSON response.

## Debater → `debate_bull` / `debate_bear`  (ONE class, two roles)

- **Responsibility:** Argue the strongest **upside** (bull) or **downside** (bear) case, rebutting
  the other side's most recent argument. Implemented as **one `DebaterAgent` class** parameterised
  by `stance` (`"bull"` or `"bear"`); the two nodes share the same code.
- **I/O:** `DebaterInput` → `BullCase | BearCase | DebaterFailure`
- **What it does:**
  1. `_build_messages()` injects `evidence_summary`, `technical_summary`, and — if the opponent's history is non-empty — their last argument to rebut.
  2. `LLM.complete` (haiku, 768 tokens), single shot.
  3. `parse_json_dict()` → `argument` + `key_points`.
- **Code/deps:** `LLM.complete`, `parse_json_dict`, private `_build_messages`.
- **Failure modes:** `DebaterFailure` — `llm_error: …` or `non-object JSON`.
- **Target change from current code:** the two near-identical classes `bull_researcher` /
  `bear_researcher` are merged into one `DebaterAgent` class (Ticket R4).

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

- **Responsibility:** Write the investment memo and dispatch the full cycle report through ≥2
  channels (Excel + Slack) via the `make_report` tool.
- **I/O:** `ReportingInput` → `ReportSent | ReportFailure`
- **What it does:**
  1. Calls the `make_report` tool, which assembles a `DailyReport` (NAV, P&L, positions, trades,
     memo prose) and dispatches it through `ReportSink.send_daily_report`.
  2. The LLM writes the narrative investment memo; all numbers come from the ledger (no hard-coded zeros).
- **Code/deps:** `make_report` tool, `LedgerRepository`, `ReportSink`, `DailyReport`.
- **Failure modes:** `ReportFailure` — ledger unavailable; sink dispatch error.
- **Target change from current code:** the current code hard-codes `pnl = Decimal("0")` and
  passes no prices; that is corrected in Ticket R7. The old `synthesis` node's memo-writing is
  subsumed here (Ticket R4).

## judge → `judge_node`

- **Responsibility:** Independent LLM-as-judge auditor — score the whole cycle for coherence
  (evidence ↔ TA ↔ decision ↔ memo) on a 1–5 scale. The verdict is recorded and feeds the
  eval's process-quality metrics. Must run **after** `reporting` so it can grade the memo; must
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

## size_position  *(target: `src/firm/tools/size_position.py`)*

- **Replaces:** `portfolio_manager` agent (dissolved — Ticket R1).
- **Signature:** `size_position(recommendation, conviction, nav, price, policy) -> TradeProposal | Hold`
- **Logic:** conviction scales the target notional (conviction=1.0 → 10% NAV); capped by
  `RiskPolicy` (per-trade ≤ 10% NAV); floored to whole shares; qty rounds to 0 → `Hold`.
  Direction comes from the Research Manager's `recommendation` — `size_position` **never**
  re-derives direction.
- **Why deterministic:** guarantees no hallucinated quantities; all numbers are arithmetic.

## check_risk  *(target: `src/firm/tools/check_risk.py`)*

- **Replaces:** the advisory part of the old `risk` agent.
- **Signature:** `check_risk(trade, portfolio, prices, policy) -> Approved | Rejected`
- **Logic:** thin wrapper around `RiskPolicy.check_trade`. Used as a self-check step after sizing.
  Does **not** replace the mandatory risk guardrail at execution (defense-in-depth).

## search_news — see research agent above

## fetch_live_news  *(target: `src/firm/tools/fetch_live_news.py`)*

- **Replaces:** the orphaned `news_ingestion` agent (now wired deliberately — Ticket R5).
- **Logic:** `yfinance` pull → filter to NewsDoc objects newer than the lookback cutoff → `EvidenceStore.embed_and_store`. Production only; not run in replay/CI.

## price_indicators — see technical agent above

## make_report  *(target: `src/firm/tools/make_report.py`)*

- **Logic:** assembles `DailyReport` (NAV computed from ledger + live prices, real P&L, positions,
  trades, memo prose) and dispatches through `ReportSink.send_daily_report` (Excel + Slack).

## ledger_commit  *(target: `src/firm/tools/ledger_commit.py`)*

- **Logic:** the atomic ACID write that is currently inline in the `execution` node — extracted to
  a named tool so it can be unit-tested independently of the graph.

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
The **target** registry (post-R1/R4):

```python
_NODE_FACTORIES = {
    "research":          make_research_node,
    "technical":         make_technical_node,
    "debate_bull":       make_debater_node("bull"),
    "debate_bear":       make_debater_node("bear"),
    "research_manager":  make_research_manager_node,
    # size_position + check_risk are deterministic steps, not LLM nodes
    "execution":         make_execution_node,
    "reporting":         make_reporting_node,
    "judge":             make_judge_node,
}
```

Every factory is `(ports: NodePorts) -> Callable`. `NodePorts` is the single DI container holding
all external dependencies (LLM, market data, evidence store, ledger, report sink, guardrails,
risk policy, portfolio).

---

# Limitations worth knowing

1. **`FakeLLM.complete_with_tools` uses hardcoded wrong arg names.** Calls executors with
   `{"query":"test","k":5}` — wrong for `price_indicators` which takes `lookback_days`. Also
   wraps executors in `except Exception: pass`, masking real bugs. Fix tracked in Ticket R7.
2. **Eval does not test the real graph.** `eval/replay.py` builds its own inline 5-node pipeline
   that imports nothing from `firm.orchestration`. Convergence to one graph is Ticket R2.
3. **Reporting emits hard-coded zeros in current code.** `pnl`, `benchmark_return`,
   `current_price`, `unrealized_pnl` are all literal `0`. Fix (real ledger numbers + prices
   passed from graph state) is Ticket R7.
4. **RAG embeddings are random in current code.** `_embed()` ignores the API key and returns a
   random vector; retrieval is keyword-only. Fix is Ticket R7.
5. **Observability spans are no-ops.** Span decorators exist but do nothing; trace-replay relies
   solely on the audit log. Wire tracked in Ticket R5.
6. **`fetch_live_news` (née `news_ingestion`) orphaned in current code.** Nothing imports it.
   Wiring it as the production live-data tool is Ticket R5.
