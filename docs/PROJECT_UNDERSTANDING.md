# Project Understanding — Key Notes

> Shared source of truth. We return here whenever the flow feels off. If code or other
> docs contradict this, that's a flag to resolve — not a silent override.

## 1. What this project is
An **AI Investment Firm**: a multi-agent paper-trading US-equities desk where each desk role is
a specialized agent. It holds positions, makes decisions, reports P&L. Large trades pause for a
**human Risk Committee (HITL)**. Every decision is **grounded in cited evidence**, **persisted
transactionally**, and **replayable from a trace**.

**The goal is NOT beating the market.** It is to prove we can take a messy, stateful business
problem, decompose it into believable agents, ground them in real data, persist state safely,
wrap them in guardrails, observe them, and evaluate them honestly. This is a **senior AI engineer
home task (Cato Networks)** — graded on production engineering and the ability to defend trade-offs.

## 2. What it's graded on
Persistent state surviving a crash · continuous operation during market hours (scheduled **and**
event triggers) · ≥4 specialized agents with typed contracts · RAG with citations and **no
hallucinated numbers/dates/quotes** · HITL for trades over threshold · observability (replay a
trade end-to-end from the trace alone) · reports through **≥2 channels** · reproducible eval
(return **vs SPY** + **process metrics**, reported honestly) · token/cost awareness, scalability,
production readiness.

## 3. Environments — three tiers
1. **Historic-data replay (offline):** frozen bars + synthetic corpus + cassette LLM. Deterministic,
   reproducible, no network/keys. This is what CI and `make eval` run.
2. **Agents on offline input (offline):** the agents run against fakes/cassettes for fast tests.
3. **Production (live):** live market data + live news (`yfinance`) + live Anthropic.
   **Live findings are appended back to the corpus** so each future offline eval is richer.
   → `news_ingestion` is the production ingestion path, not dead weight.

## 4. HITL — human approves every cycle, with override
The shipped policy is **human-approves-every-cycle, with override**
(`RiskPolicyConfig.hitl_mode` defaults to `"always"`; a `"threshold"` mode that only pauses
> 5% NAV still exists). The risk node fires a LangGraph `interrupt()` every cycle and checkpoints
state; the human either:
- **Approves** — execute the desk's recommended action (buy/sell as sized, or hold), or
- **Rejects → picks an alternative action** — Buy, Sell, or Hold, overriding the desk.

`firm.orchestration.hitl.resume_decision(graph, thread_id, decision)` carries the structured
decision back into the risk node. An **override rewrites the cycle's `trade_proposal`** so the
synthesis memo / report / judge describe **what actually executed**. Hard `RiskPolicy` limits still
enforce at the execution guardrail even after approval.

Every approve / override is **recorded** (audit log + `ApprovalRow`), so we can understand *why* a
human overrode the desk, feed those decisions back, and measure HITL latency + override rate as
process metrics. The **approval channel is a pluggable skill** — Telegram is the shipped operator
channel (`firm bot` / `firm run --hitl telegram`); Slack / email / SMS slot in behind the same
`resume_decision` core. See `docs/telegram_flow.md`.

## 5. Locked decisions (do not change without asking)
Per-trade ≤ 10% NAV · single-name ≤ 25% NAV · daily-loss halt −3% · HITL above 5% NAV ·
watchlist AAPL/MSFT/NVDA/GOOGL/META/AMD + SPY benchmark · replay window NVDA earnings week
Oct 21–25 2024 · slippage 5bps + $0.005/share · qualifying news event relevance > 0.7,
max 3/symbol/hour.

## 6. Target architecture (agreed direction)

**Tools / skills layer** (deterministic capabilities, folded into the agent that owns them):
`search_news` · `fetch_live_news` (prod; appends to corpus) · `price_indicators` ·
`compute_signal` · `size_position` · `check_risk` · `make_report` · `ledger_commit`

**Real agents** (LLM judgment):
- **Research** — `search_news`, and in prod `fetch_live_news`. Grounds cited claims.
- **Technical** — `price_indicators`. Produces a bias signal.
- **Debater** — one class, two roles (bull ⇄ bear), adversarial debate.
- **Research Manager** — the **SOLE decision agent**: adjudicates the debate and outputs direction
  (strong_buy…strong_sell) + conviction (0–1). One brain decides *what* to do.

> Portfolio Manager is **NOT** an agent. It dissolves into the deterministic `size_position` +
> `check_risk` tools below: sizing math turns the Manager's recommendation + conviction into an
> exact share quantity, capped by RiskPolicy. This removes the old two-decision-makers conflict
> (where PM could re-derive its own signal and override the Manager with a Hold).
- **Reporting agent** — writes the investment memo and builds the Excel + Slack report via `make_report`.
- **Judge (final)** — independent LLM-as-judge auditor: scores the cycle's coherence 1–5; the
  verdict is recorded and feeds the eval's process-quality metrics.

**Guardrails / mandatory steps** (deterministic, cross-cutting — the LLM **cannot** skip them):
- **Risk guardrail** — a guardrail step that runs in **every** pipeline. Before any ledger write it
  re-validates against RiskPolicy and interrupts for human approval every cycle (`hitl_mode="always"`;
  the `"threshold"` mode routes only > 5% NAV). Runs even if the Manager already self-checked risk —
  defense-in-depth. Hard limits are enforced at execution regardless of the human's decision.
- **Execution** — atomic ledger commit (cash debit + FIFO lot + audit + idempotency key). The only
  thing that moves money.
- Cross-cutting: **injection scan** on retrieved text, **token-budget circuit breaker**,
  **output-schema validation**.

**Pipeline:**
```
research + technical (parallel)
        → debate (bull ⇄ bear ×N)
        → Research Manager (decide direction + conviction)   [SOLE decider — LLM]
        → size_position tool (deterministic sizing) + check_risk
        → [RISK GUARDRAIL] → HITL interrupt (every cycle) → human Approve / Reject→(Buy|Sell|Hold)  (RECORDED)
        → Execution (atomic ledger write; hard RiskPolicy limits still enforced)
        → Reporting agent (memo + Excel/Slack)
        → Judge (independent coherence audit, recorded)
```

## 7. Refactor outcomes (historical — from the pre-refactor audit in `REPO_AUDIT.md`)

The pre-refactor audit (below) framed most "dead code" as graded requirements that were built but
NOT WIRED. The refactor (R1–R8 + the HITL/bot work) converged to one graph and wired the pieces.
All of the items below are now **resolved**:

- **One graph.** Both `cli.py` and `eval/replay.py` now run `firm.orchestration.graph.build_graph`.
- **Eval shows a real trade** — single decision-maker (Research Manager) + deterministic
  `size_position`; no more `llm=None → Hold`.
- **Real RAG embeddings** — `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, 384-dim) in pgvector.
- **Reporting computes real NAV / P&L / benchmark** from the ledger + live prices.
- **Wired deliverables:** Excel + Slack sinks (`MultiReportSink`, ≥2 channels), `NYSECalendar`
  gating, `TokenBudgetLLM` breaker, `OutputSchemaValidator`, `ApprovalRow` HITL recording.
- **Merged duplication:** one `DebaterAgent` (bull/bear stances); shared cycle-summary helpers;
  unified HITL status enum.
- **`news_ingestion` is wired** as the live news fetch (yfinance → pgvector) in production.
- **HITL** is now approve-every-cycle with override + a pluggable approval channel; the persistent
  `firm bot` Telegram operator service ships on top of the same `resume_decision` core.

## 8. Plan
1. ✅ Agree this understanding.
2. **Audit the whole repo** (summarize every file) to find duplication, dead code, bad/stale
   markdowns, and flow misunderstandings — before refactoring.
3. Update `docs/agents_and_tools.md` to the agreed target.
4. Dispatch Sonnet to build: `tools/` layer · agents use tools · Risk as a guardrail step ·
   converge to one graph · wire `eval`/CLI to it · fix eval-zeros · record HITL decisions.
5. Verify: tests green + eval shows ≥1 real trade, a HITL pause, non-zero tokens.

## Open questions
1. ~~Manager merge?~~ **RESOLVED** — no merge. Research Manager is the sole decision agent;
   Portfolio Manager dissolves into the `size_position` + `check_risk` tools. One decision-maker.
2. ~~Judge fold?~~ **RESOLVED** — Judge stays a **standalone independent auditor** (its own final
   node). It grades the whole cycle including the memo, so it must not be the agent that wrote the
   memo. Its 1–5 coherence score is recorded and feeds the eval's process-quality metrics. (Reversible.)
