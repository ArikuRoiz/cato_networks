# Technical Overview — The AI Investment Firm

> Architecture narrative, key design decisions, and documented limitations.
> Companion to `docs/runbook.md` (operations) and `docs/agents_and_tools.md` (agent roster).
>
> **This document describes the TARGET design** agreed in `docs/PROJECT_UNDERSTANDING.md`.
> The build work to reach this target is tracked in `docs/REFACTOR_TICKETS.md`.
> Where the current code differs (e.g. five-node eval graph, `portfolio_manager` still present),
> this document reflects the target.

---

## 1. System purpose

The AI Investment Firm is a multi-agent paper-trading desk that researches, decides, sizes,
and executes trades in US equities, with human approval required for large positions. Every
decision is grounded in cited evidence from a RAG corpus, persisted transactionally, and
replayable end-to-end from the audit log.

**The graded subject is production engineering, not trading alpha.** The strategy is
deliberately simple so every decision is explainable and the eval is reproducible.

---

## 2. Logical view

```
Scheduler (scheduled triggers)         News Event Listener (debounced)
         │                                        │
         └────────────────┬───────────────────────┘
                          ▼
                    [LangGraph Pipeline — durable, checkpointed]
          ┌──────────────────────────────────────────────────────────┐
          │  research + technical (parallel)                         │
          │       → debate (bull ⇄ bear ×N)                         │
          │       → Research Manager   [SOLE decision agent — LLM]  │
          │       → size_position + check_risk  [deterministic]      │
          │       → RISK GUARDRAIL ─┬─(auto)──→ Execution           │
          │                         └─(HITL)──→ Execution           │
          │                                ↓                         │
          │                          Reporting agent                 │
          │                          Judge (independent auditor)     │
          └──────────────────────────────────────────────────────────┘
                          │                    │
                   pgvector RAG            Postgres ledger
                   (news corpus)           (cash / lots / trades / audit)
```

**Agent responsibilities**

| Agent | Input | Output | Notes |
|---|---|---|---|
| Research | `(symbol, decision_ts)` | `Evidence \| Refusal` | RAG agent; grounded cited claims |
| Technical | price bars | `TechnicalSignal \| TechnicalUnavailable` | RSI/MACD/Bollinger |
| Debater (bull ⇄ bear) | evidence + TA + opponent history | `BullCase \| BearCase \| DebaterFailure` | One class, two roles |
| Research Manager | debate transcript | `ResearchPlan \| ResearchManagerFailure` | **Sole direction decider** |
| Reporting | cycle + ledger state | `ReportSent \| ReportFailure` | Memo + Excel + Slack |
| Judge | full cycle + memo | `Verdict \| JudgeFailure` | Independent 1–5 coherence audit |

**Deterministic sizing (not an agent):**

| Step | Input | Output |
|---|---|---|
| `size_position` | `ResearchPlan` + NAV + price + policy | `TradeProposal \| Hold` |
| `check_risk` | trade stub + portfolio + prices | `Approved \| Rejected` |

`Portfolio Manager is NOT an agent.` It dissolved into the deterministic `size_position` and
`check_risk` tools (Ticket R1). This removes the old two-decision-makers conflict where the PM
could re-derive its own signal and override the Research Manager with a `Hold`.

All failure modes are result unions — no exceptions cross agent boundaries. A pipeline
failure checkpoints state and halts the cycle (fail-safe). No trade reaches the ledger
without passing the mandatory risk guardrail step.

---

## 3. Ports and adapters — the IO seam

Four Protocol ports define every external IO boundary:

```
MarketDataSource   — OHLCV bars; live adapter vs frozen CSV adapter (replay/CI)
EvidenceStore      — news chunk retrieval; pgvector adapter vs in-memory fake
LLM                — language model calls; Anthropic adapter vs cassette adapter (CI)
ReportSink         — Excel + Slack output; real adapter vs no-op fake (tests)
```

Agents depend on the Protocol interfaces, never on concrete adapters. Swapping
live-to-replay is a single wiring change at the composition root (`orchestration/nodes.py`).

The **LedgerRepository** is not a port. It is a concrete Postgres repository tested against
an ephemeral Postgres (testcontainers). Abstracting it would buy nothing at this scale.

The **recorded-LLM cassette** on the `LLM` port is what makes the eval reproducible
despite LLM nondeterminism — responses are recorded once and replayed exactly in CI.

---

## 4. Data stores

**One Postgres instance — ledger, checkpoints, and RAG corpus.**

A single ACID boundary means the trade's money-write and its graph checkpoint commit
under the same transactional guarantees. One store to back up, one to restore.

`pgvector` hosts the news-corpus embeddings. At 5K-10K chunks (<100 MB) a dedicated
vector database would add operational cost with no throughput benefit. Retrieval joins
on `published_at` and `symbol` metadata enforce the no-lookahead filter at the data layer.

**Frozen files for market data and news corpus** — committed CSVs and `corpus.json`.
CI cannot call live APIs; reproducibility requires fixed inputs.

**No Redis.** No hot-key, no cache-coherency, no fan-out problem at 10 symbols.

---

## 5. Orchestration — LangGraph pipeline graph

The orchestration pattern is a **pipeline graph with a conditional edge**, not a
supervisor or hierarchical router. Reasons:

- The desk workflow is a pipeline (evidence → decision → risk gate → fill → report).
  The graph mirrors the domain; every path is deterministic.
- Replay is trivial: same inputs, same path.
- A supervisor adds dynamic routing nondeterminism that fights auditability.
- At six agents there is nothing to dynamically route.

**One graph, one source of truth.** Both `cli.py` and `eval/replay.py` must run
`firm.orchestration.graph.build_graph`. The current code diverges (eval builds its own
inline 5-node pipeline); convergence is Ticket R2.

**Interrupt/resume for HITL** — the risk guardrail step calls `interrupt()` when notional
exceeds 5% NAV. LangGraph checkpoints the full graph state to Postgres before pausing.
A human responds via Slack; the graph resumes from the checkpoint, re-validates limits
against the current bar, and proceeds to execution (or rejects if limits are now breached).
Every human approve/edit/reject is recorded as an `ApprovalRow` in the audit log so
override rate and latency are measurable process metrics (Ticket R6).

---

## 6. Ledger invariants

Every ledger write is a single serializable transaction with an idempotency key:

```sql
BEGIN;
  UPDATE portfolios SET cash_balance = cash_balance - :notional - :commission WHERE id = :pid;
  INSERT INTO lots (...) VALUES (...);
  INSERT INTO trades (idempotency_key, ...) VALUES (:key, ...)
    ON CONFLICT (idempotency_key) DO NOTHING;
  INSERT INTO audit_log (...) VALUES (...);
COMMIT;
```

`ON CONFLICT DO NOTHING` on `idempotency_key` makes retried executions no-ops.
Partial writes are impossible: either the entire transaction commits or nothing changes.

The mandatory **risk guardrail** re-validates against `RiskPolicy` immediately before
every `ledger_commit` call — even if `check_risk` already ran after sizing. This
defense-in-depth means the LLM can never bypass risk limits regardless of what path the
graph took to reach execution.

---

## 7. Key design decisions

### Decision 1 — Postgres for everything (ledger + checkpoints + RAG)

**Why:** A single ACID boundary eliminates the distributed-transaction problem between
the ledger write and the graph checkpoint. One store to operate. pgvector at <100 MB
performs identically to a dedicated vector database.

**Trade-off accepted:** pgvector is less feature-rich than Pinecone or Chroma.
Irrelevant at 5K chunks and a write rate of a few trades per day.

---

### Decision 2 — Pipeline graph over supervisor

**Why:** The desk workflow is a strict pipeline; determinism and auditability are
required. A supervisor introduces nondeterministic routing that makes replay harder
and traces harder to read.

**Trade-off accepted:** no dynamic agent selection. Not needed for this problem.

---

### Decision 3 — Recorded-LLM cassette on the LLM port

**Why:** LLMs are nondeterministic. CI must be reproducible and offline. The cassette
adapter records real Anthropic responses once and replays them byte-for-byte in CI.
Eval metrics are then bit-reproducible across runs.

**Trade-off accepted:** cassette maintenance — if prompts change, cassettes must be
re-recorded. The cassette path is `data/cassettes/eval.jsonl`.

---

### Decision 4 — Ledger as concrete repository, not a port

**Why:** The port seam exists where we must swap live for replay (market data, evidence,
LLM) or write to an external sink. The ledger has no replay variant — we test it against
real Postgres (testcontainers). Adding a fake ledger port would let unit tests pass while
hiding real ACID bugs; that is the wrong trade-off for money writes.

**Trade-off accepted:** agents are not unit-testable against a fake ledger. Integration
tests using real Postgres cover the ledger paths.

---

### Decision 5 — Serializable isolation for the ledger

**Why:** Cash/holdings are strongly consistent. Eventual consistency is unacceptable for
money. Serializable isolation eliminates all anomalies (dirty reads, phantoms, write skew)
at the cost of slightly lower throughput. At this cadence (a few trades per day) the cost
is zero.

**Trade-off accepted:** higher lock contention under concurrent writes — irrelevant at
this scale.

---

### Decision 6 — One decision-maker (Research Manager); Portfolio Manager dissolved

**Why:** Having both a Research Manager (LLM direction) and a Portfolio Manager
(deterministic but parallel signal) created a two-decision-makers conflict: the PM could
re-derive its own momentum/sentiment signal and output `Hold` even when the Manager said
`BUY`. The Judge agent existed partly to catch this incoherence. Collapsing to one
decision-maker removes the conflict, eliminates hallucinated-direction risk, and keeps all
quantities LLM-free (guaranteed no hallucinated numbers).

**What replaced it:** `size_position(recommendation, conviction, nav, price, policy) -> qty`
is pure arithmetic. `check_risk` wraps `RiskPolicy.check_trade` as an advisory step. The
mandatory risk guardrail before execution is unchanged — defense-in-depth.

**Trade-off accepted:** momentum is no longer a direction signal. If price momentum should
influence *size*, it is passed as an input to `size_position`, never used to flip direction.

---

## 8. Deployment view

```
docker-compose (local demo)          AWS equivalent (Terraform — not applied)
─────────────────────────            ──────────────────────────────────────
firm-app container                   ECS Fargate task (aws_ecs_service)
postgres + pgvector container        RDS Postgres 16 (aws_db_instance, Multi-AZ)
langfuse container                   ECS Fargate task (self-hosted Langfuse)
secrets via .env                     SSM Parameter Store (aws_ssm_parameter)
```

The Terraform in `infra/` documents the production path but is not applied.
Applying it is a deliberate non-goal for a local demo.

---

## 9. Documented limitations

**Single-node SPOF.** Postgres runs on one node. Crash recovery is supported (restart +
resume from checkpoint); zero-downtime failover is not. The HA path is documented in
`infra/main.tf` (RDS Multi-AZ) but not applied.

**No live data feed (replay/CI).** Market data is frozen OHLCV CSVs. The `MarketDataSource`
port makes adding a live feed adapter a one-file change. In production, `yfinance` is used
via the `fetch_live_news` tool and a live `MarketDataSource` adapter.

**Synthetic news corpus.** `data/news/corpus.json` contains 50-100 synthetic articles,
license-clean for commit. Real news requires a vendor subscription.

**Cassette recordings generated with FakeLLM.** Cassettes were generated with the fake LLM
adapter to keep CI offline; a production eval would record real Anthropic responses. The
current cassette means eval metrics reflect fake responses, not real model quality.

**Eval graph diverged (Ticket R2).** `eval/replay.py` builds its own inline 5-node pipeline
rather than calling `firm.orchestration.graph.build_graph`. Until R2 is merged, the eval
does not exercise the real pipeline topology described here.

**Reporting emits hard-coded zeros (Ticket R7).** Current code hard-codes `pnl = Decimal("0")`
and computes NAV cash-only without prices. Fix tracked in R7.

**RAG embeddings are random (Ticket R7).** `_embed()` ignores the API key and returns a
random vector; retrieval is keyword-only. Fix tracked in R7.

**Observability spans are no-ops (Ticket R5).** Span decorators exist but emit nothing.
Trace-replay currently relies solely on the audit log.

**Long-only, v1.** No shorting or margin mechanics.

**Deferred by design:**
- `Trade`-as-Command / event-sourcing formalism
- `ReportSink` as a pluggable-strategy registry
- Four-way ISP port splitting
