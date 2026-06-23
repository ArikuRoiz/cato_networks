> **⚠️ SUPERSEDED — historical.** Current gaps live in `docs/REPO_AUDIT.md`; target design in `docs/PROJECT_UNDERSTANDING.md`.

# The AI Investment Firm — Gaps & Readiness

> The three planning docs (`PROJECT_BRIEF`, `ARCHITECTURE`, `SPEC`) are aligned and submission-grade **as plans**. This is the punch-list of everything standing between a ready plan and a shippable submission. Four gap classes, ordered by what unblocks the most.

---

## Gap classes at a glance

| Class | What it is | Closed by |
|---|---|---|
| **A. Blocking decisions** | Choices only you can make and defend | You, now — minutes |
| **B. Open questions** | Design-affecting unknowns | A decision + light research |
| **C. Unvalidated assumptions** | The design is a hypothesis until code runs | A spike / running tests |
| **D. Implementation gap** | The graded deliverables don't exist yet | Building |

**D dominates.** The job grades a running firm, a trace, and an eval report — none exist. No amount of A–C work substitutes for it.

---

## A. Blocking decisions (need your call, not code)

These don't change the architecture's shape, but the eval and the HITL demo cannot run without them.

| Decision | Why it blocks | Defensible default (override freely) |
|---|---|---|
| **RiskPolicy values** — per-trade notional %, single-name concentration cap, daily-loss kill-switch | Risk node + ledger guardrail have nothing to enforce; HITL threshold is undefined | Per-trade ≤ 10% NAV; single-name ≤ 25% NAV; daily-loss halt at −3% NAV; HITL above 5% NAV notional |
| **Replay window** | Eval has no window; CI has nothing to replay | One recent earnings week for one watchlist name (exercises the event path) |
| **Watchlist (5–10 tickers)** | RAG corpus + market data have no universe | One sector, ~6 liquid large-caps + SPY as benchmark |

> Owning these yourself matters: a reviewer expects *you* to defend the risk appetite, not inherit it from a default.

---

## B. Open questions (design-affecting; resolve before the full build)

Not needed for the first spike, but needed before the system is complete.

- **"Qualifying" news event** — relevance-score threshold + per-symbol rate limit that defines when an event fires a cycle. *(FR-2 behavior is currently undefined.)*
- **Slippage / commission model** — fixed bps + per-share, or depth-aware? Exact parameters.
- **News corpus source** — which provider/format to snapshot license-cleanly into `data/`.

---

## C. Unvalidated assumptions (the design is a hypothesis until it runs)

Each is an assumption the architecture leans on that no code has tested. Listed by risk. The mandatory test that proves each is named.

| Assumption | Risk if wrong | Proven by |
|---|---|---|
| **Durable HITL resume across a process restart** (LangGraph Postgres checkpointer interrupts, survives `docker kill`, resumes hours later) | **Highest.** It's the newest mechanism; if it behaves differently, the orchestration design changes | `test_hitl_resumes_after_restart` + a vertical spike |
| **Crash mid-trade leaves no partial state** (single ACID txn + idempotency) | Money integrity — a partial write is a silent corruption | `test_crash_mid_trade_reconciles`, `test_idempotent_execution` |
| **Recorded-LLM cassette makes evals reproducible** | The eval's reproducibility claim is false; CI flakes | `make eval` run twice → identical |
| **Partial-failure halt** (agent error → checkpoint + halt, no trade leaks) | Fail-open in a money system | A fault-injection test on each node |
| **Token/cost estimates** (~1–2M tokens/day, 50K/cycle cap) | Cost or rate-limit surprise | Measured during the spike, not assumed |

**The cheap de-risk:** a ~50-line vertical spike of the riskiest path — one hardcoded trade that interrupts at the Risk gate, persists, survives a kill, and resumes to a committed ledger write — validates the spine before the rest is built on top of it.

---

## D. Implementation gap (what the job actually grades)

Zero lines written. The brief's graded deliverables are all absent:

- [ ] Runnable multi-agent firm (clone → demo < 10 min)
- [ ] The five agents with typed Pydantic contracts + defined failure modes
- [ ] Persistent transactional ledger (survives restart, reconciles)
- [ ] Durable HITL with Slack approval
- [ ] RAG layer with citations + refusal path
- [ ] Observability: replayable trace per trade
- [ ] Eval harness: return vs SPY + process metrics, in CI
- [ ] The ten mandatory tests (currently not even failing stubs)
- [ ] Dockerfile + docker-compose + Terraform deployment view
- [ ] Docs: README, technical overview, operational runbook, eval report
- [ ] One full historical day replayed and committed with reports + traces

---

## Priority order

1. **Make the three A-decisions** (minutes).
2. **Scaffold the repo** — tree + README per package, `pyproject.toml`, `docker-compose.yml`, the ten mandatory tests as *failing stubs* → a red suite that encodes "done."
3. **Spike the riskiest path** — durable HITL resume — and measure tokens while doing it (closes the top of class C).
4. **Build to green**, vertical slice by slice: ledger → graph skeleton → tracing → agents → RAG → eval harness.
5. **Resolve class B** as the relevant slice is built.
6. **Capture the sample run + docs** last.

---

## Definition of "ready" (the bar to clear)

| Sense of "ready" | Now |
|---|---|
| Planning docs consistent & defensible | ✅ |
| All design decisions made (class A) | ❌ — three open |
| Architecture proven against reality (class C) | ❌ — nothing has run |
| Graded submission exists (class D) | ❌ — no code |

**Submission-ready when:** the ten tests pass, one historical day is replayed and committed with reports + trace, the eval report shows return *and* process metrics honestly, and clone-to-demo runs in under 10 minutes.
