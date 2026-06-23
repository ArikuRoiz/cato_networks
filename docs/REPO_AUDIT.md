# Repository Audit — Pre-Refactor

> **Historical document — this is the pre-refactor audit; all items below have since been resolved
> by R1–R8 + the HITL/bot work (see `REFACTOR_TICKETS.md`). Kept for context.**

> Full-repo audit (122 Python files, 30 markdowns) done by parallel review before the refactor.
> Goal: find duplication, dead code, stale docs, and flow misunderstandings so we converge to one
> excellent system. Pairs with `PROJECT_UNDERSTANDING.md`.

## Headline insight

This is **not primarily a dead-code problem — it's a "built-but-not-wired" problem.** Most of
what looks dead is actually a **graded requirement** that exists in the codebase but nothing plugs
in. The refactor's job is to **converge to one graph and wire the required pieces**, not just delete.

---

## 🔴 Flow problems (highest priority — these are what a reviewer hits)

| # | Problem | Evidence |
|---|---|---|
| 1 | **Two diverged graphs.** `cli.py` runs the real 11-node graph; `eval/replay.py` builds its **own old 5-node graph** inline (~150 duplicated lines, untyped `StateGraph(dict)`). The eval does not test the real system. | `cli.py:294` → `build_graph`; `eval/replay.py:754` → `_build_eval_graph`; eval imports zero symbols from `firm.orchestration` |
| 2 | **Eval shows zeros.** Eval uses `FakeLLM` returning `"[]"` → 0 claims → 0 grounding; PM runs with `llm=None` → `Hold` → 0 trades. | `eval/replay.py:808` (FakeLLM), `:745` (llm=None), `signals.py:136` |
| 3 | **RAG embeddings are random.** `_embed()` reads `ANTHROPIC_API_KEY`, ignores it, returns a random unit vector. Retrieval ranks on keyword boost only — the semantic/vector layer is decorative. | `adapters/evidence_pgvector.py::_embed` |
| 4 | **Reporting emits fake numbers.** `pnl`, `benchmark_return`, `current_price`, `unrealized_pnl` hard-coded `0`. Even wired, P&L is wrong. | `agents/reporting/agent.py` |
| 5 | **Observability is no-op.** Span decorators do nothing; trace-replay leans entirely on the Postgres audit log. | `observability/tracing.py` |

---

## 🟡 "Dead code" that is actually a required deliverable — WIRE, don't delete

| Built but unwired | Requirement it satisfies |
|---|---|
| `adapters/report/excel.py` + `report/slack.py` (only tested; Slack HITL always returns `EXPIRED`) | "Reports through ≥2 channels" + Slack HITL approval |
| `services/calendar.py::NYSECalendar` (tested, never called at runtime) | Market-hours gating |
| `TokenBudgetCircuitBreaker` + `TokenBudgetExceeded` (never instantiated) | Token/cost awareness guardrail |
| `OutputSchemaValidator` (zero callers) | Output-schema-validator guardrail |
| `persistence/models.py::ApprovalRow` (defined, never used) | HITL decision recording (the feedback loop) |
| `agents/news_ingestion/` (orphaned) | Production live-data ingestion path → becomes a `tools/` capability |

---

## 🟢 Genuinely safe to delete

- `files/` directory — 4 byte-for-byte duplicates of top-level docs + 1 stale 5-node `ARCHITECTURE.md`
- `adapters/market_data_live.py` — every method `NotImplementedError`, zero callers
- `strategy/signals.py::compute_momentum_legacy` — superseded by `strategy/momentum.py::compute_momentum`
- `Chunk.embedding` + `Chunk.is_relevant` — never read *(defer: pgvector adapter writes embedding — verify first)*
- `persistence/models.py::DecisionCycleRow`, `EvidenceRow` — never accessed *(defer: migrations — verify first)*
- 5 stale `xfail` stubs in `tests/integration/test_mandatory.py` — real impls already pass in `test_ledger.py`/`test_guardrails.py`
- `_StubResearchAgent` tests in `tests/unit/test_fakes.py` — real `ResearchAgent` now tested in `test_agents.py`

## 🔵 Duplication to merge

- `bull_researcher` + `bear_researcher` → one `DebaterAgent(stance)` (6 files → 3)
- `SynthesisInput` ≈ `JudgeInput` → one `CycleSnapshot` schema (8 shared fields)
- `_evidence_summary`/`_technical_summary`/… triplicated across `synthesis`, `judge`, `nodes.py` → one helper
- `HITLStatus` ≈ `ApprovalStatus` → one enum (drop dead `PENDING`, `EDITED`/`edited_qty`)
- `eval/replay.py::_str_to_uuid` → use `firm.utils.str_to_uuid`
- `_build_llm` / `_build_demo_llm` → shared selector
- `domain/portfolio.py` fill-cost constants vs `RiskPolicyConfig` fields → single source of truth (config silently ignored today)

---

## 📄 Markdown verdicts

- **Delete:** entire `files/` directory (5 files — 4 exact dupes + 1 stale).
- **Stale, update don't delete** (all say "five agents → PM → Risk → …"): `README.md`, `PROJECT_BRIEF.md`, `GAPS.md`, `TICKETS.md`, `SPEC.md`.
- **Keep (current):** `ARCHITECTURE.md`, `TESTING.md`, all `docs/*`, all 12 per-package READMEs.
- **Note:** `ARCHITECTURE.md` + `docs/agents_and_tools.md` are accurate to current code but **not** the agreed target — update them *after* the refactor.

---

## Test suite notes (baseline: 233 passed, 5 xfailed)

- Remove 5 stale `xfail` stubs in `test_mandatory.py` (real impls exist elsewhere).
- Remove `_StubResearchAgent` smoke tests in `test_fakes.py`.
- Remove the consolidated `test_market_calendar_gating` duplicate in `test_calendar.py`.
- Mandatory-requirement tests ARE covered by real tests (ledger/guardrails/hitl/rag) — the `test_mandatory.py` stubs are just uncleaned scaffolding.
