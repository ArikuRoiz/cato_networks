# Refactor Tickets

Build phase to converge the firm onto the target in `PROJECT_UNDERSTANDING.md`.
Each ticket is scoped to be dispatchable on its own (e.g. to a Sonnet build agent).

> **Status: all DONE.** R1–R8 and the follow-on HITL/bot work have shipped. The firm runs on a
> single converged graph, with real embeddings, real NAV/P&L, wired Excel+Slack reporting, and a
> live production path (yfinance + Anthropic + Postgres). HITL is now approve-every-cycle with
> override, and the persistent `firm bot` Telegram operator service ships on top of the shared
> `resume_decision` core. The ticket bodies below are kept as a historical record.

## Ticket map
- ✅ **R1 — Single decision-maker: dissolve Portfolio Manager into tools** — DONE
- ✅ R2 — Converge to ONE graph (eval/CLI run `firm.orchestration.graph`; `_build_eval_graph` gone) — DONE
- ✅ R3 — Stand up the `tools/` layer (size_position, check_risk; search_news / price_indicators as inline agent closures; live news via `NewsIngestionAgent`) — DONE
- ✅ R4 — Merge duplicates (bull/bear → `DebaterAgent`; shared cycle-summary helpers; unified HITL status enum) — DONE
- ✅ R5 — Wire built-but-unwired deliverables (Excel + Slack sinks, NYSECalendar gating, TokenBudget breaker, OutputSchemaValidator) — DONE
- ✅ R6 — HITL recording feedback loop (`ApprovalRow`: persist every approve/override) — DONE
- ✅ R7 — Fix correctness gaps (reporting NAV/P&L real numbers; real `SentenceTransformerEmbedder` embeddings; eval shows a real trade) — DONE
- ✅ R8 — Live production path + HITL channel skill (`firm run` live, `firm bot` Telegram, `resume_decision`, always+override HITL) — DONE
- ✅ Judge: kept standalone (independent final auditor) — DONE

---

## R1 — Single decision-maker: dissolve Portfolio Manager into tools

**Status:** ✅ DONE · **Depends on:** none (safe to start first; doesn't need the graph convergence)

### Why
Today **two** components decide trade direction with different logic:
- **Research Manager** (LLM) adjudicates the bull/bear debate → `recommendation` + `conviction`.
- **Portfolio Manager** (deterministic) re-derives its *own* momentum/sentiment signal, runs its
  *own* threshold check, and can output `Hold` even when the Manager said buy.

They can disagree — the **Judge agent exists partly to catch this incoherence**. Collapsing to one
decision-maker removes the conflict and the double-counting, and keeps all numbers LLM-free.

### Change
1. **Research Manager = the sole decision agent.** It owns direction (strong_buy…strong_sell) +
   conviction (0–1). No other component decides direction.
2. **Dissolve `PortfolioManagerAgent`.** Move its sizing math into a deterministic tool
   `src/firm/tools/size_position.py`:
   `size_position(recommendation, conviction, nav, price, policy) -> qty`
   — conviction scales the target notional; capped by RiskPolicy (per-trade ≤ 10% NAV);
   floored to whole shares; qty rounding to 0 → `Hold`.
3. **Expose risk as an advisory tool** `src/firm/tools/check_risk.py` wrapping
   `RiskPolicy.check_trade` (used pre-sizing so the Manager/sizing step can self-validate).
   ⚠️ The **mandatory risk GATE at execution stays unchanged** — defense-in-depth; the LLM can
   never skip it.
4. **Replace `make_pm_node`** with a deterministic sizing step that consumes the Manager's
   `ResearchPlan` + current price + NAV → `TradeProposal | Hold`. No LLM, no parallel signal.
5. **Subsume the old momentum/sentiment scoring.** Fundamental view comes from the debate;
   technical/price view from the Technical agent. If we want price momentum to influence *size*,
   pass it as an input to `size_position` — never as a second *direction* decision.

### Acceptance criteria
- [ ] No LLM emits a number; every quantity comes from `size_position` (deterministic).
- [ ] Exactly one component decides direction (Research Manager). The sizing step never flips a
      buy/sell into `Hold` for *directional* reasons — only because qty rounds to 0 or RiskPolicy caps it.
- [ ] `size_position` respects per-trade ≤ 10% NAV; conviction scales size monotonically.
- [ ] Risk guardrail still re-validates at execution (unchanged, still mandatory).
- [ ] `PortfolioManagerAgent` removed (or reduced to the tool); the dead `llm=None` param is gone.
- [ ] New tests: high-conviction buy → non-zero `TradeProposal`, notional ≤ 10% NAV; zero/low
      conviction → `Hold`; sizing honors the cap. Existing suite stays green (currently 230 passed).

### Files
- **New:** `src/firm/tools/__init__.py`, `src/firm/tools/size_position.py`, `src/firm/tools/check_risk.py`
- **Edit (3 PM call sites — PM is imported in all three):** `src/firm/orchestration/nodes.py`
  (`make_pm_node` → sizing step), `eval/replay.py` (its `_pm_node`), `src/firm/cli.py` (PM construction)
- **Remove:** `PortfolioManagerAgent` class + its momentum/sentiment/llm logic. **Keep** the
  `TradeProposal`/`Hold` schemas (used by risk/execution/eval) — relocate if needed without breaking importers.
- **Tests:** new `tests/unit/test_tools.py` (size_position + check_risk); update `tests/unit/test_agents.py`
- **Note:** graph convergence stays in R2 — eval keeps its own graph here, just swaps the PM agent for the tool.

### Out of scope (other tickets)
- Graph convergence → R2 · tools layer for retrieval/reporting → R3 · merges → R4 ·
  wiring deliverables → R5 · HITL recording → R6 · correctness gaps → R7 · Judge decision → pending.
