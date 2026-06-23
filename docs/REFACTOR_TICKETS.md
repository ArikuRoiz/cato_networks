# Refactor Tickets

Build phase to converge the firm onto the target in `PROJECT_UNDERSTANDING.md`.
Each ticket is scoped to be dispatchable on its own (e.g. to a Sonnet build agent).

## Ticket map
- **R1 — Single decision-maker: dissolve Portfolio Manager into tools** ← *this ticket, start here*
- R2 — Converge to ONE graph (eval/CLI run `firm.orchestration.graph`; delete `_build_eval_graph`)
- R3 — Stand up the `tools/` layer (search_news, fetch_live_news, price_indicators, size_position, check_risk, make_report, ledger_commit)
- R4 — Merge duplicates (bull/bear → `DebaterAgent`; `SynthesisInput`≈`JudgeInput`; triplicated helpers; `HITLStatus`≈`ApprovalStatus`)
- R5 — Wire built-but-unwired deliverables (Excel + Slack sinks, NYSECalendar gating, TokenBudget breaker, OutputSchemaValidator)
- R6 — HITL recording feedback loop (`ApprovalRow`: persist every approve/edit/reject)
- R7 — Fix correctness gaps (reporting NAV/P&L real numbers; real embeddings; eval shows a real trade)
- (pending) Judge: keep standalone vs fold into Reporting agent

---

## R1 — Single decision-maker: dissolve Portfolio Manager into tools

**Status:** Ready · **Depends on:** none (safe to start first; doesn't need the graph convergence)

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
- **Edit:** `src/firm/orchestration/nodes.py` (`make_pm_node` → deterministic sizing step)
- **Remove/reduce:** `src/firm/agents/portfolio_manager/` (dissolve the agent; keep its schemas if
  `TradeProposal`/`Hold` are reused)
- **Tests:** new `tests/unit/test_tools.py` (size_position + check_risk); update `tests/unit/test_agents.py`

### Out of scope (other tickets)
- Graph convergence → R2 · tools layer for retrieval/reporting → R3 · merges → R4 ·
  wiring deliverables → R5 · HITL recording → R6 · correctness gaps → R7 · Judge decision → pending.
