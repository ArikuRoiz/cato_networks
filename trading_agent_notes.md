# TradingAgents vs. Our Firm — Architecture & Context Notes

Comparison of our system (`cat_networks` / the AI investment firm) against
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents),
focused on **agent architecture, context engineering, and system prompts**.

> **TL;DR.** Both are LangGraph multi-agent trading firms with the same skeleton
> (analysts → bull/bear debate → manager → trade → risk → execution). They optimize
> for opposite things:
> - **TradingAgents** = a *research lab* — more analysts, two debates, full-text
>   context, and a learn-from-P&L reflection loop. Optimized for reasoning depth.
> - **Our firm** = a *production desk* — fewer agents, deterministic gates, typed
>   and bounded context, full audit trail, bit-reproducible evals. Optimized for control.
>
> The one idea worth importing is their **outcome-reflection loop**. The one thing to
> keep resisting is their **LLM-in-the-execution-path** design. The one place we
> *over*-optimized is the **debate context** (too curated) and the **decision-agent
> prompts** (JSON-only, little room to reason).

---

## 1. Roster comparison

| Stage | TradingAgents | Our firm |
|---|---|---|
| **Analysts** | 4 LLM analysts: Fundamentals, Sentiment, News, Technical | 2: Research (news RAG, tool-using) + Technical (indicators, tool-using) |
| **Researcher debate** | Bull vs Bear, multi-round, + Research Manager judge | Bull vs Bear (1 class, 2 stances), `MAX_DEBATE_ROUNDS=1`, + Research Manager (sole decider) |
| **Trader** | LLM Trader composes a trade plan | ❌ Dissolved — Research Manager emits recommendation + conviction directly |
| **Sizing** | Implicit in Trader/PM LLM reasoning | ✅ `size_position()` — **deterministic** (conviction × NAV → whole shares) |
| **Risk** | A *second debate*: Aggressive / Conservative / Neutral analysts, then Risk Manager | `check_risk()` + risk guardrail — **deterministic policy**, LLM-unreachable, HITL interrupt > 5% NAV |
| **Execution** | Order to simulated exchange | ACID ledger write, idempotency key |
| **Post-hoc** | Reflection on realized returns → memory | Judge (coherence audit) + Synthesis memo + recorded HITL |

**Net:** They have more LLM nodes (incl. Fundamentals + Sentiment we lack) and *two*
LLM debates. We collapsed everything downstream of the decision into deterministic code.

---

## 2. Architecture philosophy

**TradingAgents** runs on two LLM debates — one for direction (bull/bear) and a second
for risk (aggressive/conservative/neutral). Nearly every node is an LLM. Its standout
feature is a **financial reflection loop**: after trades resolve it computes realized
returns and injects "recent same-ticker decisions + cross-ticker lessons" back into the
manager's prompt. It genuinely learns from P&L.

**Our firm** keeps the bull/bear debate but replaces everything after the decision with
deterministic code. Trader and Portfolio Manager are *gone* — folded into
`size_position()` ("sizing has a correct answer"). Risk is a hard policy gate the LLM
cannot reach. Our "reflection" is a **Judge that scores coherence**, not a loop that
learns from returns.

---

## 3. Context engineering — system context vs. agent context

### Two opposite context models

- **TradingAgents = accumulating free-text blackboard.** `AgentState` extends LangChain's
  `MessagesState`. Each agent writes a **full prose report** into a named slot
  (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`). Downstream
  agents read those reports **verbatim**, plus the **entire debate transcript** as one
  `history` string, plus retrieved memories (`past_context`). Context grows monotonically —
  by the time the risk judge runs, its window holds 4 full reports + full bull/bear
  transcript + trader plan + full 3-way risk-debate transcript + retrieved past situations.

- **Our firm = typed, bounded, serialized envelope.** `GraphState` is a flat `TypedDict`,
  every value JSON-serializable (it has to round-trip through a Postgres checkpoint and a
  cassette). Agents read **structured summaries, not transcripts**: debaters get
  `evidence_summary` + `technical_summary` + the opponent's *last* argument, not the whole
  history. Outputs are frozen Pydantic (`Evidence` → `Claim(text ≤ 120 chars, chunk_id,
  source_url)`). Context is **curated per node**, not a growing conversation.

### Side by side

| Dimension | TradingAgents | Our firm |
|---|---|---|
| **Container** | `MessagesState` subclass — messages accumulate | Flat `TypedDict`, explicit keys |
| **Payload type** | Free-text reports + transcripts | Validated Pydantic schemas |
| **What a debater sees** | All 4 full reports + full debate `history` + opponent's `current_response` | `evidence_summary` + `technical_summary` + opponent's last turn |
| **Provenance** | Inline prose, no structured citation | `chunk_id` + `source_url` per claim |
| **Cross-cycle memory** | Injected (`past_context`, situation-keyed retrieval) | None — single-cycle by design |
| **Growth pattern** | Monotonic; biggest at the final node | Bounded; each node reads only its keys |
| **Inbound-context safety** | Raw news/Reddit/StockTwits text → prompt | InjectionGuard scans before LLM sees it; token-budget breaker caps total |
| **Why shaped this way** | Max signal for reasoning | Replayable, auditable, injection-resistant |

### What the trade-off costs each side

- **Their richness is real.** Their bull researcher reasons against the *specific
  sentences* of the bear's argument and cites the *full* fundamentals report. For a debate —
  whose whole point is rebutting specific claims — their context model is genuinely better
  suited. Our 120-char claims + pre-summarized evidence may be **starving the debate** of
  the nuance that makes it worth running.
- **Our discipline is also real.** Their richness costs them: (a) **context bloat** at the
  final node — the risk judge is a textbook "lost-in-the-middle" victim; (b) **no
  provenance** — a claim can't be traced to a chunk; (c) an **injection surface** — raw
  social text flows straight into prompts; (d) **non-replayability** — `MessagesState` +
  retrieved memory + temperature make bit-reproduction hard. We designed all four away.

---

## 4. System prompts — ours vs. theirs

This is where the two styles diverge most concretely.

### Ours: static, terse, schema-enforced

All our system prompts are **hardcoded constants** (no f-strings/templating). Context
goes in the *user* message, not the system prompt. Two sub-styles:

- **Terse-structured** (Research, Technical) — schema embedded *in* the system prompt.
- **Terse-persona** (Debater, Research Manager, Synthesis, Judge) — one-line persona,
  schema deferred to the user message.

Examples (verbatim):

> **Research:** "You are a financial research assistant. Use the search_news tool to find
> relevant evidence — call it multiple times with different queries to cover earnings,
> guidance, analyst outlook, and risks. When you have enough evidence, respond ONLY with a
> JSON array of claim objects … Do NOT emit prices, quantities, P&L figures, or dates. Use
> ONLY information present in the retrieved excerpts."

> **Research Manager:** "You are a research manager at a quantitative trading firm. You have
> just observed a structured debate between a bull analyst and a bear analyst. Your job is to
> objectively weigh their arguments against the underlying evidence and produce a clear,
> actionable recommendation. Be decisive — 'hold' is valid but requires justification.
> Respond ONLY with valid JSON, no markdown fences."

> **Judge:** "You are an independent risk auditor at a quantitative trading firm. Your sole
> job is to find logical gaps and inconsistencies in trading decisions. Be specific and
> critical — generic observations are useless. Respond ONLY with valid JSON…"

Characteristics: ~1–4 sentences, role framing is a single clause, behavioral constraints
are explicit ("Do NOT emit prices…", "Be decisive"), and **every prompt forces JSON-only
output with no markdown fences**.

### Theirs: prose, persona-rich, soft format

TradingAgents prompts are **longer natural-language briefs** (~200 words for the Research
Manager), heavy on team/role narrative, with the output taxonomy described *in prose* and
structured output bound *softly* (freetext fallback allowed). State and **past memories**
are interpolated directly into the prompt.

> **Trader:** "You are a trading agent analyzing market data to make investment decisions."
> … "provide a specific recommendation to buy, sell, or hold" … "Anchor your reasoning in
> the analysts' reports and the research plan." (Framed as "a comprehensive analysis by a
> team of analysts" → "Leverage these insights to make an informed and strategic decision.")

> **Research Manager:** positioned as "Research Manager and debate facilitator"; instructed
> to "critically evaluate this round of debate and deliver a clear, actionable investment
> plan", to commit to a stance, reserving Hold for genuinely balanced cases. Specifies an
> explicit rating taxonomy — **Buy / Overweight / Hold / Underweight / Sell** — each with
> conviction + positioning semantics, and "use exactly one". Past debate context is injected
> via a "Debate History" section.

### System-prompt comparison table

| Aspect | Ours | TradingAgents |
|---|---|---|
| **Construction** | Static constant strings | Prose templates with interpolated state + memory |
| **Length** | 1–4 sentences (terse) | ~150–200 words (verbose) |
| **Persona** | One-line role clause | Multi-sentence team/role narrative |
| **Output format** | Hard JSON schema; "respond ONLY with JSON" | Prose-described taxonomy; soft schema, freetext fallback |
| **Reasoning room** | Minimal — JSON-only suppresses CoT | Encourages explicit reasoning before the call |
| **Rating vocabulary** | Enum `Recommendation` (strong_buy…strong_sell) | Buy/Overweight/Hold/Underweight/Sell **with positioning semantics** |
| **Memory in prompt** | None | Retrieved past decisions/lessons injected |
| **Failure mode** | Format-rigid, replay-safe, may under-elicit reasoning | Rich reasoning, but format drift + non-replayable |

### Takeaway on prompts

Our JSON-only discipline is right for parsing, cost, and replay — **keep it**. But for the
two *decision* agents (Research Manager, Judge), forcing JSON-only with a one-line persona
may **under-elicit reasoning quality**. Two cheap, replay-safe fixes:

1. **Reason-then-emit.** Put a `reasoning`/`rationale` field *first* in the schema so the
   model thinks before committing the verdict. (`ResearchPlan` already has `rationale` —
   make sure it's ordered first; replicate for the Judge.)
2. **Borrow their rating semantics.** Their Buy/Overweight/Hold/Underweight/Sell scale
   carries *positioning* meaning, not just direction. Consider enriching our
   `Recommendation` enum docs / manager prompt with sizing-tilt semantics, plus their
   "commit to a stance; reserve Hold for genuinely balanced cases" decisiveness nudge.

---

## 5. What each does better

### TradingAgents' edges
- **Richer signal coverage** — 4 analysts incl. Fundamentals + Sentiment; we have neither.
- **Reflection loop (the real prize)** — closes the loop on *outcomes* (realized returns →
  lessons → future prompts). We record HITL + judge verdicts but don't feed P&L back.
- **Risk-as-debate** surfaces a spectrum of risk appetites for *judgment* calls.
- **Provider-agnostic** (11 LLM backends; deep/quick model split).
- **Debate-appropriate context** — full reports + full transcript suit rebuttal.

### Our firm's edges
- **Replayability is architecturally enforced** — CassetteLLM + frozen Parquet +
  deterministic pipeline = bit-reproducible evals.
- **Safety is structural, not prompted** — risk is a code gate the LLM can't bypass;
  execution is one idempotent ACID transaction; InjectionGuard scans news pre-LLM.
- **Determinism where it belongs** — `size_position()` removes LLM nondeterminism from
  arithmetic.
- **Auditability** — full chain recorded: evidence → debate → gate → approval → fill →
  memo → verdict.
- **HITL** — real human interrupt + recorded override loop.
- **Cost discipline** — haiku for extraction, sonnet for decisions, hard token budget breaker.

---

## 6. What to add (prioritized)

1. **Returns-driven reflection loop — highest value.** On trade resolution, summarize the
   outcome and inject "recent lessons for this ticker" into the Research Manager context.
   Keep replayability by storing lessons as **frozen, versioned artifacts** (same discipline
   as cassettes), not a live mutable vector DB.
2. **Widen the debate context specifically.** Let `DebaterInput` carry richer evidence —
   full `Claim` text (drop the 120-char cap *on the debate path*) and the full
   `bull_history`/`bear_history`, not just the last turn. Keep it structured and cited; just
   stop pre-summarizing the one stage that needs detail.
3. **Reason-then-emit + richer rating semantics** for Research Manager and Judge (see §4).
4. **A Fundamentals analyst.** Slots cleanly into the parallel fan-out beside Research and
   Technical. We currently have no valuation/financials signal.
5. **Advisory risk color (not a gate).** For trades *inside* the limits, a lightweight
   aggressive-vs-conservative perspective feeding the Research Manager could improve sizing
   nuance — *without* weakening the deterministic backstop.
6. **Sentiment sources (Reddit/StockTwits)** — only if they fit under no-lookahead +
   InjectionGuard. Lower priority, high noise.

---

## 7. What NOT to copy

- **Don't re-add the LLM Trader / Portfolio Manager.** Dissolving them into
  `size_position()` is the better call — they use an LLM where arithmetic belongs.
- **Don't make risk a debate that can override the gate.** Their "LLM PM approves orders"
  pattern is exactly the safety surface we correctly removed. Advisory only.
- **Don't adopt `MessagesState`-style accumulation.** It causes their final-node context
  bloat and breaks replay. Per-node curated keys are the better backbone.
- **Don't import raw-prose context** along with their nuance. When we widen the debate
  context (#2 above), do it as *more structured context*, not raw prose — keep typed I/O +
  InjectionGuard.
- **Don't adopt mutable in-prompt memory naively** — version the lesson store as frozen data
  so cassettes stay reproducible.

---

## One-line summary

Their **system context is optimized for reasoning depth** (everything, verbatim,
accumulated, persona-rich prompts); our **agent context is optimized for control** (typed,
bounded, cited, replayable, schema-enforced prompts). Import their **reflection loop**,
loosen our **debate context** and **decision-agent prompts**, and keep everything else.
