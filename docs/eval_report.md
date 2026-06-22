# Eval Report — The AI Investment Firm

> Replay window: Oct 21-25 2024 (NVDA earnings week)
> Watchlist: AAPL, MSFT, NVDA, GOOGL, META, AMD | Benchmark: SPY

---

## How to generate results

```bash
make eval
```

This runs the full historical replay against frozen bar CSVs and the synthetic news corpus,
using recorded LLM responses (cassettes) so results are bit-reproducible and require no
live API calls. The report is printed to stdout and saved to `eval/output/eval_report.md`.

---

## Methodology

### Replay harness

The eval harness (`eval/replay.py`) wires the five-agent pipeline with:

- `FrozenMarketData` — reads OHLCV CSVs from `data/bars/` (committed, immutable)
- `FakeEvidenceStore` — populates from `data/news/corpus.json` at startup
- `CassetteLLM` (or `FakeLLM` when no cassette exists) — replays recorded responses
- `_FakeLedger` — in-memory ledger, no Postgres required for eval
- `MemorySaver` — in-memory LangGraph checkpointer, no Postgres required for eval

Each `(symbol, day)` pair is an isolated decision cycle. Weekends and holidays (days with
no bar data) are skipped. The portfolio accumulates across cycles within the run, mirroring
what a live run would see.

### No-lookahead enforcement

The `FakeEvidenceStore` enforces `published_at <= decision_ts` at retrieval time. No
future-dated news article can influence a past decision cycle. The unit test
`test_no_lookahead` asserts this invariant.

### HITL handling in eval

Large trades that would trigger HITL approval are **auto-approved** in the eval harness
(the `hitl_auto_approved` flag is set in the cycle record). This is by design: the eval
measures strategy performance under the assumption that all proposed trades are approved.
In a live run, human rejections would reduce trade frequency.

### Metrics reported

| Metric | Description |
|---|---|
| Total return | `(final_nav - initial_nav) / initial_nav` as % |
| SPY return | SPY benchmark return over the same window |
| Alpha | Total return minus SPY return (not risk-adjusted; n=5 days is too short) |
| Sharpe ratio | Daily returns / std dev × sqrt(252), annualised; reported but not meaningful over 5 days |
| Groundedness % | `(cited claims / total claims) × 100` across all Evidence results |
| Refusal rate | `(Refusal cycles / total cycles) × 100` |
| Guardrail triggers | Count of cycles where Risk rejected or Execution failed |
| HITL invocations | Count of cycles that required human approval (auto-approved in eval) |
| Tokens used | Sum of estimated token counts across all agent calls |
| Estimated cost (USD) | At blended Haiku/Sonnet pricing (~$0.50/1M tokens) |

---

## Honest statement on results

**Run `make eval` to generate current results.** The numbers below are placeholders;
actual metrics depend on the cassette content and the signal thresholds in
`config/risk_policy.yaml`.

```
[Placeholder — run make eval to populate]

Total return     : __.__%
SPY return       : __.__%
Alpha            : __.__%
Sharpe (annual.) : _.__
Cycles run       : ___
Refusal rate     : __.__%
Groundedness     : __.__%
HITL invocations : _
Guardrail hits   : _
Tokens used      : ______
Estimated cost   : $_.__
```

**Results may show underperformance vs SPY — reported without filtering.**

This is expected and unsurprising:
1. The strategy is momentum + news-sentiment with deliberately simple thresholds.
   It is designed to be explainable, not to generate alpha.
2. Five trading days is not a statistically meaningful evaluation period for any strategy.
3. The synthetic news corpus may not produce signal patterns that align with actual
   price movements over the window.
4. Transaction costs (5 bps slippage + $0.005/share commission) reduce returns on
   every round trip.

The eval's purpose is to verify **process quality** — groundedness, guardrail behavior,
HITL mechanics, and reproducibility — not to demonstrate a profitable strategy.
Underperformance vs SPY is reported plainly, not filtered or explained away.

---

## Interpreting process metrics

**Groundedness % < 100%** — some claims were returned by the LLM without a traceable
source URL. This can happen when the corpus has thin coverage for a symbol or when
the extraction model fails to map a claim to a retrieved chunk.

**Refusal rate > 0%** — the Research agent refused to generate evidence because the
corpus returned fewer than the minimum required chunks. This is correct behavior; the
agent must refuse rather than fabricate. Days with thin corpus coverage will have
higher refusal rates.

**Guardrail triggers** — the Risk agent blocked a trade that would have breached a
RiskPolicy limit. This is the defense-in-depth mechanism working as designed.
High guardrail trigger counts indicate the PM agent is regularly proposing oversized
trades; review the buy_threshold and sizing config.

**HITL invocations (auto-approved in eval)** — in a live run, these would pause
the cycle for human review. High HITL counts with the current 5% NAV threshold
indicate a strategy that frequently proposes large positions.
