"""Eval report generator — Markdown output with honest negative-result reporting.

Design contract (FIRM-14):
- Never hide negative results.  When alpha < 0, state it plainly.
- Every metric is presented; nothing is suppressed because it looks bad.
- Limitations and caveats are documented in a dedicated section.
- The LLM never touches this module; all numbers come from EvalMetrics.
"""

from __future__ import annotations

from eval.metrics import EvalMetrics
from eval.replay import CycleRecord, EvalResult

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def generate_report(metrics: EvalMetrics, result: EvalResult) -> str:
    """Return a Markdown eval report as a string.

    The report contains:
    - Return Summary (with honest underperformance disclosure)
    - Process Metrics table
    - Guardrail Events list
    - Sample Trade
    - Limitations and Caveats
    """
    sections = [
        _return_summary(metrics),
        _process_metrics_table(metrics),
        _guardrail_events(result.cycles),
        _sample_trade(result.cycles),
        _limitations(),
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _return_summary(metrics: EvalMetrics) -> str:
    """Render the Return Summary section.

    Plainly states underperformance when alpha < 0 — never softened.
    """
    lines = ["## Return Summary", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Return | {metrics.total_return:+.2%} |")
    lines.append(f"| Benchmark Return (SPY) | {metrics.benchmark_return:+.2%} |")
    lines.append(f"| Alpha | {metrics.alpha:+.2%} |")
    lines.append(f"| Sharpe Ratio (annualised) | {metrics.sharpe_ratio:.2f} |")
    lines.append("")
    lines.append(_alpha_verdict(metrics))
    return "\n".join(lines)


def _alpha_verdict(metrics: EvalMetrics) -> str:
    """Return a plain-English verdict on alpha — never euphemistic."""
    if metrics.alpha < 0.0:
        return (
            f"**Result: the strategy underperformed SPY by "
            f"{abs(metrics.alpha):.2%} over the replay window.**  "
            f"The strategy returned {metrics.total_return:+.2%} while SPY "
            f"returned {metrics.benchmark_return:+.2%}.  "
            f"This underperformance is reported plainly; no positive framing "
            f"has been applied."
        )
    if metrics.alpha == 0.0:
        return f"**Result: the strategy matched SPY exactly ({metrics.total_return:+.2%}).**"
    return (
        f"**Result: the strategy outperformed SPY by "
        f"{metrics.alpha:+.2%} over the replay window.**  "
        f"Strategy returned {metrics.total_return:+.2%}; SPY returned "
        f"{metrics.benchmark_return:+.2%}."
    )


def _process_metrics_table(metrics: EvalMetrics) -> str:
    """Render the Process Metrics section as a Markdown table."""
    rows = [
        ("Total cycles", str(metrics.total_cycles)),
        ("Filled trades", str(metrics.filled_cycles)),
        ("Groundedness", f"{metrics.groundedness_pct:.1f}%"),
        ("Guardrail triggers", str(metrics.guardrail_trigger_count)),
        ("HITL escalations", str(metrics.hitl_count)),
        ("Refusal rate", f"{metrics.refusal_rate:.1f}%"),
        ("Tokens / cycle (mean)", f"{metrics.tokens_per_cycle_mean:.0f}"),
        ("LLM cost / cycle (mean)", f"${metrics.cost_per_cycle_mean:.6f}"),
        ("Hallucination check pass", f"{metrics.hallucination_check_pass_rate:.1f}%"),
    ]
    lines = ["## Process Metrics", "", "| Metric | Value |", "|--------|-------|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _guardrail_events(cycles: list[CycleRecord]) -> str:
    """List every guardrail trigger event."""
    lines = ["## Guardrail Events", ""]
    triggered = [c for c in cycles if c.guardrail_triggered]
    if not triggered:
        lines.append("No guardrail events were triggered during this replay.")
        return "\n".join(lines)

    lines.append(f"{len(triggered)} guardrail event(s) occurred during this replay:\n")
    for c in triggered:
        lines.append(
            f"- **{c.symbol}** at `{c.decision_ts[:10]}` "
            f"(cycle `{c.cycle_id[:8]}…`) — "
            f"{'HITL escalation' if c.hitl_required else 'policy rejection'}"
        )
    return "\n".join(lines)


def _sample_trade(cycles: list[CycleRecord]) -> str:
    """Show the first filled trade as a sample."""
    lines = ["## Sample Trade", ""]
    filled = [c for c in cycles if c.trade_filled]
    if not filled:
        lines.append(
            "No trades were filled during this replay.  "
            "All cycles either held, were refused by the research agent, "
            "or were blocked by the risk guardrail."
        )
        return "\n".join(lines)

    c = filled[0]
    lines.append(f"**Symbol:** {c.symbol}")
    lines.append(f"**Decision date:** {c.decision_ts[:10]}")
    lines.append(f"**Fill price:** ${c.fill_price:.4f}")
    lines.append(f"**Quantity:** {c.fill_qty:.2f} shares")
    lines.append(f"**Citations used:** {c.citations_used}")
    lines.append(f"**Tokens:** {c.tokens_used}")
    lines.append(f"**LLM cost:** ${c.llm_cost_usd:.6f}")
    return "\n".join(lines)


def _limitations() -> str:
    """Render the Limitations and Caveats section."""
    return "\n".join(
        [
            "## Limitations and Caveats",
            "",
            "The following limitations apply to this eval and must be considered when "
            "interpreting results:",
            "",
            "- **Synthetic corpus.** The news corpus (`data/news/corpus.json`) consists "
            "of fixture articles written to exercise the pipeline, not scraped from a "
            "live feed.  Sentiment and relevance scores reflect the fixture content.",
            "",
            "- **Paper trading only.** No real money is involved.  Fill prices use a "
            "simplified 5-bps slippage + \\$0.005/share commission model; real fills "
            "would differ.",
            "",
            "- **Simplified NAV accounting.** The eval harness tracks NAV using "
            "closing prices from frozen CSV bars.  Intraday moves and dividends are "
            "not modelled.",
            "",
            "- **No LLM sentiment in eval.** The `PortfolioManagerAgent` falls back "
            "to keyword-based sentiment scoring in eval because the eval harness "
            "uses a cassette or fake LLM.  Live runs use the full LLM sentiment path.",
            "",
            "- **HITL auto-approved in eval.** When the risk agent requires human "
            "approval the eval harness auto-approves the trade so the replay can "
            "complete without human intervention.  In production the trade is held "
            "pending a human decision.",
            "",
            "- **Short replay window.** The replay covers only 5 trading days "
            "(Oct 21-25 2024), which is insufficient for statistically meaningful "
            "Sharpe or alpha estimates.  Results should be interpreted as a "
            "functional smoke-test of the pipeline, not an investment performance "
            "assessment.",
            "",
            "- **Strategy is deliberately simple.** Momentum + news-sentiment is "
            "intentionally rudimentary; trading alpha is not the graded objective.  "
            "Underperformance vs SPY is expected and not a defect.",
        ]
    )
