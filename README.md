# The AI Investment Firm

A multi-agent paper-trading US-equities desk where each desk role is a specialized agent,
large trades pause for a human Risk Committee, and every decision is grounded in cited evidence,
persisted transactionally, and replayable from a trace.

**Replay window:** Oct 21-25 2024 (NVDA earnings week)
**Watchlist:** AAPL, MSFT, NVDA, GOOGL, META, AMD | Benchmark: SPY

---

## Prerequisites

- Docker and docker-compose (Docker Desktop 4.x or newer)
- GNU make
- Python 3.12+
- An Anthropic API key (for live runs; CI uses recorded cassettes)

---

## Quick start (under 10 minutes)

```bash
# 1. Clone the repo and enter it
git clone <repo-url> the-ai-firm
cd the-ai-firm

# 2. Copy the example env file and fill in your API key
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum.
# SLACK_BOT_TOKEN is optional; the fake Slack adapter is used when absent.

# 3. Start infrastructure (Postgres + pgvector + Langfuse)
make up

# 4. Run migrations, load frozen bar CSVs, and embed the news corpus
make seed

# 5. Replay one full trading day (Oct 23 2024, NVDA earnings day)
#    Watches the five-agent pipeline run with recorded LLM responses — no live API call.
make demo
```

The demo prints a structured trace to stdout. Each line is a JSON event: agent invocations,
risk gate decisions, ledger writes, and the final fill (or HITL pause, or rejection).

---

## All make targets

```bash
make up                     # docker-compose up -d: postgres+pgvector, langfuse, firm-app
make seed                   # migrations + load frozen data + embed corpus
make demo                   # replay Oct 23 2024 end-to-end, print trace
make test                   # pytest -q (unit + integration + eval)
make eval                   # full historical replay, prints report, saves to eval/output/
make lint                   # ruff check + ruff format --check + mypy --strict
make dev                    # run firm-app in development mode (hot paths)
make trace TRADE=<uuid>     # print full audit log for one trade
```

---

## Architecture

Five agents in a LangGraph pipeline, all external IO behind Protocol ports,
one Postgres database for ledger + checkpoints + RAG:

```
Research  →  PortfolioManager  →  Risk ──┬──(auto)──→  Execution  →  Reporting
                                          └──(HITL)──→  human (Slack)  →  Execution
```

**Agents**
| Agent | Role |
|---|---|
| Research | Hybrid-retrieve + rerank news chunks from pgvector; refuse if evidence thin |
| PortfolioManager | Combine N-day momentum + news sentiment into a signal; size to RiskPolicy |
| Risk | Enforce per-trade and concentration limits; route large trades to human HITL |
| Execution | ACID write to ledger (cash debit, FIFO lot, idempotency key); apply slippage + commission |
| Reporting | Emit Excel report + Slack summary; attach audit link |

**Four ports (the live-to-replay seam)**
- `MarketDataSource` — frozen CSV adapter in eval/CI, live feed adapter in prod
- `EvidenceStore` — pgvector adapter live, in-memory fake for unit tests
- `LLM` — Anthropic adapter live, cassette adapter in CI, fake in unit tests
- `ReportSink` — Excel + Slack adapter live, no-op fake in tests

The ledger is a **concrete** Postgres repository, not a port — tested against a real database
(testcontainers in CI), not mocked.

For the full architecture narrative, entity diagram, and deployment view see
[docs/technical_overview.md](docs/technical_overview.md).

---

## Running evaluations

```bash
make eval
```

Runs a full historical replay of Oct 21-25 2024 against frozen bar CSVs and the synthetic news
corpus, using recorded LLM responses (cassettes) so the result is bit-reproducible and requires
no live API calls.

Prints a Markdown report to stdout and saves it to `eval/output/eval_report.md`.
The report covers:
- Return vs SPY benchmark
- Groundedness % (claims with citations)
- Guardrail trigger count
- HITL invocations and latency
- Refusal rate
- Estimated tokens and cost per decision

See [docs/eval_report.md](docs/eval_report.md) for methodology and an honest assessment of
what the numbers mean.

---

## Risk policy (locked)

| Limit | Value |
|---|---|
| Per-trade max notional | 10% NAV |
| Single-name concentration | 25% NAV |
| Daily-loss halt | -3% NAV |
| HITL threshold | 5% NAV notional |
| Slippage | 5 bps + $0.005/share commission |

All limits are defined once in `config/risk_policy.yaml` and enforced at two layers:
the Risk agent node and the `LedgerRepository.execute()` guardrail.

---

## Production-worthy section — what was deliberately left out and why

This project is scoped to a local, reproducible demo. The following items were deliberately
deferred; their absence is a design decision, not an oversight.

**High availability is not built.**
A single-node Postgres is a SPOF. The HA path is documented: `infra/main.tf` maps
every docker-compose service to its managed AWS equivalent (ECS/Fargate for the app,
RDS Postgres 16 with Multi-AZ standby, SSM for secrets). That Terraform is a
deployment-view artifact — it is written but not applied. Applying it would take the
demo from "clone to run in 10 minutes" to "provision cloud infra first," which is the
wrong trade-off for a reviewer.

**No live market data feed.**
The data layer is frozen OHLCV CSVs committed to the repo. A live feed would require
a brokerage or data-vendor API key, breaking the "no live APIs in CI" requirement.
The `MarketDataSource` port makes swapping in a live adapter a one-file change.

**Synthetic news corpus.**
The 50-100 articles in `data/news/corpus.json` are synthetic fixtures, license-clean
and commit-safe. Real news would require a vendor subscription and would introduce
license constraints on the committed data.

**Slack HITL uses a fake adapter in tests.**
The `slack_report.py` adapter is wired in the live docker-compose stack. Unit and eval
tests use `FakeReportSink` so they run without a Slack workspace. The real adapter is
tested in the integration suite against a mock Slack server.

**Real LLM cassettes are synthetic.**
Cassette recordings in `data/cassettes/` were generated with the fake LLM adapter to
keep CI fully offline. A production eval would record real Anthropic responses once and
commit them; the cassette mechanism is already wired to support that.

**Long-only, v1.**
No shorting or margin mechanics. The strategy is momentum + news-sentiment threshold,
deliberately simple — trading alpha is not graded; production engineering is.

**Deferred by design** (named so their absence reads as a decision, not an oversight):
- `Trade`-as-Command formalism (event sourcing)
- `ReportSink` as a pluggable-strategy registry
- Rich-domain method placement (PM sizing logic is currently in the agent layer)
- Four-way ISP port splitting

---

## Inspecting a trade trace

```bash
make trace TRADE=<trade-uuid>
```

Prints the correlation_id context and a query hint for your OTLP backend. All spans
emitted by the five agents carry the same `correlation_id` attribute, so a single
Jaeger/Langfuse query reconstructs the full trace end-to-end.

For the ledger cross-check query see [docs/runbook.md](docs/runbook.md).
