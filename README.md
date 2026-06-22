# The AI Investment Firm

Multi-agent paper-trading desk: five agents (Research → PM → Risk → Execution → Reporting) in a
LangGraph pipeline, large trades pausing for human approval, every decision grounded in cited evidence.

**Replay window:** Oct 21-25 2024 (NVDA earnings week) | **Watchlist:** AAPL, MSFT, NVDA, GOOGL, META, AMD

---

## Quick start

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY
make up                       # Postgres + pgvector
make seed                     # migrations + frozen data + corpus
make demo                     # replay Oct 23 2024, prints NDJSON trace
```

No live API needed — `make demo` replays from recorded cassettes.

---

## Make targets

| Target | Description |
|---|---|
| `make up` | docker-compose: Postgres + pgvector + Langfuse |
| `make seed` | migrations + load bar CSVs + embed news corpus |
| `make demo` | replay Oct 23 2024 end-to-end, print trace |
| `make dev` | foreground loop against frozen data |
| `make test` | pytest -q (unit + integration + eval) |
| `make eval` | full 5-day replay, saves report to eval/output/ |
| `make lint` | ruff check + ruff format + mypy --strict |
| `make trace TRADE=<uuid>` | print audit log for one trade |

---

## Architecture

```
Research → PortfolioManager → Risk ──(auto)──→ Execution → Reporting
                                  └──(HITL)──→ human (Slack) → Execution
```

Four protocol ports (`MarketDataSource`, `EvidenceStore`, `LLM`, `ReportSink`) isolate live from
replay. The ledger is a concrete Postgres repository — tested against a real database, not mocked.

**Start reading:** `src/firm/ports/` for the seams, `src/firm/agents/` for the decision logic.

---

## Risk policy

| Limit | Value |
|---|---|
| Per-trade max notional | 10% NAV |
| Single-name concentration | 25% NAV |
| Daily-loss halt | −3% NAV |
| HITL threshold | 5% NAV |
| Slippage + commission | 5 bps + $0.005/share |
