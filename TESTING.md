# Testing Guide — The AI Investment Firm

## Prerequisites

- Docker Desktop running
- `.env` file at project root with `ANTHROPIC_API_KEY` set
- Python 3.12+ and `uv` installed (for local runs)

---

## Quick start

```bash
# Full production run in Docker (build → infra → demo)
make prod-demo

# Local demo (uses .env auto-loaded, no Docker needed)
make demo
```

---

## What "demo" does

The demo command runs one full decision cycle for 6 symbols: `AAPL MSFT NVDA GOOGL META AMD`.

For each symbol the pipeline executes 11 agents in sequence:

```
Research → [Bull debate ↔ Bear debate] → ResearchManager
→ Technical → PM → Execution → Synthesis → Judge → Reporting
```

Each cycle emits three JSON lines to stdout:

```json
{"event": "demo_start", "date": "...", "watchlist": [...]}
{"event": "cycle_start", "symbol": "AAPL", "correlation_id": "..."}
{"event": "cycle_done",  "symbol": "AAPL", "outcome": "rejected", ...}
```

---

## Reading the output

| Field | What it means |
|---|---|
| `outcome` | `"rejected"` = PM held; `"approved"` = trade was sent to execution |
| `synthesis_title` | Real LLM title → real API call worked; `"none"` → FakeLLM fallback |
| `judge_score` | 1–5 coherence score from Judge agent (3 = fallback / FakeLLM) |
| `judge_alignment` | `"aligned"` / `"partial"` / `"misaligned"` |
| `research_plan` | `"dict(symbol, ...)"` means the field is a dict object (truncated for display) |

**FakeLLM vs real LLM:**

| Signal | FakeLLM (offline) | Real Claude API |
|---|---|---|
| Duration | ~1 second total | 3–5 minutes total |
| `synthesis_title` | `"none"` | Proper memo title |
| `judge_score` | always `3` | varies per symbol |
| Trigger | No `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` set in env |

---

## Test scenarios

### 1. Offline / CI (no API key)

Verifies the pipeline wires up and all nodes run without errors.

```bash
make demo
```

Expected: all 6 symbols complete, `outcome: rejected`, `synthesis_title: "none"`, duration ~1s.

---

### 2. Production run in Docker

Verifies the full stack: Docker build, Postgres health, real Claude API calls.

```bash
make prod-demo
```

Expected:
- Build output from Docker, then infra start, then demo output
- Duration 3–5 minutes
- `synthesis_title` contains a real memo title (e.g. `"Investment Decision Memo: AAPL — 2024-10-23"`)
- `judge_score` varies across symbols (not all 3)
- Report written to `reports/report_2024-10-23.txt`

---

### 3. Check the text report

After any demo run a report file is written locally:

```bash
cat reports/report_2024-10-23.txt
```

Expected sections: `PORTFOLIO SUMMARY`, and optionally `TRADES`, `POSITIONS`, `EVIDENCE CITED`.
With FakeLLM or a reject-only run, trades and positions will be empty (NAV stays at $100,000).

---

### 4. Unit tests

```bash
make test
```

Runs pytest with coverage. All tests use cassette replay or FakeLLM — no API key needed.

---

### 5. Lint and type check

```bash
make lint
```

Must pass cleanly: ruff (format + lint) and mypy on all 91 source files.

---

### 6. Infra only (Postgres + Langfuse, no app)

```bash
make up        # start Postgres + Langfuse in background
make down      # stop
```

Langfuse UI available at `http://localhost:3000` once up.

---

## Make targets reference

| Target | Description |
|---|---|
| `make demo` | Local demo, reads `.env` automatically |
| `make dev` | Local dev loop (scheduler + event listener) |
| `make seed` | Seed the corpus / run migrations locally |
| `make test` | pytest with coverage |
| `make lint` | ruff + mypy |
| `make up` | Start Postgres + Langfuse only |
| `make down` | Stop infra |
| `make build` | Build the Docker image |
| `make prod-demo` | Full Docker production demo (recommended for first run) |
| `make prod-seed` | Run migrations + corpus seed inside Docker |
| `make prod` | Start full stack in background (persistent) |
| `make prod-logs` | Tail firm-app container logs |
| `make prod-down` | Stop all containers and delete volumes |

---

## Common issues

**Demo runs in ~1 second with `synthesis_title: "none"`**
→ `ANTHROPIC_API_KEY` is not in the environment. Check `.env` has the key and re-run.

**`make prod-demo` fails during Docker build**
→ Check that Docker Desktop is running and you have internet access for image pulls.

**`make prod-demo` exits immediately after infra starts**
→ Look at the last line before exit — likely a Python import error. Run `make demo` locally first to isolate.

**`reports/` directory not created**
→ The `reports/` volume mount in `docker-compose.yml` maps `./reports:/app/reports`. Create the dir: `mkdir -p reports`.
