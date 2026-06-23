# The AI Investment Firm

Multi-agent paper-trading desk: Research, Technical, Debater (bull ⇄ bear), Research Manager
(sole decider), Reporting, and Judge agents in a LangGraph pipeline. Risk + Execution are
deterministic guardrail steps, not agents. Large trades pause for human approval; every decision
is grounded in cited evidence.

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

## Quick start — live production run

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY + DATABASE_URL
make up                       # Postgres + pgvector
make seed                     # migrations + frozen data + corpus
firm run --tickers NVDA,AAPL --lookback-days 7
```

`firm run` pulls the last 7 days of real market data and news via yfinance,
runs the full 11-node graph for each ticker against a live Postgres ledger, and
writes a daily Excel + Slack report.  Large trades (> 5% NAV) pause for console
HITL approval before execution.

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
research + technical (parallel)
        → debate (bull ⇄ bear ×N)
        → Research Manager (decide direction + conviction)   [SOLE decider — LLM]
        → size_position tool (deterministic sizing) + check_risk
        → [RISK GUARDRAIL]  →(>5% NAV)→ HITL interrupt → human approve/edit/reject
        → Execution (atomic ledger write)
        → Reporting agent (memo + Excel/Slack)
        → Judge (independent coherence audit, recorded)
```

**Agents (LLM judgment):** Research · Technical · Debater (one class, two roles) ·
Research Manager (sole decision agent) · Reporting · Judge.

**Portfolio Manager is not an agent** — it dissolves into the deterministic `size_position` +
`check_risk` tools. Risk and Execution are mandatory deterministic gates, not agents.

**Tools layer:** `search_news` · `fetch_live_news` · `price_indicators` · `compute_signal` ·
`size_position` · `check_risk` · `make_report` · `ledger_commit`.

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

---

## Telegram HITL — setup and local test

Large trades (> 5% NAV) pause the graph and await a human approve/reject decision.
The Telegram adapter delivers the request as an inline keyboard message and
long-polls for the tap.

### 1. Create the bot (@BotFather)

```
/start
/newbot
→ choose a name, e.g. "My Risk Committee Bot"
→ copy the token: 123456789:ABC-defGhi...
```

### 2. Get your chat ID

Send any message to the bot, then call:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
# Look for "chat":{"id":<YOUR_CHAT_ID>} in the response.
# For a private chat the ID is a positive integer; for groups it starts with -100.
```

Alternatively, message @userinfobot — it replies with your personal chat ID.

### 3. Add to .env

```
TELEGRAM_BOT_TOKEN=123456789:ABC-defGhi...
TELEGRAM_CHAT_ID=-100123456789
```

### 4. Run the HITL end-to-end test locally

```bash
firm run --force-buy --hitl telegram --tickers NVDA
```

What happens:
- `--force-buy` injects a synthetic conviction=1.0 BUY plan (skips LLM call in
  research-manager node) that sizes a ~10% NAV trade — well above the 5% threshold.
- The graph's risk node fires a genuine `interrupt()` (LangGraph checkpoint).
- The CLI builds a `HITLRequest` from the interrupt payload and calls
  `TelegramHITL.send_hitl_request(...)`.
- A message with **Approve / Reject / Edit qty** buttons appears in your Telegram chat.
- Tap a button; the bot answers the callback and resumes the graph with your decision.
- On timeout (10 min default) the trade is auto-rejected (EXPIRED, fail-safe).

Without `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` the adapter runs in dry-run mode:
payloads are logged at INFO level, the graph receives EXPIRED, and the trade is
rejected — no crash, no blocking.

### 5. Fallback to console

```bash
firm run --force-buy --hitl console --tickers NVDA
```

Prompts `Decision > [a]pprove [r]eject [e]dit <qty>` on stdin.
