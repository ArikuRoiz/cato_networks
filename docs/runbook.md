# Operational Runbook — The AI Investment Firm

> Day-to-day operations: restart after crash, verify ledger integrity,
> re-run a failed cycle, inspect a trade trace.

---

## 0. Running the live production mode

### Prerequisites

```bash
# 1. Copy and edit .env — set at minimum:
#    ANTHROPIC_API_KEY=sk-ant-...
#    DATABASE_URL=postgresql://firm:firm@localhost:5432/firm  (or Docker service name)
cp .env.example .env

# 2. Start Postgres + pgvector
make up

# 3. Apply migrations and embed news corpus
make seed
```

### Run a single analysis cycle

```bash
# Analyse two tickers, pulling 7 days of market data and news:
firm run --tickers NVDA,AAPL --lookback-days 7

# Use the default watchlist (AAPL, MSFT, NVDA, GOOGL, META, AMD):
firm run

# Custom lookback:
firm run --tickers MSFT --lookback-days 14
```

Output is NDJSON streamed to stdout (`run_start`, `news_ingestion`, one
`cycle_start` / `cycle_done` per ticker, `run_done`).  Excel reports are
written to `reports/` and, when `SLACK_BOT_TOKEN` is set, a Block Kit message
is sent to `SLACK_CHANNEL`.

### HITL (human-in-the-loop) approval

**Every cycle pauses for human approval** (`RiskPolicyConfig.hitl_mode="always"`).
The risk node fires a LangGraph `interrupt()`, checkpoints state to Postgres, and
waits. On the console:

```
[HITL] Decision required for NVDA (every cycle pauses)
  Recommendation: strong_buy
  Proposed trade: {...}
  Options: [a]pprove  [b]uy  [s]ell  [h]old
Decision (a/b/s/h) >
```

Type `a` to **approve** the desk's recommendation, or override it directly with
`b` (buy), `s` (sell), or `h` (hold). Unrecognised input defaults to a hold
override (safe — no trade, but a human decision is still recorded). An override
rewrites the cycle's `trade_proposal` so the synthesis memo, report, and judge
describe what actually executed.

`firm.orchestration.hitl.resume_decision(graph, thread_id, decision)` carries the
choice back into the graph. The decision is durably recorded in `approvals` +
`audit_log` so the full audit trail is intact. **Hard `RiskPolicy` limits still
enforce at the execution guardrail** even after approval — a human cannot wave
through a trade that breaches a hard cap.

To approve from Telegram instead, run with `--hitl telegram`, or operate the
persistent bot (see §0a below). The approval channel is a pluggable skill —
Slack / email / SMS adapters slot in behind the same `resume_decision` core.

### Durable checkpointing

`firm run` uses a `PostgresSaver` checkpointer.  If the process crashes during
a cycle the graph state is preserved in the `checkpoints` table and can be
resumed (see §3 below).

---

## 0a. Operating the Telegram bot

`firm bot` (or `make bot`) is the persistent operator service. It builds the same
live pipeline as `firm run` and then long-polls Telegram (`getUpdates` — no
webhook).

```bash
# Prereqs: make up + make seed, plus TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env
make bot          # blocking; Ctrl+C to stop
```

Operator workflow in the chat:

1. Type `/run NVDA` (or a bare `NVDA`). The bot runs the graph in a background thread.
2. When the risk node interrupts, an **approval card** appears: recommendation +
   💡 Why / 👍 Pros / 👎 Cons, with **✅ Approve** / **❌ Reject** buttons.
3. Tap **Approve** to execute the recommendation, or **Reject** to get the
   alternatives keyboard (**🟢 Buy / 🔴 Sell / ⏸️ Hold**) and override the desk.
4. The bot calls `resume_decision`, executes, and sends back a "what I did & why"
   message plus a run report.

**Restart:** the pause is durable via the PostgresSaver checkpoint. If the bot
process dies while a card is outstanding, restart `make bot`; the operator can
still tap the button and `resume_decision` picks the paused thread up from the
last checkpoint. If the token ends with `...` or is missing, the bot logs
payloads at INFO and never calls the API (dry-run, fail-safe).

See `docs/telegram_flow.md` for the full sequence and Telegram API calls.

---

## 1. Restart after crash

The application stores all durable state in Postgres. LangGraph checkpoints are written
before every interrupt (HITL pause) and after every node completion. A crash loses only
in-flight LLM reasoning — state is re-derivable.

### Full restart

```bash
# 1. Restart the compose stack (recreates containers; data volume survives)
docker-compose restart

# 2. Confirm Postgres is healthy
docker-compose ps postgres
# Expected: "healthy" in the STATUS column

# 3. Confirm the app container is running
docker-compose ps firm-app

# 4. If the app container exited, check logs and bring it back
docker-compose logs --tail=50 firm-app
docker-compose up -d firm-app
```

### If Postgres failed to start

```bash
docker-compose up -d postgres
# Wait for the healthcheck to pass (up to 30 s), then:
docker-compose up -d firm-app
```

### Confirm the schema is intact after restart

```bash
docker-compose exec postgres psql -U firm -d firm -c "\dt"
# Expected: portfolios, holdings, lots, trades, approvals,
#           decision_cycles, evidence, audit_log tables listed.
```

If tables are missing, re-run migrations:

```bash
make seed
```

---

## 2. Verify ledger integrity

This query cross-checks that the current cash balance is consistent with all filled trades.
It should return zero rows. If it returns rows, a partial write occurred and must be investigated.

```sql
-- Connect:  docker-compose exec postgres psql -U firm -d firm

-- Cross-check: starting cash + credits - debits should equal current cash_balance.
-- Starting cash is $100,000 for a fresh portfolio.
WITH filled AS (
    SELECT
        t.portfolio_id,
        SUM(
            CASE t.side
                WHEN 'buy'  THEN -(t.fill_price * t.qty + COALESCE(t.commission, 0))
                WHEN 'sell' THEN  (t.fill_price * t.qty - COALESCE(t.commission, 0))
                ELSE 0
            END
        ) AS net_flow
    FROM trades t
    WHERE t.status = 'filled'
    GROUP BY t.portfolio_id
)
SELECT
    p.id            AS portfolio_id,
    p.cash_balance  AS recorded_cash,
    100000 + COALESCE(f.net_flow, 0) AS expected_cash,
    p.cash_balance - (100000 + COALESCE(f.net_flow, 0)) AS discrepancy
FROM portfolios p
LEFT JOIN filled f ON f.portfolio_id = p.id
WHERE ABS(p.cash_balance - (100000 + COALESCE(f.net_flow, 0))) > 0.01;
-- A result set with rows means a discrepancy exists — investigate audit_log.
```

### Check for duplicate fills (idempotency regression)

```sql
SELECT idempotency_key, COUNT(*) AS cnt
FROM trades
WHERE status = 'filled'
GROUP BY idempotency_key
HAVING COUNT(*) > 1;
-- Should return 0 rows.
```

### Inspect recent audit entries for a cycle

```sql
SELECT actor, action, payload, ts
FROM audit_log
WHERE correlation_id = '<cycle-uuid>'
ORDER BY ts;
```

---

## 3. Re-run a failed cycle

Decision cycles can fail at any agent step. LangGraph checkpoints state before the risk
interrupt and after each node. If a cycle failed mid-run:

### Identify the failed cycle

```sql
SELECT id, trigger_type, started_at, outcome
FROM decision_cycles
WHERE outcome IS NULL OR outcome = 'error'
ORDER BY started_at DESC
LIMIT 10;
```

### Check the checkpoint

LangGraph stores checkpoints in the `checkpoints` table (created by the
`langgraph-checkpoint-postgres` package, not by Alembic). If a checkpoint exists,
the cycle can be resumed:

```python
# In a Python shell wired to the same DATABASE_URL:
from firm.orchestration.checkpointer import open_connection, setup_checkpointer
from firm.orchestration.graph import build_graph
from firm.orchestration.hitl import resume_decision

with open_connection(database_url) as conn:
    saver = setup_checkpointer(conn)           # wraps a caller-managed connection
    graph = build_graph(saver, ports)

    # If the thread is paused at a HITL interrupt, resume with a decision:
    resume_decision(graph, "<cycle-uuid>", "approve")   # or "override:buy", "override:hold", …

    # If it crashed mid-node (no interrupt), resume from the last checkpoint:
    graph.invoke(None, config={"configurable": {"thread_id": "<cycle-uuid>"}})
```

### If no checkpoint exists (crash before first node)

Re-trigger the cycle with the original inputs:

```bash
# For a demo re-run of Oct 23 2024
make demo
```

For a production event-triggered cycle, re-inject the event via the news event listener.

---

## 4. Inspect a trade trace

```bash
make trace TRADE=<trade-uuid>
```

This prints the `correlation_id` bound to the trade and a query hint. Use the
`correlation_id` to pull all related spans from your OTLP backend (Langfuse or Jaeger).

### Reconstruct from the audit log alone (no OTLP backend)

```sql
SELECT
    al.actor,
    al.action,
    al.payload,
    al.ts
FROM audit_log al
JOIN trades t ON t.cycle_id = al.correlation_id
WHERE t.id = '<trade-uuid>'
ORDER BY al.ts;
```

This returns every agent invocation, risk gate decision, and ledger write that belongs
to the same decision cycle as the trade. The full lifecycle — trigger → research →
PM proposal → risk gate → (HITL if applicable) → fill → report — is reconstructable
from these rows alone, with the binary closed.

### Langfuse query

If Langfuse is running (`make up` includes it on port 3000), open
`http://localhost:3000`, search traces by tag `correlation_id = <value>`.

---

## 5. Check pending HITL approvals

Approvals expire automatically. Check for stuck approvals:

```sql
SELECT
    a.id,
    a.trade_id,
    a.status,
    a.expires_at,
    t.symbol,
    t.qty,
    t.requested_price
FROM approvals a
JOIN trades t ON t.id = a.trade_id
WHERE a.status = 'pending'
ORDER BY a.expires_at;
```

Approvals past `expires_at` should be in status `expired` (auto-rejected by the Risk
agent timeout path). If they are stuck in `pending`, the firm-app process may have
crashed before the timeout handler ran — restart `firm-app` and the handler will pick up.

---

## 6. Emergency halt

If the daily-loss halt fires (`-3% NAV`), the risk agent rejects all new trades for the
remainder of the trading day. No manual intervention is required. To confirm:

```sql
SELECT action, payload, ts
FROM audit_log
WHERE action = 'daily_loss_halt'
ORDER BY ts DESC
LIMIT 5;
```

To resume trading the next calendar day, simply allow the scheduler to trigger the next
open-market cycle. The daily P&L counter resets at market open.

---

## 7. Useful one-liners

```bash
# Tail firm-app logs
docker-compose logs -f firm-app

# Postgres interactive shell
docker-compose exec postgres psql -U firm -d firm

# Check Langfuse health
curl -s http://localhost:3000/api/public/health | jq .

# List all filled trades
docker-compose exec postgres psql -U firm -d firm \
  -c "SELECT id, symbol, side, qty, fill_price, filled_at FROM trades WHERE status='filled' ORDER BY filled_at;"
```
