# Operational Runbook — The AI Investment Firm

> Day-to-day operations: restart after crash, verify ledger integrity,
> re-run a failed cycle, inspect a trade trace.

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
from firm.orchestration.checkpointer import setup_checkpointer
from firm.orchestration.graph import build_graph

checkpointer = setup_checkpointer()
graph = build_graph(checkpointer)

# Resume with the same thread_id (= cycle_id) — LangGraph picks up from the last checkpoint
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
