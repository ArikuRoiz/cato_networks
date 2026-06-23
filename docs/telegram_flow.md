# Telegram Operator Flow

> How the persistent Telegram bot (`firm bot`) drives a human-in-the-loop (HITL)
> decision cycle end-to-end, the Telegram API calls it uses, and how it differs
> from the one-shot `firm run --hitl telegram` path.

The approval channel is a **pluggable skill**. Telegram is the shipped operator
channel; Slack / email / SMS are drop-in adapters behind the same
`resume_decision` core (`firm.orchestration.hitl.resume_decision`). Nothing in
the graph knows about Telegram — it raises a LangGraph `interrupt()` and waits.

---

## Sequence — `firm bot`

```
operator                bot service (single getUpdates loop)         graph + PostgresSaver
   │                              │                                          │
   │  /run NVDA  ───────────────► │                                          │
   │   (or bare "NVDA")           │  spawn background thread:                │
   │                              │    graph.stream(thread_id=cid) ────────► │ research → … → pm
   │                              │                                          │   → risk node
   │                              │                                          │   risk: interrupt()
   │                              │                                          │   ─ checkpoint to PG ─
   │                              │  ◄──── GraphInterrupt(payload) ──────────│   thread paused
   │  ◄── sendMessage (card +     │                                          │
   │      inline_keyboard) ───────│  build approval card from payload        │
   │                              │                                          │
   │  tap ✅ Approve ───────────► │  getUpdates → callback_query             │
   │                              │  answerCallbackQuery (clear spinner)     │
   │                              │  resume_decision(graph, cid, APPROVE) ──►│ risk re-executes,
   │                              │                                          │   interrupt() returns,
   │                              │                                          │   → execution → reporting
   │                              │                                          │   → synthesis → judge → END
   │  ◄── sendMessage             │  format "what I did & why" + run report  │
   │      (decision + report) ────│                                          │
```

### Reject → override (two-step)

When the operator taps **❌ Reject**, the bot does **not** book a rejection. It
replaces the keyboard with the alternative-action keyboard so the human can pick
the action that actually executes:

```
   │  tap ❌ Reject ────────────► │  getUpdates → callback_query             │
   │  ◄── sendMessage (alts:      │  answerCallbackQuery                     │
   │      🟢 Buy 🔴 Sell ⏸️ Hold)──│                                          │
   │  tap 🟢 Buy ───────────────► │  resume_decision(graph, cid,             │
   │                              │      OVERRIDE_BUY) ─────────────────────►│ override rewrites
   │                              │                                          │ trade_proposal → Buy,
   │                              │                                          │ sizes + executes
   │  ◄── sendMessage (override   │                                          │
   │      report) ────────────────│                                          │
```

The override **rewrites the cycle's `trade_proposal`** inside the risk node
(`_proposal_from_approved` / `_override_hold_proposal` in
`src/firm/orchestration/nodes.py`) so the downstream synthesis memo, run report,
and judge all describe **what actually executed** — an override-buy never yields
a stale "Hold / no position change" memo.

Hard `RiskPolicy` limits still enforce at the execution guardrail even after a
human approval/override: `LedgerGuardrail.enforce_hitl_approved` bypasses only the
soft HITL gate (`HITLRequired`); a `Rejected` (hard limit breach) still raises
`LimitExceeded` and the trade fails.

---

## Telegram API calls used

All calls go to `https://api.telegram.org/bot<TOKEN>/<method>` (`TelegramHITL`
and `BotService` in `src/firm/adapters/telegram.py`).

| Method | Use |
|---|---|
| `getUpdates` | **Single long-poll consumer** — offset-based, ~30s windows. Pulls the operator's commands and button taps. Used by both the bot and the one-shot HITL. |
| `sendMessage` | Posts the approval card (with `parse_mode: MarkdownV2` + `reply_markup.inline_keyboard`) and the post-decision report. |
| `answerCallbackQuery` | Clears the button spinner / shows a toast the instant a button is tapped. |

Callback-data scheme (`src/firm/bot/service.py`): `approve:<cid>`,
`reject:<cid>`, `act:<verb>:<cid>` where `verb ∈ {buy, sell, hold}` and `<cid>`
is the cycle correlation id.

### No webhook, durable

The bot uses **getUpdates long-polling, not a webhook** — no public URL, no
inbound port, runs anywhere with outbound HTTPS. Durability comes from the
**PostgresSaver checkpoint**: the graph state is written to Postgres at the
`interrupt()`, so if the bot process restarts while a card is outstanding, the
operator can still tap the button and `resume_decision` picks the thread up from
the last checkpoint. The bot's pending-thread registry maps `correlation_id →
(pending run, resume signal)` so the callback handler can wake the right paused
cycle.

If the token is missing or ends with `...` (dry-run), payloads are logged at INFO
and never sent; the HITL surface returns `EXPIRED` (fail-safe — no trade).

---

## Contrast — `firm run --hitl telegram` (one-shot)

`firm run --hitl telegram` is the **one-shot** live path (`_run_live_graph_loop`
in `src/firm/cli.py`): it runs the graph once per ticker, and when the risk node
interrupts it sends a single approval card via `TelegramHITL.send_hitl_request`
and **blocks** on a getUpdates long-poll for the tap, then calls
`resume_decision`. It is a simpler surface than the bot:

| | `firm bot` | `firm run --hitl telegram` |
|---|---|---|
| Lifetime | persistent operator service | single invocation, exits when done |
| Trigger | operator types `/run <ticker>` / bare ticker | CLI `--tickers` list |
| Reject UI | two-step → Buy / Sell / Hold alternatives | single tap; approve→APPROVE, reject→OVERRIDE_HOLD, timeout→EXPIRE |
| Concurrency | non-blocking; background thread per run | blocking long-poll inline |
| Reports back | decision report + run report to the chat | NDJSON to stdout + Excel/Slack report |

Both share the same `resume_decision` core and the same PostgresSaver checkpoint.
Run the bot with `make bot` (or `firm bot`).
