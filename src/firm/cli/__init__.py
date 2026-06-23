"""The ``firm`` command-line interface.

Commands
--------
seed   -- Run Alembic migrations, load frozen bar CSVs into the frozen-data
          directory, and embed the news corpus into pgvector.
demo   -- Run one full replay day (Oct 23 2024, NVDA earnings day) against
          frozen data with recorded LLM responses. Prints structured trace
          to stdout in NDJSON format.
dev    -- Start the scheduler + event listener in a foreground loop against
          frozen data. Use for local development with hot-reload.
run    -- LIVE production run: pull real market data + news for the last N
          days, run the 11-node graph against Postgres, and write a daily
          report. Requires Docker (make up), make seed, and a .env with
          ANTHROPIC_API_KEY + DATABASE_URL.
bot    -- Start the Telegram bot service. Operator types /run TICKER in the
          chat; the bot runs the live pipeline, sends a rich approval card on
          HITL interrupt, and reports back the fill + memo. Requires
          TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env (plus all live deps).
trace  -- Print the audit log for a single trade from the database,
          identified by --trade-id (UUID). Outputs NDJSON to stdout.

All commands read DATABASE_URL (and other config) from the environment.
No secrets are hard-coded here.
"""

from __future__ import annotations

from firm.cli.main import main

__all__ = ["main"]
