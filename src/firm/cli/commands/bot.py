"""``firm bot`` — Telegram operator interface."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from firm.cli.commands.run import _validate_live_settings
from firm.cli.output import _emit, _load_settings
from firm.constants import DEFAULT_INITIAL_CASH

if TYPE_CHECKING:
    from firm.config.settings import Settings


def _cmd_bot(args: argparse.Namespace) -> None:
    """Start the Telegram bot service (blocking — press Ctrl+C to stop).

    Prerequisites (same as 'firm run'):
      - ``make up``   — Postgres + pgvector running.
      - ``make seed`` — migrations applied, tables exist.
      - ``.env``      — ANTHROPIC_API_KEY, DATABASE_URL,
                        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
    """
    from firm.cli.output import _project_root
    from firm.composition import build_live_pipeline  # deferred: heavy import

    root = _project_root()
    settings = _load_settings()
    _validate_live_settings(settings)
    _validate_telegram_settings(settings)

    _emit(
        {
            "event": "bot_start",
            "ts": datetime.now(tz=UTC).isoformat(),
        }
    )

    pipeline = build_live_pipeline(settings, root=root, initial_cash=DEFAULT_INITIAL_CASH)

    from firm.bot.service import build_bot_service

    bot = build_bot_service(settings=settings, graph=pipeline.graph)
    try:
        bot.run()
    except KeyboardInterrupt:
        _emit({"event": "bot_stop", "ts": datetime.now(tz=UTC).isoformat()})


def _validate_telegram_settings(settings: Settings) -> None:
    """Raise SystemExit with a clear message when Telegram credentials are absent."""
    if not settings.has_telegram:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required for 'firm bot'. "
            "Add them to .env or export before running."
        )
