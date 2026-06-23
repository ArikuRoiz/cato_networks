"""Argument parser and ``main()`` dispatch for the ``firm`` CLI.

This module wires the per-command handlers (``firm.cli.commands.*``) to the
argparse subcommands and is intentionally thin: all command logic lives in the
command modules, all shared I/O helpers in ``firm.cli.output``.
"""

from __future__ import annotations

import argparse
from typing import Any

from firm.cli.commands.bot import _cmd_bot
from firm.cli.commands.demo import _cmd_demo
from firm.cli.commands.dev import _cmd_dev
from firm.cli.commands.run import _cmd_run
from firm.cli.commands.seed import _cmd_seed
from firm.cli.commands.trace import _cmd_trace
from firm.cli.commands.web import _cmd_web
from firm.cli.output import _load_dotenv
from firm.constants import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_WATCHLIST,
    DEFAULT_WEB_PORT,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="firm",
        description="The AI Investment Firm — multi-agent paper-trading desk",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_seed_subcommand(sub)
    _add_demo_subcommand(sub)
    _add_dev_subcommand(sub)
    _add_run_subcommand(sub)
    _add_bot_subcommand(sub)
    _add_trace_subcommand(sub)
    _add_web_subcommand(sub)
    return parser


def _add_seed_subcommand(sub: Any) -> None:
    """Register the 'seed' subcommand."""
    sub.add_parser(
        "seed",
        help=(
            "Run Alembic migrations, verify frozen bar CSVs, and embed the news corpus "
            "into the evidence store."
        ),
    )


def _add_demo_subcommand(sub: Any) -> None:
    """Register the 'demo' subcommand."""
    sub.add_parser(
        "demo",
        help=(
            "Replay Oct 23 2024 (NVDA earnings day) end-to-end against frozen data "
            "and recorded LLM responses. Prints a structured trace to stdout."
        ),
    )


def _add_dev_subcommand(sub: Any) -> None:
    """Register the 'dev' subcommand."""
    sub.add_parser(
        "dev",
        help=(
            "Start the scheduler + event listener in a foreground loop against frozen data. "
            "Press Ctrl+C to stop."
        ),
    )


def _add_run_subcommand(sub: Any) -> None:
    """Register the 'run' subcommand (live production mode)."""
    run_p = sub.add_parser(
        "run",
        help=(
            "LIVE production run: fetch real market data + news, run the 11-node graph "
            "against Postgres, and write a daily report. "
            "Requires: make up, make seed, ANTHROPIC_API_KEY in .env."
        ),
    )
    run_p.add_argument(
        "--tickers",
        default=None,
        metavar="TICKER,TICKER,...",
        help=(
            f"Comma-separated list of tickers to analyse (default: {','.join(DEFAULT_WATCHLIST)})."
        ),
    )
    run_p.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        dest="lookback_days",
        metavar="N",
        help=f"Number of calendar days of market data + news to pull (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    from firm.adapters.approval import AVAILABLE_CHANNELS

    run_p.add_argument(
        "--hitl",
        choices=list(AVAILABLE_CHANNELS),
        default="auto",
        dest="hitl",
        help=(
            "HITL approval channel. Choices: "
            "'console' = interactive stdin prompt; "
            "'telegram' = Telegram bot blocking long-poll "
            "(requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env); "
            "'slack' = Slack Block Kit card (send wired; receive needs an interactivity "
            "webhook — returns EXPIRED until then); "
            "'auto' = telegram when configured, else console (default: auto)."
        ),
    )
    run_p.add_argument(
        "--force-buy",
        action="store_true",
        default=False,
        dest="force_buy",
        help=(
            "DEMO OVERRIDE: inject a synthetic high-conviction BUY plan that sizes a trade "
            "> 5%% NAV, guaranteeing a HITL interrupt through the real graph. "
            "Does not affect default behaviour; intended for end-to-end HITL testing only."
        ),
    )


def _add_trace_subcommand(sub: Any) -> None:
    """Register the 'trace' subcommand."""
    trace_p = sub.add_parser(
        "trace",
        help="Print the full audit log for one trade, identified by --trade-id.",
    )
    trace_p.add_argument(
        "--trade-id",
        required=True,
        metavar="UUID",
        help="UUID of the trade whose audit log should be printed.",
    )


def _add_bot_subcommand(sub: Any) -> None:
    """Register the 'bot' subcommand (Telegram operator interface)."""
    sub.add_parser(
        "bot",
        help=(
            "Start the Telegram bot service. Operator types /run TICKER; the bot runs "
            "the live pipeline, sends a rich HITL approval card, and reports back the "
            "fill + memo. Requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env."
        ),
    )


def _add_web_subcommand(sub: Any) -> None:
    """Register the 'web' subcommand (browser dashboard)."""
    web_p = sub.add_parser(
        "web",
        help=(
            f"Start the FastAPI dashboard on http://localhost:{DEFAULT_WEB_PORT}. "
            "Requires DATABASE_URL; ANTHROPIC_API_KEY enables HITL endpoints."
        ),
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        metavar="PORT",
        help=f"TCP port to listen on (default: {DEFAULT_WEB_PORT}).",
    )
    web_p.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Host to bind (default: 0.0.0.0).",
    )
    web_p.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn hot-reload (dev mode).",
    )


def main() -> None:
    """Entry point invoked by pyproject.toml [project.scripts] and python -m firm.cli."""
    _load_dotenv()
    from firm.observability import setup_telemetry  # deferred: heavy import

    setup_telemetry()
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "seed": _cmd_seed,
        "demo": _cmd_demo,
        "dev": _cmd_dev,
        "run": _cmd_run,
        "bot": _cmd_bot,
        "trace": _cmd_trace,
        "web": _cmd_web,
    }
    dispatch[args.command](args)
    from firm.observability import flush_telemetry

    flush_telemetry()
