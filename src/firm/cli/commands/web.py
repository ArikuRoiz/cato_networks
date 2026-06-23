"""``firm web`` — FastAPI dashboard server."""

from __future__ import annotations

import argparse


def _cmd_web(args: argparse.Namespace) -> None:
    """Start the FastAPI dashboard server via uvicorn."""
    from firm.web.server import run_server  # deferred: heavy import

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
