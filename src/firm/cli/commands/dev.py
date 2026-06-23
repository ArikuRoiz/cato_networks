"""``firm dev`` — foreground scheduler loop against frozen data."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from firm.cli.commands.demo import _run_graph_loop
from firm.cli.output import _emit
from firm.constants import DEFAULT_INITIAL_CASH, DEFAULT_WATCHLIST, DEV_POLL_INTERVAL_SECONDS


def _cmd_dev(args: argparse.Namespace) -> None:
    """Start the scheduler + event listener in a foreground loop.

    Fires a decision cycle for each watchlist symbol on a 30-second polling
    interval using frozen data and the fake/cassette LLM.  Press Ctrl+C to
    stop.
    """
    import time

    from firm.cli.output import _project_root
    from firm.composition import build_offline_pipeline  # deferred: heavy import

    root = _project_root()
    watchlist = list(DEFAULT_WATCHLIST)
    poll_interval_s = DEV_POLL_INTERVAL_SECONDS

    _emit({"event": "dev_start", "watchlist": watchlist, "poll_interval_s": poll_interval_s})

    pipeline = build_offline_pipeline(root, initial_cash=DEFAULT_INITIAL_CASH)
    cycle_count = 0
    try:
        while True:
            now = datetime.now(tz=UTC)
            _emit({"event": "tick", "ts": now.isoformat(), "cycle": cycle_count})
            _run_graph_loop(pipeline.graph, watchlist, now)
            cycle_count += 1
            _emit({"event": "sleeping", "seconds": poll_interval_s})
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        _emit({"event": "dev_stop", "cycles_run": cycle_count})
