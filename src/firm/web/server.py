"""``firm web`` command — launch the dashboard with uvicorn.

Usage:
    firm web                          # port 8000
    firm web --port 8080
    firm web --reload                 # dev mode hot-reload

The server wires a SQLAlchemy engine from DATABASE_URL and optionally a live
graph (when ANTHROPIC_API_KEY is present) so the HITL endpoints work.

Splitting this from app.py keeps the app factory import-time-clean (no heavy
DB/adapter imports at module level) so TestClient tests stay fast.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Wire DB + live graph, inject into the app, then start uvicorn."""
    _load_dotenv()
    database_url = os.environ.get("DATABASE_URL", "postgresql://firm:firm@localhost:5432/firm")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    engine = _build_engine(database_url)
    portfolio_id = _resolve_portfolio_id(engine)

    live_graph = None
    if anthropic_key:
        try:
            from firm.config.settings import load_settings
            from firm.web.runtime import build_live_graph

            settings = load_settings()
            live_graph = build_live_graph(settings)
        except Exception as exc:
            print(f"[web] Live graph unavailable ({exc}); HITL endpoints disabled.")

    from firm.web.app import configure, create_app

    app = create_app()
    configure(engine=engine, portfolio_id=portfolio_id, live_graph=live_graph)

    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    root = Path(__file__).parent.parent.parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def _build_engine(database_url: str) -> object:
    from sqlalchemy import create_engine

    from firm.persistence.db_url import to_sqlalchemy_url

    return create_engine(to_sqlalchemy_url(database_url))


def _resolve_portfolio_id(engine: object) -> uuid.UUID:
    """Return the first portfolio_id found in the DB, or a fresh UUID.

    A real deployment would have exactly one portfolio.  This gracefully
    handles a fresh database (no portfolio yet) by returning a new UUID —
    the portfolio card will show zeros until the first ``make seed`` or ``POST /api/run``.
    """
    try:
        from sqlalchemy import text
        from sqlalchemy.engine import Engine

        with Engine.connect(engine) as conn:  # type: ignore[arg-type]
            row = conn.execute(text("SELECT id FROM portfolios LIMIT 1")).fetchone()
            if row is not None:
                return uuid.UUID(str(row[0]))
    except Exception:
        pass
    return uuid.uuid4()
