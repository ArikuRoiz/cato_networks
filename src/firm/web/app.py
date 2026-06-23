"""FastAPI application factory for the AI Investment Firm web dashboard.

Usage:
    uvicorn firm.web.app:create_app --factory --host 0.0.0.0 --port 8000

Module order: factory → routers → lifespan.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from firm.web.queries import (
    fetch_cycle_trace,
    fetch_portfolio,
    fetch_recent_cycles,
    fetch_recent_trades,
)
from firm.web.schemas import ApprovalRequest, RunRequest

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# App state container — typed, not a dict
# ---------------------------------------------------------------------------


class _AppState:
    """Holds shared resources available across request lifetimes."""

    engine: Engine
    portfolio_id: uuid.UUID

    def __init__(self, engine: Engine, portfolio_id: uuid.UUID) -> None:
        self.engine = engine
        self.portfolio_id = portfolio_id


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return the configured FastAPI application.

    Reads DATABASE_URL from the environment.  All routers are mounted here.
    """
    app = FastAPI(
        title="AI Investment Firm Dashboard",
        description="Observe and operate the AI paper-trading desk.",
        version="0.1.0",
    )

    _mount_static(app)
    _register_routes(app)
    return app


def _mount_static(app: FastAPI) -> None:
    static_dir = _HERE / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _register_routes(app: FastAPI) -> None:
    _dashboard_html = (_HERE / "templates" / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request) -> HTMLResponse:
        return HTMLResponse(content=_dashboard_html)

    _register_portfolio_route(app)
    _register_trades_route(app)
    _register_cycles_routes(app)
    _register_approvals_routes(app)
    _register_run_route(app)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def _register_portfolio_route(app: FastAPI) -> None:
    @app.get("/api/portfolio")
    async def get_portfolio() -> JSONResponse:
        """Return cash, holdings, NAV, and P&L for the active portfolio."""
        engine, portfolio_id = _get_engine_and_portfolio_id()
        try:
            dto = fetch_portfolio(engine, portfolio_id)
            return JSONResponse(dto.to_dict())
        except KeyError:
            return JSONResponse({"cash": "0", "nav": "0", "pnl": "0", "holdings": []})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


def _register_trades_route(app: FastAPI) -> None:
    @app.get("/api/trades")
    async def get_trades() -> JSONResponse:
        """Return the 50 most-recent trades."""
        engine, _ = _get_engine_and_portfolio_id()
        try:
            dtos = fetch_recent_trades(engine)
            return JSONResponse([d.to_dict() for d in dtos])
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Decision cycles + trace
# ---------------------------------------------------------------------------


def _register_cycles_routes(app: FastAPI) -> None:
    @app.get("/api/cycles")
    async def get_cycles() -> JSONResponse:
        """Return the 50 most-recent decision cycles."""
        engine, _ = _get_engine_and_portfolio_id()
        try:
            dtos = fetch_recent_cycles(engine)
            return JSONResponse([d.to_dict() for d in dtos])
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/cycles/{correlation_id}/trace")
    async def get_cycle_trace(correlation_id: str) -> JSONResponse:
        """Return ordered audit_log entries for one decision cycle."""
        engine, _ = _get_engine_and_portfolio_id()
        try:
            dtos = fetch_cycle_trace(engine, correlation_id)
            return JSONResponse([d.to_dict() for d in dtos])
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# HITL approvals
# ---------------------------------------------------------------------------


def _register_approvals_routes(app: FastAPI) -> None:
    @app.get("/api/approvals/pending")
    async def get_pending_approvals() -> JSONResponse:
        """Return trades currently paused for human-in-the-loop approval.

        Inspects the PostgresSaver checkpointer for threads that have an active
        interrupt.  Returns an empty list when no threads are pending or when
        the web server was started without a live graph (e.g. during tests).
        """
        live_graph = _get_live_graph()
        if live_graph is None:
            return JSONResponse([])

        from firm.web.runtime import pending_approvals
        from firm.web.schemas import PendingApprovalDTO

        pending = pending_approvals(live_graph)
        dtos = [
            PendingApprovalDTO(
                thread_id=p.thread_id,
                correlation_id=p.correlation_id,
                symbol=p.symbol,
                notional=p.notional,
                interrupt_payload=p.interrupt_payload,
            )
            for p in pending
        ]
        return JSONResponse([d.to_dict() for d in dtos])

    @app.post("/api/approvals/{thread_id}")
    async def post_approval(thread_id: str, body: ApprovalRequest) -> JSONResponse:
        """Resume a HITL-interrupted graph thread with approve or reject.

        Builds a ``Command(resume=…, update={"hitl_status": …})`` and streams
        the graph to completion.  The existing ``record_approval`` path in the
        risk node persists the decision.

        Returns the resolved cycle outcome.
        """
        live_graph = _get_live_graph()
        if live_graph is None:
            raise HTTPException(
                status_code=503,
                detail="No live graph available; start the server with make web.",
            )

        from firm.web.runtime import resume_approval
        from firm.web.schemas import ApprovalResultDTO

        try:
            result = resume_approval(
                live_graph,
                thread_id=thread_id,
                decision=body.decision,
                edited_qty=body.edited_qty,
            )
            dto = ApprovalResultDTO(
                thread_id=result["thread_id"],
                outcome=result["outcome"],
                hitl_status=result["hitl_status"],
            )
            return JSONResponse(dto.to_dict())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Run trigger
# ---------------------------------------------------------------------------


def _register_run_route(app: FastAPI) -> None:
    @app.post("/api/run")
    async def post_run(body: RunRequest, background_tasks: BackgroundTasks) -> JSONResponse:
        """Start a live run in the background for each ticker.

        HITL interrupts are left at the checkpoint — the operator resumes them
        via ``POST /api/approvals/{thread_id}``.

        Returns the list of started thread IDs.
        """
        live_graph = _get_live_graph()
        if live_graph is None:
            raise HTTPException(
                status_code=503,
                detail="No live graph available; start the server with make web.",
            )

        from firm.web.runtime import run_cycle_background
        from firm.web.schemas import RunStartedDTO

        thread_ids = []
        decision_ts = datetime.now(tz=UTC).isoformat()
        for ticker in body.tickers:
            thread_id = str(uuid.uuid4())
            thread_ids.append(thread_id)
            background_tasks.add_task(
                run_cycle_background,
                live_graph,
                ticker,
                decision_ts,
                thread_id,
                body.force_buy,
            )

        dto = RunStartedDTO(
            thread_ids=thread_ids,
            tickers=body.tickers,
            lookback_days=body.lookback_days,
            force_buy=body.force_buy,
        )
        return JSONResponse(dto.to_dict(), status_code=202)


# ---------------------------------------------------------------------------
# Process-level singletons (set by the CLI entry point)
# ---------------------------------------------------------------------------

# These are set once at startup by the ``firm web`` command.
# They are intentionally module-level (not FastAPI lifespan state) so
# the TestClient can inject fakes without lifespan machinery.
_ENGINE: Any = None
_PORTFOLIO_ID: uuid.UUID | None = None
_LIVE_GRAPH: Any = None


def configure(
    *,
    engine: Any,
    portfolio_id: uuid.UUID,
    live_graph: Any = None,
) -> None:
    """Inject engine + portfolio_id (and optionally a live graph) into the app.

    Called once at startup from the ``firm web`` CLI command.
    Also called by tests with mock objects.
    """
    global _ENGINE, _PORTFOLIO_ID, _LIVE_GRAPH
    _ENGINE = engine
    _PORTFOLIO_ID = portfolio_id
    _LIVE_GRAPH = live_graph


def _get_engine_and_portfolio_id() -> tuple[Any, uuid.UUID]:
    if _ENGINE is None or _PORTFOLIO_ID is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    return _ENGINE, _PORTFOLIO_ID


def _get_live_graph() -> Any:
    return _LIVE_GRAPH
