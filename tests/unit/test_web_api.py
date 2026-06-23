"""Unit tests for the web API endpoints.

Uses FastAPI TestClient with a mock engine + mock ledger (no real DB).
All tests run in the standard unit suite (no 'integration' marker).

Coverage:
  - GET /api/portfolio   — returns cash/nav/pnl/holdings from mock data
  - GET /api/trades      — returns recent trades from mock data
  - GET /api/cycles      — returns recent cycles from mock data
  - GET /api/cycles/{id}/trace — returns audit entries from mock data
  - GET /api/approvals/pending — returns [] when no live graph
  - POST /api/approvals/{thread_id} — returns 503 when no live graph
  - POST /api/run        — returns 503 when no live graph

Tests for the schemas module:
  - RunRequest validates tickers, lookback_days
  - ApprovalRequest validates decision field
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from firm.web.app import configure, create_app
from firm.web.schemas import (
    ApprovalRequest,
    AuditEntryDTO,
    CycleDTO,
    HoldingDTO,
    PortfolioDTO,
    RunRequest,
    TradeDTO,
    compute_nav_and_pnl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """TestClient with a mock engine injected; no live graph."""
    app = create_app()
    configure(
        engine=_make_mock_engine(),
        portfolio_id=uuid.uuid4(),
        live_graph=None,
    )
    return TestClient(app, raise_server_exceptions=False)


def _make_mock_engine() -> MagicMock:
    """Return a MagicMock that quacks like a SQLAlchemy engine."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Helpers — build mock DTO factories for patch targets
# ---------------------------------------------------------------------------


def _portfolio_dto() -> PortfolioDTO:
    return PortfolioDTO(
        cash="90000.000000",
        nav="105000.000000",
        pnl="5000.000000",
        holdings=[HoldingDTO(symbol="NVDA", quantity="10.00000000", avg_cost="150.0000")],
    )


def _trade_dtos() -> list[TradeDTO]:
    return [
        TradeDTO(
            id=str(uuid.uuid4()),
            symbol="NVDA",
            side="buy",
            qty="10.00000000",
            status="FILLED",
            fill_price="150.000000",
            filled_at="2024-10-23T14:30:00+00:00",
        )
    ]


def _cycle_dtos() -> list[CycleDTO]:
    return [
        CycleDTO(
            id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            trigger_type="scheduled",
            outcome="filled",
            started_at="2024-10-23T14:00:00+00:00",
            judge_score=4,
            alignment="aligned",
        )
    ]


def _audit_dtos() -> list[AuditEntryDTO]:
    return [
        AuditEntryDTO(
            id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            actor="research_manager",
            action="research.done",
            payload={"symbol": "NVDA", "recommendation": "buy"},
            ts="2024-10-23T14:01:00+00:00",
        ),
        AuditEntryDTO(
            id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            actor="system",
            action="cycle.outcome",
            payload={"outcome": "filled"},
            ts="2024-10-23T14:05:00+00:00",
        ),
    ]


# ---------------------------------------------------------------------------
# GET /api/portfolio
# ---------------------------------------------------------------------------


def test_portfolio_returns_summary(client: TestClient) -> None:
    with patch("firm.web.app.fetch_portfolio", return_value=_portfolio_dto()):
        resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cash"] == "90000.000000"
    assert data["nav"] == "105000.000000"
    assert data["pnl"] == "5000.000000"
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["symbol"] == "NVDA"


def test_portfolio_returns_empty_on_missing(client: TestClient) -> None:
    with patch("firm.web.app.fetch_portfolio", side_effect=KeyError("not found")):
        resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cash"] == "0"
    assert data["holdings"] == []


# ---------------------------------------------------------------------------
# GET /api/trades
# ---------------------------------------------------------------------------


def test_trades_returns_list(client: TestClient) -> None:
    with patch("firm.web.app.fetch_recent_trades", return_value=_trade_dtos()):
        resp = client.get("/api/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "NVDA"
    assert data[0]["side"] == "buy"


def test_trades_returns_empty_list(client: TestClient) -> None:
    with patch("firm.web.app.fetch_recent_trades", return_value=[]):
        resp = client.get("/api/trades")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/cycles
# ---------------------------------------------------------------------------


def test_cycles_returns_list(client: TestClient) -> None:
    with patch("firm.web.app.fetch_recent_cycles", return_value=_cycle_dtos()):
        resp = client.get("/api/cycles")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["outcome"] == "filled"
    assert data[0]["trigger_type"] == "scheduled"


def test_cycles_returns_empty_list(client: TestClient) -> None:
    with patch("firm.web.app.fetch_recent_cycles", return_value=[]):
        resp = client.get("/api/cycles")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/cycles/{correlation_id}/trace
# ---------------------------------------------------------------------------


def test_cycle_trace_returns_entries(client: TestClient) -> None:
    cid = str(uuid.uuid4())
    with patch("firm.web.app.fetch_cycle_trace", return_value=_audit_dtos()):
        resp = client.get(f"/api/cycles/{cid}/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["actor"] == "research_manager"
    assert data[1]["action"] == "cycle.outcome"


def test_cycle_trace_returns_empty_for_unknown(client: TestClient) -> None:
    with patch("firm.web.app.fetch_cycle_trace", return_value=[]):
        resp = client.get("/api/cycles/nonexistent/trace")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/approvals/pending — no live graph
# ---------------------------------------------------------------------------


def test_pending_approvals_empty_without_live_graph(client: TestClient) -> None:
    resp = client.get("/api/approvals/pending")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /api/approvals/{thread_id} — no live graph
# ---------------------------------------------------------------------------


def test_approve_without_live_graph_returns_503(client: TestClient) -> None:
    resp = client.post(
        "/api/approvals/some-thread",
        json={"decision": "approve"},
    )
    assert resp.status_code == 503


def test_reject_without_live_graph_returns_503(client: TestClient) -> None:
    resp = client.post(
        "/api/approvals/some-thread",
        json={"decision": "reject"},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/run — no live graph
# ---------------------------------------------------------------------------


def test_run_without_live_graph_returns_503(client: TestClient) -> None:
    resp = client.post(
        "/api/run",
        json={"tickers": ["AAPL", "MSFT"], "lookback_days": 7},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/approvals — with a mock live graph
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_live_graph() -> TestClient:
    """TestClient with a mock live graph injected for approval tests."""
    app = create_app()
    configure(
        engine=_make_mock_engine(),
        portfolio_id=uuid.uuid4(),
        live_graph=_make_mock_live_graph(),
    )
    return TestClient(app, raise_server_exceptions=False)


def _make_mock_live_graph() -> MagicMock:
    mock = MagicMock()
    mock.graph = MagicMock()
    mock.graph.stream = MagicMock(
        return_value=[{"cycle_outcome": "filled", "hitl_status": "approved"}]
    )
    mock.checkpointer = MagicMock()
    return mock


def test_approve_with_live_graph_returns_result(client_with_live_graph: TestClient) -> None:
    thread_id = str(uuid.uuid4())
    with patch("firm.web.runtime.resume_approval") as mock_resume:
        mock_resume.return_value = {
            "thread_id": thread_id,
            "hitl_status": "approved",
            "outcome": "filled",
        }
        resp = client_with_live_graph.post(
            f"/api/approvals/{thread_id}",
            json={"decision": "approve"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hitl_status"] == "approved"
    assert data["outcome"] == "filled"


def test_reject_with_live_graph_records_rejection(client_with_live_graph: TestClient) -> None:
    thread_id = str(uuid.uuid4())
    with patch("firm.web.runtime.resume_approval") as mock_resume:
        mock_resume.return_value = {
            "thread_id": thread_id,
            "hitl_status": "rejected",
            "outcome": "rejected",
        }
        resp = client_with_live_graph.post(
            f"/api/approvals/{thread_id}",
            json={"decision": "reject"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hitl_status"] == "rejected"


def test_run_with_live_graph_returns_thread_ids(client_with_live_graph: TestClient) -> None:
    resp = client_with_live_graph.post(
        "/api/run",
        json={"tickers": ["AAPL", "MSFT"], "lookback_days": 7},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert len(data["thread_ids"]) == 2
    assert data["tickers"] == ["AAPL", "MSFT"]
    assert data["lookback_days"] == 7


def test_run_normalises_tickers(client_with_live_graph: TestClient) -> None:
    resp = client_with_live_graph.post(
        "/api/run",
        json={"tickers": ["aapl", " msft "], "lookback_days": 3},
    )
    assert resp.status_code == 202
    assert resp.json()["tickers"] == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


def test_run_request_rejects_empty_tickers() -> None:
    with pytest.raises(ValidationError, match="tickers"):
        RunRequest(tickers=[], lookback_days=7)


def test_run_request_rejects_zero_lookback() -> None:
    with pytest.raises(ValidationError, match="lookback_days"):
        RunRequest(tickers=["AAPL"], lookback_days=0)


def test_run_request_normalises_tickers() -> None:
    req = RunRequest(tickers=["aapl", " msft "], lookback_days=5)
    assert req.tickers == ["AAPL", "MSFT"]


def test_approval_request_rejects_invalid_decision() -> None:
    with pytest.raises(ValidationError, match="decision"):
        ApprovalRequest(decision="maybe")


def test_approval_request_is_approve_predicate() -> None:
    approve = ApprovalRequest(decision="approve")
    reject = ApprovalRequest(decision="reject")
    assert approve.is_approve is True
    assert reject.is_approve is False


def test_approval_request_accepts_edited_qty() -> None:
    req = ApprovalRequest(decision="approve", edited_qty=Decimal("15.5"))
    assert req.edited_qty == Decimal("15.5")


# ---------------------------------------------------------------------------
# compute_nav_and_pnl
# ---------------------------------------------------------------------------


def test_nav_pnl_with_no_holdings() -> None:
    nav, pnl = compute_nav_and_pnl(
        cash=Decimal("100000"),
        holdings=[],
        initial_cash=Decimal("100000"),
    )
    assert nav == Decimal("100000")
    assert pnl == Decimal("0")


def test_nav_pnl_with_holdings() -> None:
    nav, pnl = compute_nav_and_pnl(
        cash=Decimal("90000"),
        holdings=[("NVDA", Decimal("10"), Decimal("150"))],
        initial_cash=Decimal("100000"),
    )
    # equity = 10 * 150 = 1500; nav = 91500; pnl = -8500
    assert nav == Decimal("91500")
    assert pnl == Decimal("-8500")


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------


def test_dashboard_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI Investment Firm" in resp.text
    assert "portfolio" in resp.text.lower()
