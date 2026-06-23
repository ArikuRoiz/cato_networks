"""Unit tests for src/firm/cli.py.

Covers:
- _build_parser(): correct subcommands registered, --trade-id required for trace.
- _safe_load_risk_policy(): returns defaults when YAML is absent, returns
  RiskPolicyConfig from a valid YAML, swallows parse errors and returns defaults.
- _embed_corpus(): returns article count; the database_url parameter is wired
  to PgvectorEvidenceStore (verified via mock, not a live DB).

No live Postgres or Anthropic calls are made in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from firm.cli import _build_parser, _parse_hitl_response, _parse_tickers, _safe_load_risk_policy
from firm.config.settings import RiskPolicyConfig

# ---------------------------------------------------------------------------
# _build_parser — argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_seed_subcommand_registered(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["seed"])
        assert args.command == "seed"

    def test_demo_subcommand_registered(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"

    def test_dev_subcommand_registered(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["dev"])
        assert args.command == "dev"

    def test_run_subcommand_registered(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_run_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.tickers is None
        assert args.lookback_days == 7

    def test_run_accepts_tickers(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--tickers", "NVDA,AAPL"])
        assert args.tickers == "NVDA,AAPL"

    def test_run_accepts_lookback_days(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--lookback-days", "14"])
        assert args.lookback_days == 14

    def test_trace_requires_trade_id(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["trace"])

    def test_trace_accepts_trade_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["trace", "--trade-id", "abc-123"])
        assert args.command == "trace"
        assert args.trade_id == "abc-123"

    def test_unknown_command_exits(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown"])

    def test_no_command_exits(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# _parse_tickers — comma-separated ticker parsing
# ---------------------------------------------------------------------------


class TestParseTickers:
    def test_none_returns_default_watchlist(self) -> None:
        from firm.cli import _DEFAULT_WATCHLIST

        result = _parse_tickers(None)
        assert result == _DEFAULT_WATCHLIST

    def test_single_ticker_uppercased(self) -> None:
        result = _parse_tickers("nvda")
        assert result == ["NVDA"]

    def test_comma_separated_tickers(self) -> None:
        result = _parse_tickers("NVDA,AAPL,MSFT")
        assert result == ["NVDA", "AAPL", "MSFT"]

    def test_strips_whitespace(self) -> None:
        result = _parse_tickers(" NVDA , AAPL ")
        assert result == ["NVDA", "AAPL"]

    def test_empty_string_returns_default(self) -> None:
        from firm.cli import _DEFAULT_WATCHLIST

        # Empty string produces empty list after filtering — fall back
        result = _parse_tickers("")
        assert result == _DEFAULT_WATCHLIST


# ---------------------------------------------------------------------------
# _parse_hitl_response — HITL console decision parsing
# ---------------------------------------------------------------------------


class TestParseHitlResponse:
    def test_approve_short(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("a")
        assert resume == "approved"
        assert status == HITLStatus.APPROVED

    def test_approve_full(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("approve")
        assert resume == "approved"
        assert status == HITLStatus.APPROVED

    def test_reject_short(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("r")
        assert resume == "rejected"
        assert status == HITLStatus.REJECTED

    def test_reject_full(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("reject")
        assert resume == "rejected"
        assert status == HITLStatus.REJECTED

    def test_edit_with_integer_qty(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("e 50")
        assert resume == "edit:50"
        assert status == HITLStatus.APPROVED

    def test_edit_with_decimal_qty(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("edit 12.5")
        assert resume == "edit:12.5"
        assert status == HITLStatus.APPROVED

    def test_unknown_input_defaults_to_reject(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("maybe")
        assert resume == "rejected"
        assert status == HITLStatus.REJECTED

    def test_empty_string_defaults_to_reject(self) -> None:
        from firm.domain.enums import HITLStatus

        resume, status = _parse_hitl_response("")
        assert resume == "rejected"
        assert status == HITLStatus.REJECTED


# ---------------------------------------------------------------------------
# _safe_load_risk_policy — fallback behaviour
# ---------------------------------------------------------------------------


class TestSafeLoadRiskPolicy:
    def test_returns_defaults_when_yaml_missing(self, tmp_path: Path) -> None:
        # tmp_path has no risk_policy.yaml
        config = _safe_load_risk_policy(tmp_path)
        assert isinstance(config, RiskPolicyConfig)

    def test_defaults_match_locked_decisions(self, tmp_path: Path) -> None:
        config = _safe_load_risk_policy(tmp_path)
        assert config.max_trade_notional_pct == pytest.approx(0.10)
        assert config.max_name_concentration_pct == pytest.approx(0.25)
        assert config.daily_loss_halt_pct == pytest.approx(0.03)
        assert config.hitl_threshold_pct == pytest.approx(0.05)

    def test_returns_policy_from_valid_yaml(self, tmp_path: Path) -> None:
        import yaml

        policy_data = {
            "max_trade_notional_pct": 0.10,
            "max_name_concentration_pct": 0.25,
            "daily_loss_halt_pct": 0.03,
            "hitl_threshold_pct": 0.05,
            "buy_threshold": 0.05,
            "sell_threshold": -0.05,
            "momentum_weight": 0.6,
            "sentiment_weight": 0.4,
            "momentum_lookback_days": 5,
            "max_events_per_symbol_per_hour": 3,
            "event_relevance_threshold": 0.7,
            "token_budget_per_cycle": 50000,
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "risk_policy.yaml").write_text(yaml.dump(policy_data))

        config = _safe_load_risk_policy(tmp_path)
        assert isinstance(config, RiskPolicyConfig)
        assert config.max_trade_notional_pct == pytest.approx(0.10)

    def test_returns_defaults_when_yaml_is_corrupt(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # Write invalid YAML that will fail validation
        (config_dir / "risk_policy.yaml").write_text("not: valid: yaml: [")

        config = _safe_load_risk_policy(tmp_path)
        assert isinstance(config, RiskPolicyConfig)
        # Should be the hardcoded defaults
        assert config.max_trade_notional_pct == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# _embed_corpus — count returned, database_url is wired (not silently dropped)
# ---------------------------------------------------------------------------


class TestEmbedCorpus:
    def _write_corpus(self, tmp_path: Path, n: int) -> Path:
        """Write a synthetic corpus.json with *n* articles to *tmp_path*."""
        articles = [
            {
                "symbol": "NVDA",
                "text": f"Article {i} about NVDA earnings.",
                "source_url": f"https://example.com/article-{i}",
                "published_at": "2024-10-23T10:00:00Z",
            }
            for i in range(n)
        ]
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(json.dumps(articles))
        return corpus_path

    def test_returns_article_count(self, tmp_path: Path) -> None:
        """_embed_corpus must return len(corpus), not always 0."""
        corpus_path = self._write_corpus(tmp_path, 5)

        mock_store = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("psycopg.connect", return_value=mock_conn),
            patch(
                "firm.adapters.evidence_pgvector.PgvectorEvidenceStore",
                return_value=mock_store,
            ),
            patch(
                "firm.orchestration.checkpointer._normalise_database_url",
                return_value="postgresql://...",
            ),
        ):
            from firm.cli import _embed_corpus

            count = _embed_corpus(corpus_path, "postgresql://firm:firm@localhost:5432/firm")

        assert count == 5

    def test_database_url_is_passed_to_psycopg(self, tmp_path: Path) -> None:
        """The database_url parameter must reach psycopg.connect — not be discarded."""
        corpus_path = self._write_corpus(tmp_path, 2)

        mock_store = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        expected_url = "postgresql://firm:s3cr3t@myhost:5432/mydb"

        with (
            patch("psycopg.connect", return_value=mock_conn) as mock_connect,
            patch(
                "firm.adapters.evidence_pgvector.PgvectorEvidenceStore",
                return_value=mock_store,
            ),
            patch(
                "firm.orchestration.checkpointer._normalise_database_url",
                side_effect=lambda url: url,
            ),
        ):
            from firm.cli import _embed_corpus

            _embed_corpus(corpus_path, expected_url)

        mock_connect.assert_called_once_with(expected_url)

    def test_embed_and_store_called_per_article(self, tmp_path: Path) -> None:
        """embed_and_store must be called once per article."""
        n = 3
        corpus_path = self._write_corpus(tmp_path, n)

        mock_store = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("psycopg.connect", return_value=mock_conn),
            patch(
                "firm.adapters.evidence_pgvector.PgvectorEvidenceStore",
                return_value=mock_store,
            ),
            patch(
                "firm.orchestration.checkpointer._normalise_database_url",
                return_value="postgresql://...",
            ),
        ):
            from firm.cli import _embed_corpus

            _embed_corpus(corpus_path, "postgresql://firm:firm@localhost:5432/firm")

        assert mock_store.embed_and_store.call_count == n
