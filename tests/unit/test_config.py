"""Unit tests for src/firm/config/settings.py.

Covers:
- load_risk_policy(): locked-decision field values, FileNotFoundError on
  missing file, ValueError on bad weights, ValueError on out-of-range limits.
- load_settings(): env-var injection, defaults, has_anthropic_key predicate.

No IO beyond reading the committed config/risk_policy.yaml fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from firm.config import (
    RiskPolicyConfig,
    Settings,
    load_risk_policy,
    load_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _write_yaml(tmp_path: Path, data: dict[object, object]) -> Path:
    p = tmp_path / "risk_policy.yaml"
    p.write_text(yaml.dump(data))
    return p


def _valid_yaml_data() -> dict[str, object]:
    return {
        "max_trade_notional_pct": 0.10,
        "max_name_concentration_pct": 0.25,
        "daily_loss_halt_pct": 0.03,
        "hitl_threshold_pct": 0.05,
        "buy_threshold": 0.15,
        "sell_threshold": -0.10,
        "momentum_weight": 0.60,
        "sentiment_weight": 0.40,
        "momentum_lookback_days": 5,
        "max_events_per_symbol_per_hour": 3,
        "event_relevance_threshold": 0.70,
        "slippage_bps": 5,
        "commission_per_share": 0.005,
        "token_budget_per_cycle": 50000,
    }


# ---------------------------------------------------------------------------
# load_risk_policy — locked-decision values
# ---------------------------------------------------------------------------


class TestLoadRiskPolicyLockedValues:
    def test_loads_without_error(self) -> None:
        policy = load_risk_policy()
        assert isinstance(policy, RiskPolicyConfig)

    def test_max_trade_notional_pct(self) -> None:
        assert load_risk_policy().max_trade_notional_pct == pytest.approx(0.10)

    def test_max_name_concentration_pct(self) -> None:
        assert load_risk_policy().max_name_concentration_pct == pytest.approx(0.25)

    def test_daily_loss_halt_pct(self) -> None:
        assert load_risk_policy().daily_loss_halt_pct == pytest.approx(0.03)

    def test_hitl_threshold_pct(self) -> None:
        assert load_risk_policy().hitl_threshold_pct == pytest.approx(0.05)

    def test_buy_threshold(self) -> None:
        assert load_risk_policy().buy_threshold == pytest.approx(0.15)

    def test_sell_threshold(self) -> None:
        assert load_risk_policy().sell_threshold == pytest.approx(-0.10)

    def test_momentum_weight(self) -> None:
        assert load_risk_policy().momentum_weight == pytest.approx(0.60)

    def test_sentiment_weight(self) -> None:
        assert load_risk_policy().sentiment_weight == pytest.approx(0.40)

    def test_weights_sum_to_one(self) -> None:
        assert load_risk_policy().weights_sum_to_one is True

    def test_momentum_lookback_days(self) -> None:
        assert load_risk_policy().momentum_lookback_days == 5

    def test_max_events_per_symbol_per_hour(self) -> None:
        assert load_risk_policy().max_events_per_symbol_per_hour == 3

    def test_event_relevance_threshold(self) -> None:
        assert load_risk_policy().event_relevance_threshold == pytest.approx(0.70)

    def test_slippage_bps(self) -> None:
        assert load_risk_policy().slippage_bps == 5

    def test_commission_per_share(self) -> None:
        assert load_risk_policy().commission_per_share == pytest.approx(0.005)

    def test_token_budget_per_cycle(self) -> None:
        assert load_risk_policy().token_budget_per_cycle == 50_000


# ---------------------------------------------------------------------------
# load_risk_policy — error handling
# ---------------------------------------------------------------------------


def test_missing_yaml_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_risk_policy(tmp_path / "does_not_exist.yaml")


def test_bad_weights_raises_value_error(tmp_path: Path) -> None:
    data = _valid_yaml_data()
    data["momentum_weight"] = 0.70
    data["sentiment_weight"] = 0.40  # sum = 1.10
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match=r"must equal 1\.0"):
        load_risk_policy(path)


def test_negative_slippage_raises_value_error(tmp_path: Path) -> None:
    data = _valid_yaml_data()
    data["slippage_bps"] = -1
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="slippage_bps"):
        load_risk_policy(path)


def test_out_of_range_trade_pct_raises_value_error(tmp_path: Path) -> None:
    data = _valid_yaml_data()
    data["max_trade_notional_pct"] = 1.50  # > 1.0
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="max_trade_notional_pct"):
        load_risk_policy(path)


def test_out_of_range_relevance_threshold_raises_value_error(tmp_path: Path) -> None:
    data = _valid_yaml_data()
    data["event_relevance_threshold"] = 1.10  # > 1.0
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="event_relevance_threshold"):
        load_risk_policy(path)


# ---------------------------------------------------------------------------
# load_settings — env-var injection and defaults
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_defaults_load_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "DATABASE_URL",
            "ANTHROPIC_API_KEY",
            "SLACK_BOT_TOKEN",
            "SLACK_CHANNEL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)
        settings = load_settings()
        assert isinstance(settings, Settings)

    def test_anthropic_key_injected_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-abc123")
        settings = load_settings()
        assert settings.anthropic_api_key == "sk-test-abc123"
        assert settings.has_anthropic_key is True

    def test_has_anthropic_key_false_when_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = load_settings()
        assert settings.anthropic_api_key == ""
        assert settings.has_anthropic_key is False

    def test_has_slack_false_when_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        settings = load_settings()
        assert settings.has_slack is False

    def test_has_slack_true_when_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        settings = load_settings()
        assert settings.has_slack is True

    def test_database_url_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        settings = load_settings()
        assert settings.database_url == "postgresql://firm:firm@localhost:5432/firm"

    def test_database_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@myhost:5432/mydb")
        settings = load_settings()
        assert settings.database_url == "postgresql://user:pass@myhost:5432/mydb"
