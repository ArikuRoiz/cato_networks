"""Application settings and RiskPolicy loading — single authoritative source of truth."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Project-root anchor (robust regardless of caller's cwd)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
_DEFAULT_RISK_POLICY_PATH: Path = _PROJECT_ROOT / "config" / "risk_policy.yaml"


# ---------------------------------------------------------------------------
# RiskPolicy
# ---------------------------------------------------------------------------


def _assert_fraction(name: str, value: float) -> float:
    """Return *value* unchanged; raise ValueError when not in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]; got {value!r}")
    return value


def _assert_positive(name: str, value: float) -> float:
    """Return *value* unchanged; raise ValueError when not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value!r}")
    return value


class RiskPolicyConfig(BaseModel, frozen=True):
    """Typed representation of config/risk_policy.yaml.

    All fraction fields are expressed as a positive decimal (e.g. 0.10 = 10%).
    ``daily_loss_halt_pct`` is stored as a positive value; callers negate it
    when comparing against P&L (i.e. halt when daily_pnl < -daily_loss_halt_pct).

    Validation rules:
      - Fraction fields (``*_pct``, ``event_relevance_threshold``,
        ``momentum_weight``, ``sentiment_weight``) must be in [0.0, 1.0].
      - ``buy_threshold`` must be strictly positive (> 0).
      - ``sell_threshold`` must be strictly negative (< 0); a positive value
        would invert buy/sell signals.
      - ``momentum_weight + sentiment_weight`` must equal 1.0 (±1e-9).
      - Integer counts must be strictly positive.

    Fill-cost model (slippage_bps=5, commission_per_share=$0.005) is the
    authoritative single source of truth in ``firm.domain.portfolio`` as module
    constants ``_SLIPPAGE_BPS`` and ``_COMMISSION_PER_SHARE``.  These values
    are consumed directly by ``firm.persistence.ledger`` and
    ``firm.domain.portfolio.Portfolio.can_afford``.
    """

    # Core risk limits
    max_trade_notional_pct: float
    max_name_concentration_pct: float
    daily_loss_halt_pct: float
    hitl_threshold_pct: float

    # Signal thresholds
    buy_threshold: float
    sell_threshold: float

    # Signal weights (must sum to 1.0)
    momentum_weight: float
    sentiment_weight: float

    # Momentum lookback
    momentum_lookback_days: int

    # News-event qualifying parameters
    max_events_per_symbol_per_hour: int
    event_relevance_threshold: float

    # LLM cost control
    token_budget_per_cycle: int

    # Field-level validators --------------------------------------------------

    @field_validator(
        "max_trade_notional_pct",
        "max_name_concentration_pct",
        "daily_loss_halt_pct",
        "hitl_threshold_pct",
        "event_relevance_threshold",
        "momentum_weight",
        "sentiment_weight",
        mode="after",
    )
    @classmethod
    def _must_be_fraction(cls, v: float, info: Any) -> float:
        return _assert_fraction(info.field_name, v)

    @field_validator("buy_threshold", mode="after")
    @classmethod
    def _buy_threshold_positive(cls, v: float) -> float:
        return _assert_positive("buy_threshold", v)

    @field_validator("sell_threshold", mode="after")
    @classmethod
    def _sell_threshold_negative(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(
                f"sell_threshold must be strictly negative (< 0); got {v!r}. "
                "A non-negative sell_threshold would invert buy/sell signals."
            )
        return v

    @field_validator(
        "momentum_lookback_days",
        "max_events_per_symbol_per_hour",
        "token_budget_per_cycle",
        mode="after",
    )
    @classmethod
    def _must_be_positive(cls, v: float, info: Any) -> float:
        return _assert_positive(info.field_name, v)

    # Cross-field validator ---------------------------------------------------

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> RiskPolicyConfig:
        if not self.weights_sum_to_one:
            raise ValueError(
                f"momentum_weight + sentiment_weight must equal 1.0; "
                f"got {self.momentum_weight} + {self.sentiment_weight} "
                f"= {self.momentum_weight + self.sentiment_weight}"
            )
        return self

    # Derived predicates ------------------------------------------------------

    @property
    def weights_sum_to_one(self) -> bool:
        return abs(self.momentum_weight + self.sentiment_weight - 1.0) < 1e-9


def load_risk_policy(
    path: Path = _DEFAULT_RISK_POLICY_PATH,
) -> RiskPolicyConfig:
    """Read *path* and return a validated :class:`RiskPolicyConfig`.

    Raises ``FileNotFoundError`` if *path* does not exist,
    ``KeyError`` / ``TypeError`` if required fields are missing or wrong type,
    ``ValueError`` if any value is out of range or weights do not sum to 1.0.
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return RiskPolicyConfig(
        max_trade_notional_pct=float(raw["max_trade_notional_pct"]),
        max_name_concentration_pct=float(raw["max_name_concentration_pct"]),
        daily_loss_halt_pct=float(raw["daily_loss_halt_pct"]),
        hitl_threshold_pct=float(raw["hitl_threshold_pct"]),
        buy_threshold=float(raw["buy_threshold"]),
        sell_threshold=float(raw["sell_threshold"]),
        momentum_weight=float(raw["momentum_weight"]),
        sentiment_weight=float(raw["sentiment_weight"]),
        momentum_lookback_days=int(raw["momentum_lookback_days"]),
        max_events_per_symbol_per_hour=int(raw["max_events_per_symbol_per_hour"]),
        event_relevance_threshold=float(raw["event_relevance_threshold"]),
        token_budget_per_cycle=int(raw["token_budget_per_cycle"]),
    )


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables.

    All secrets (API keys, tokens) must be injected via the environment;
    this class never reads files, never has hard-coded credentials.
    """

    database_url: str
    anthropic_api_key: str
    slack_bot_token: str
    slack_channel: str
    langfuse_public_key: str
    langfuse_secret_key: str
    otel_endpoint: str

    # Convenience predicates
    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_slack(self) -> bool:
        return bool(self.slack_bot_token)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def load_settings() -> Settings:
    """Build :class:`Settings` from the current process environment.

    Provides sensible defaults for non-secret infrastructure URLs so the
    app can start with a minimal ``.env``; secrets default to empty strings
    (the caller is responsible for validating them before use via
    ``has_anthropic_key``, ``has_slack``, ``has_langfuse``).
    """
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://firm:firm@localhost:5432/firm",
        ),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        slack_bot_token=os.environ.get("SLACK_BOT_TOKEN", ""),
        slack_channel=os.environ.get("SLACK_CHANNEL", "#trading-desk"),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        otel_endpoint=os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4317",
        ),
    )
