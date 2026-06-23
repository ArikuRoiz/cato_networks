"""Shared CLI utilities: NDJSON emit, summarise, dotenv, settings, paths.

These helpers are imported by every command module. They are deliberately
light — no heavy (Anthropic/torch/sqlalchemy/langgraph) imports happen at
module load time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firm.config.settings import Settings


def _project_root() -> Path:
    """Return the project root directory (four levels above this file)."""
    return Path(__file__).parent.parent.parent.parent


def _load_dotenv() -> None:
    """Load .env from the project root into os.environ if not already set."""
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _emit(record: dict[str, Any]) -> None:
    """Print a single JSON record to stdout followed by a newline."""
    print(json.dumps(record), flush=True)


def _summarise(value: Any) -> str:
    """Return a short human-readable summary of an agent result."""
    if value is None:
        return "none"
    if isinstance(value, dict):
        keys = list(value.keys())[:4]
        return f"dict({', '.join(keys)})"
    return type(value).__name__


def _load_settings() -> Settings:
    """Load application settings from the environment."""
    from firm.config.settings import load_settings  # deferred: heavy import

    return load_settings()
