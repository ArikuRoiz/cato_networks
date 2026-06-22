from __future__ import annotations

import json
from typing import Any


def parse_json_dict(content: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(content.strip())
        return raw if isinstance(raw, dict) else None
    except json.JSONDecodeError:
        return None
