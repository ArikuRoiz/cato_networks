from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_dict(content: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from an LLM completion.

    Models routinely wrap their JSON in markdown fences or pad it with prose
    despite a "no fences" instruction. A strict ``json.loads`` rejects those,
    which silently turns a perfectly good response into a parse failure. This
    tries the strict path first, then strips fences, then extracts the first
    balanced ``{...}`` block before giving up.
    """
    text = content.strip()
    parsed = _try_loads(text)
    if parsed is not None:
        return parsed

    fenced = _FENCE.search(text)
    if fenced is not None:
        parsed = _try_loads(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    block = _first_object(text)
    if block is not None:
        return _try_loads(block)
    return None


def _try_loads(text: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _first_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, respecting string quoting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
