"""Cassette LLM adapter — record once, replay deterministically in CI.

``CassetteLLM`` wraps any ``LLM`` implementation.  In **record** mode every
completion is forwarded to the inner LLM and the request/response pair is
appended to a JSONL file on disk.  In **replay** mode no network calls are
made; responses are looked up by their SHA-256 content key.

This is the seam that makes ``make eval`` bit-reproducible and keeps CI
offline (SPEC FR-8).

Duplicate-key behaviour in record mode
---------------------------------------
If ``complete()`` is called twice with the identical (model, messages) pair in
a single record session, two JSONL lines are written for the same key.
``_load_cassette`` uses last-write-wins (dict assignment), so replay is still
correct.  The duplicate line is harmless but the file grows without bound if
the same request is replayed many times; callers are responsible for avoiding
unnecessary repeated identical requests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, LLMResponse


class CassetteNotFound(Exception):
    """Raised when a replay lookup finds no cassette entry for *key*."""

    def __init__(self, key: str) -> None:
        super().__init__(f"No cassette entry for key: {key}")
        self.key = key


class _ResponseDict(TypedDict, total=False):
    """Typed shape of the persisted response dict inside a cassette entry."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    message: str
    retryable: bool


def _compute_key(model: str, messages: list[LLMMessage]) -> str:
    """Return a stable SHA-256 hex digest for *(model, messages)*."""
    payload = json.dumps(
        {
            "model": model,
            "messages": sorted(
                (m.model_dump() for m in messages),
                key=lambda d: json.dumps(d, sort_keys=True),
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cassette(path: Path) -> dict[str, _ResponseDict]:
    """Parse a JSONL cassette file into a ``{key: response_dict}`` mapping.

    Lines that contain a ``token_count`` field (written by ``count_tokens``
    record calls) are skipped — they are consumed by ``_load_token_counts``.
    """
    entries: dict[str, _ResponseDict] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record: dict[str, object] = json.loads(line)
        if "response" not in record:
            continue
        key = str(record["key"])
        entries[key] = cast(_ResponseDict, record["response"])
    return entries


def _append_entry(path: Path, key: str, response: LLMResponse | LLMError) -> None:
    """Append a single JSONL record to *path*."""
    record = {"key": key, "response": response.model_dump()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _compute_token_key(model: str, messages: list[LLMMessage]) -> str:
    """Return a stable SHA-256 hex digest for a *count_tokens* *(model, messages)* call.

    Uses a separate namespace from ``_compute_key`` so that a ``complete()``
    entry and a ``count_tokens()`` entry for the same messages are never
    confused, even though the request shape is identical.
    """
    payload = json.dumps(
        {
            "op": "count_tokens",
            "model": model,
            "messages": sorted(
                (m.model_dump() for m in messages),
                key=lambda d: json.dumps(d, sort_keys=True),
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _append_token_count(path: Path, key: str, count: int) -> None:
    """Append a token-count JSONL record to *path*."""
    record = {"key": key, "token_count": count}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _load_token_counts(path: Path) -> dict[str, int]:
    """Parse token-count entries from a JSONL cassette into a ``{key: count}`` mapping."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record: dict[str, object] = json.loads(line)
        if "token_count" in record:
            token_count = record["token_count"]
            assert isinstance(token_count, int)
            counts[str(record["key"])] = token_count
    return counts


class CassetteLLM(LLM):
    """Record/replay wrapper around any ``LLM`` implementation.

    Parameters
    ----------
    cassette_path:
        Path to the JSONL cassette file.  Created automatically in record mode.
    mode:
        ``"record"`` — delegate to *inner* and persist responses.
        ``"replay"`` — serve responses from the cassette; never calls *inner*.
    inner:
        Required in record mode; must be a valid ``LLM`` implementation.
    """

    def __init__(
        self,
        cassette_path: Path,
        mode: Literal["record", "replay"],
        inner: LLM | None = None,
    ) -> None:
        if mode == "record" and inner is None:
            raise ValueError("CassetteLLM in record mode requires an inner LLM")
        self._cassette_path = cassette_path
        self._mode = mode
        self._inner = inner
        self._entries: dict[str, _ResponseDict] = (
            _load_cassette(cassette_path) if mode == "replay" else {}
        )
        self._token_counts: dict[str, int] = (
            _load_token_counts(cassette_path) if mode == "replay" else {}
        )

    # ------------------------------------------------------------------
    # LLM port implementation
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        """Complete *messages*, recording or replaying as configured."""
        key = _compute_key(model, messages)
        if self._mode == "record":
            return self._record(key, messages, model=model, max_tokens=max_tokens)
        return self._replay(key)

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        """Estimate token count; delegates to inner in record, uses cassette in replay.

        In record mode the result is persisted as a dedicated ``token_count``
        JSONL entry (keyed separately from ``complete()`` entries).

        In replay mode the entry is looked up by its own key; ``CassetteNotFound``
        is raised if absent.
        """
        key = _compute_token_key(model, messages)
        if self._mode == "record":
            count = self._require_inner().count_tokens(messages, model=model)
            _append_token_count(self._cassette_path, key, count)
            return count
        cached: int | None = self._token_counts.get(key)
        if cached is None:
            raise CassetteNotFound(key)
        return cached

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_inner(self) -> LLM:
        """Return *inner*, raising ``RuntimeError`` if it is absent.

        This guard should never fire at runtime because ``__init__`` rejects
        record-mode construction without an inner LLM.  It exists solely to
        satisfy mypy --strict, which cannot prove ``self._inner`` is non-None
        after the constructor check.
        """
        if self._inner is None:
            raise RuntimeError(
                "CassetteLLM in record mode has no inner LLM; "
                "this should have been rejected at construction time"
            )
        return self._inner

    def _record(
        self,
        key: str,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        result = self._require_inner().complete(messages, model=model, max_tokens=max_tokens)
        _append_entry(self._cassette_path, key, result)
        return result

    def _replay(self, key: str) -> LLMResponse | LLMError:
        entry = self._entries.get(key)
        if entry is None:
            raise CassetteNotFound(key)
        # Discriminate on the field unique to LLMResponse; anything without
        # 'content' is an LLMError.  This is stable even if LLMError gains a
        # 'retryable'-adjacent field in the future.
        if "content" in entry:
            return LLMResponse.model_validate(entry)
        return LLMError.model_validate(entry)
