"""Citation helpers — turn retrieved chunks into citable evidence.

``cite`` is the single public function.  It either returns the chunks (each
already carrying ``source_url`` and ``chunk_id``) or an ``Insufficient``
sentinel when no evidence is available.

Using a NamedTuple for ``Insufficient`` keeps the type cheap to construct and
easy to pattern-match without importing Pydantic in test helpers.
"""

from __future__ import annotations

from typing import NamedTuple

from firm.ports.types import Chunk


class Insufficient(NamedTuple):
    """Sentinel returned when the evidence store has no relevant chunks.

    Callers should propagate this as a ``Refusal`` rather than fabricating
    claims — this is the grounding invariant from the SPEC.
    """

    reason: str = "no_relevant_chunks"


def cite(chunks: list[Chunk]) -> list[Chunk] | Insufficient:
    """Return *chunks* as-is, or ``Insufficient`` when the list is empty.

    Each ``Chunk`` already carries ``source_url`` and ``chunk_id`` so callers
    can build citations directly from the returned list without further
    transformation.

    Relevance gating (LOCKED DECISION: qualifying event score > 0.7) is the
    caller's responsibility.  Agents must filter on ``chunk.is_relevant``
    before passing chunks here, or discard the result if all returned chunks
    fail the threshold.  ``cite`` is a pass-through that only guards against
    an empty evidence set.
    """
    if not chunks:
        return Insufficient()
    return chunks
