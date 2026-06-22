"""Score-fusion reranker with a strict no-lookahead filter.

The no-lookahead invariant is defined in the SPEC:
  *"retrieval never returns a doc with published_at > decision_ts"*

This module enforces that invariant as a pure function so it can be tested
independently of the database adapter.
"""

from __future__ import annotations

from datetime import datetime

from firm.ports.types import Chunk
from firm.rag.retriever import combine_scores, extract_query_terms


def apply_no_lookahead(chunks: list[Chunk], before: datetime) -> list[Chunk]:
    """Remove any chunk whose ``published_at`` is strictly after *before*.

    The filter is applied with ``<=`` semantics: a chunk published at exactly
    *before* is retained (the boundary is inclusive).
    """
    return [c for c in chunks if c.published_at <= before]


def rerank(
    chunks: list[Chunk],
    query: str,
    *,
    before: datetime,
    k: int,
) -> list[Chunk]:
    """Apply no-lookahead filter, compute combined scores, and return top *k*.

    Steps:
    1. ``apply_no_lookahead`` — drop future documents.
    2. ``combine_scores`` — add keyword boost to each chunk's dense score.
    3. Sort descending by combined score.
    4. Return the first *k* results.
    """
    safe = apply_no_lookahead(chunks, before)
    query_terms = extract_query_terms(query)
    scored = [
        chunk.model_copy(update={"score": combine_scores(chunk.score, chunk.text, query_terms)})
        for chunk in safe
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]
