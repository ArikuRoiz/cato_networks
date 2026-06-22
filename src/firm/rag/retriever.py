"""Keyword-boost scoring helpers for the hybrid retrieval pipeline.

Dense retrieval (vector cosine similarity via the pgvector ``<=>`` operator)
lives in ``firm.adapters.evidence_pgvector._fetch_raw_rows``.  This module
supplies the pure, IO-free helpers that are composed on top of those raw
dense scores:

* ``keyword_boost`` — per-term binary bonus added to the dense score so that
  chunks containing the query terms rank higher when embedding distances are
  otherwise similar.
* ``combine_scores`` — combines a dense similarity score with keyword boost.
* ``extract_query_terms`` — tokenises a free-text query for use with the
  above.

All three functions are pure (no IO) and can be unit-tested without a
database.

Note: dense retrieval itself is not defined in this module.  It lives in
``firm.adapters.evidence_pgvector._fetch_raw_rows`` so that SQL and
pgvector coupling remains in the adapter layer.
"""

from __future__ import annotations

import re


def keyword_boost(text: str, query_terms: list[str]) -> float:
    """Return a score bonus based on how many *query_terms* appear in *text*.

    Each distinct term that appears at least once contributes 0.1 to the score,
    regardless of frequency (binary bag-of-words).  Terms are matched
    case-insensitively as whole words via a simple regex.
    """
    if not query_terms:
        return 0.0
    normalised = text.lower()
    return sum(
        0.1
        for term in query_terms
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", normalised)
    )


def combine_scores(
    dense_score: float,
    text: str,
    query_terms: list[str],
) -> float:
    """Return the combined retrieval score: ``dense_score + keyword_boost``.

    The dense score from pgvector is a *distance* (lower = more similar).
    Callers must convert it to a similarity before passing here, e.g.:
    ``similarity = 1.0 - cosine_distance``.
    """
    return dense_score + keyword_boost(text, query_terms)


def extract_query_terms(query: str) -> list[str]:
    """Tokenise *query* into lowercase alpha-numeric terms for keyword boosting.

    Stop-words are not removed; the simple approach is sufficient at this scale.
    """
    return [tok for tok in re.split(r"\W+", query.lower()) if len(tok) > 2]
