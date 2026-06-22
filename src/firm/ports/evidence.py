"""EvidenceStore port — the IO seam for RAG / news-corpus retrieval.

The no-lookahead invariant is enforced at this boundary:
``search`` must never return a chunk whose ``published_at > before``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from firm.ports.types import Chunk, NewsDoc


@runtime_checkable
class EvidenceStore(Protocol):
    """Read/write access to the news-corpus vector store.

    Implementations must honour the ``runtime_checkable`` contract so fakes
    can be verified with ``isinstance`` in tests.
    """

    def search(
        self,
        symbol: str,
        *,
        before: datetime,
        k: int = 10,
        query: str | None = None,
    ) -> list[Chunk]:
        """Return up to *k* chunks for *symbol* published at or before *before*.

        The no-lookahead rule is enforced here: ``chunk.published_at <= before``
        for every returned chunk (boundary is inclusive).

        Parameters
        ----------
        symbol:
            Ticker symbol to filter on.
        before:
            Inclusive upper bound on ``published_at``.  Chunks published at
            exactly *before* are included.
        k:
            Maximum number of chunks to return.
        query:
            Optional free-text query to direct dense retrieval.  When omitted,
            implementations may fall back to a symbol-derived default.
        """
        ...

    def embed_and_store(self, doc: NewsDoc) -> None:
        """Embed *doc* and persist it in the evidence store."""
        ...
