"""Local sentence-embedding adapter for the firm's RAG stack.

Wraps ``sentence-transformers`` with a lazy-loaded model so that importing
this module is cheap — the model weights are only pulled from disk (or
downloaded on first use) when :meth:`SentenceTransformerEmbedder.embed` is
called for the first time.

Model: ``all-MiniLM-L6-v2`` (384-dim, ~22 M params, Apache-2 licence).
Output dimension: 384.
Deterministic: identical inputs always produce identical vectors.
"""

from __future__ import annotations

_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384


class SentenceTransformerEmbedder:
    """Thin, lazy-loading wrapper around a ``SentenceTransformer`` model.

    The model is instantiated on the first call to :meth:`embed`; subsequent
    calls reuse the same in-process instance.  Thread-safety is delegated to
    the underlying ``sentence-transformers`` library, which is safe for
    concurrent reads after the model is loaded.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``all-MiniLM-L6-v2``.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return a unit-normalised embedding vector for *text*.

        Parameters
        ----------
        text:
            The string to embed.  May be a sentence, paragraph, or short
            document chunk.

        Returns
        -------
        list[float]
            A 384-dimensional float vector (for the default model).
        """
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):  # type: ignore[return]
        """Lazy-initialise and return the underlying SentenceTransformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._model_name)
        return self._model
