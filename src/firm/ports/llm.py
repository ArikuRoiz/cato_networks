"""LLM port — the IO seam for language-model completions.

Agents import this Protocol; adapters (Anthropic live, cassette replay, fake)
implement it.  The LLM MUST NOT emit prices, quantities, P&L, or dates —
those come from domain tools and market-data calls.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from firm.ports.types import LLMError, LLMMessage, LLMResponse


@runtime_checkable
class LLM(Protocol):
    """Language-model completion interface.

    Implementations must honour the ``runtime_checkable`` contract so fakes
    can be verified with ``isinstance`` in tests.
    """

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        """Send *messages* to the model and return the response.

        Returns ``LLMResponse`` on success or ``LLMError`` on failure.
        Never raises for expected failure modes — callers use the result union.
        """
        ...

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        """Estimate the token count for *messages* without sending a completion."""
        ...
