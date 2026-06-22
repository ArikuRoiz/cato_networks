"""Live Anthropic LLM adapter implementing the LLM port.

Cost routing is handled by MODEL_MAP — callers pass a short alias
("haiku", "sonnet") and the adapter resolves the full model name.
Unknown aliases fall through as-is, which lets tests pass synthetic
model strings without a mapping entry.
"""

from __future__ import annotations

from typing import cast

import anthropic
from anthropic.types import MessageParam, TextBlock

from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, LLMResponse

# Short-name → versioned model-ID routing.
# "haiku"  → cheap extraction model (~70 % of calls)
# "sonnet" → strong decision model
MODEL_MAP: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def _resolve_model(alias: str) -> str:
    """Return the canonical model ID for *alias*, or *alias* unchanged."""
    return MODEL_MAP.get(alias, alias)


def _to_api_messages(
    messages: list[LLMMessage],
) -> tuple[str | None, list[MessageParam]]:
    """Split port messages into (system_prompt, non_system_messages).

    The Anthropic API requires the system prompt at the top level, not inside
    the messages array.  Multiple system messages are joined with newlines.
    """
    system_parts: list[str] = []
    api_messages: list[MessageParam] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            api_messages.append(cast(MessageParam, {"role": msg.role, "content": msg.content}))
    system = "\n".join(system_parts) if system_parts else None
    return system, api_messages


def _parse_response(
    response: anthropic.types.Message,
    resolved: str,
) -> LLMResponse | LLMError:
    """Extract text content from an Anthropic ``Message`` into an ``LLMResponse``.

    Returns ``LLMError`` if the content list is empty or the first block is
    not a ``TextBlock`` (e.g. tool-use or image), so the bug surfaces rather
    than being silently swallowed.
    """
    if not response.content:
        return LLMError(
            message="Anthropic returned empty content list",
            retryable=False,
        )
    block = response.content[0]
    # SDK content blocks are a tagged union (TextBlock | ToolUseBlock | …).
    # Runtime narrowing via isinstance is unavoidable here — this is not a
    # defensive check but a discriminated-union dispatch mandated by the SDK.
    if not isinstance(block, TextBlock):
        return LLMError(
            message=f"Unexpected content block type: {type(block).__name__}",
            retryable=False,
        )
    return LLMResponse(
        content=block.text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=resolved,
    )


class AnthropicLLM(LLM):
    """Live Anthropic adapter implementing the ``LLM`` port.

    Constructed with an *api_key*; the underlying ``anthropic.Anthropic``
    client is created once and reused across calls.

    Failure modes are represented as ``LLMError`` values — callers inspect
    the result union rather than catching exceptions.
    """

    def __init__(self, api_key: str) -> None:
        self._client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        """Send *messages* to the Anthropic API and return the result.

        Returns ``LLMResponse`` on success.
        Returns ``LLMError`` for API status errors (4xx/5xx), connection
        failures, and empty/unexpected content — never raises for expected paths.
        """
        resolved = _resolve_model(model)
        try:
            system, api_messages = _to_api_messages(messages)
            if system:
                response = self._client.messages.create(
                    model=resolved,
                    max_tokens=max_tokens,
                    messages=api_messages,
                    system=system,
                )
            else:
                response = self._client.messages.create(
                    model=resolved,
                    max_tokens=max_tokens,
                    messages=api_messages,
                )
            return _parse_response(response, resolved)
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code in {429, 500, 502, 503}
            return LLMError(message=str(exc), retryable=retryable)
        except anthropic.APIConnectionError as exc:
            return LLMError(message=str(exc), retryable=True)

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        """Return the token count for *messages* via the Anthropic token-counting API.

        Raises ``RuntimeError`` for API status or connection errors so that
        callers receive a typed, descriptive failure rather than a raw SDK
        exception.  The same error classes handled by ``complete()`` are caught
        here for consistency.
        """
        resolved = _resolve_model(model)
        try:
            system, api_messages = _to_api_messages(messages)
            if system:
                response = self._client.messages.count_tokens(
                    model=resolved,
                    messages=api_messages,
                    system=system,
                )
            else:
                response = self._client.messages.count_tokens(
                    model=resolved,
                    messages=api_messages,
                )
            return response.input_tokens
        except anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic token-count failed (HTTP {exc.status_code}): {exc}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Anthropic token-count connection error: {exc}") from exc
