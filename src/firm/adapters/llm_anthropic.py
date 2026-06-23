"""Live Anthropic LLM adapter implementing the LLM port.

Cost routing is handled by MODEL_MAP — callers pass an ``LLMModel`` alias
(``HAIKU`` / ``SONNET``) and the adapter resolves the full model name.
Unknown aliases fall through as-is, which lets tests pass synthetic
model strings without a mapping entry.
"""

from __future__ import annotations

import logging
from typing import cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUseBlock

from firm.domain.enums import LLMModel
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, LLMResponse, ToolDef, ToolExecutors

logger = logging.getLogger(__name__)

# Short alias → versioned model-ID routing.
# HAIKU  → cheap extraction model (~70 % of calls)
# SONNET → strong decision model
MODEL_MAP: dict[str, str] = {
    LLMModel.HAIKU: "claude-haiku-4-5-20251001",
    LLMModel.SONNET: "claude-sonnet-4-6",
}


def _resolve_model(alias: LLMModel | str) -> str:
    """Return the canonical model ID for *alias*, or *alias* unchanged."""
    return MODEL_MAP.get(alias, str(alias))


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
        model: LLMModel | str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        resolved = _resolve_model(model)
        try:
            from langfuse.decorators import langfuse_context, observe

            @observe(as_type="generation", name=f"llm.{resolved}")  # type: ignore[untyped-decorator]
            def _traced() -> LLMResponse | LLMError:
                langfuse_context.update_current_observation(
                    model=resolved,
                    input=[{"role": m.role, "content": m.content} for m in messages],
                    model_parameters={"max_tokens": max_tokens},
                )
                result = self._call_api(messages, resolved=resolved, max_tokens=max_tokens)
                if isinstance(result, LLMResponse):
                    langfuse_context.update_current_observation(
                        output=result.content,
                        usage={
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "unit": "TOKENS",
                        },
                    )
                else:
                    langfuse_context.update_current_observation(
                        level="ERROR",
                        status_message=result.message,
                    )
                return result

            return _traced()  # type: ignore[no-any-return]
        except ImportError as exc:
            logger.warning("Langfuse tracing unavailable, running without: %s", exc)
            return self._call_api(messages, resolved=resolved, max_tokens=max_tokens)

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDef],
        executors: ToolExecutors,
        *,
        model: LLMModel | str,
        max_tokens: int,
        max_rounds: int = 5,
    ) -> LLMResponse | LLMError:
        resolved = _resolve_model(model)
        system, api_messages = _to_api_messages(messages)
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        total_input = total_output = 0

        for _ in range(max_rounds):
            try:
                if system:
                    response = self._client.messages.create(
                        model=resolved,
                        max_tokens=max_tokens,
                        messages=api_messages,
                        tools=api_tools,  # type: ignore[arg-type]
                        system=system,
                    )
                else:
                    response = self._client.messages.create(
                        model=resolved,
                        max_tokens=max_tokens,
                        messages=api_messages,
                        tools=api_tools,  # type: ignore[arg-type]
                    )
            except anthropic.APIStatusError as exc:
                return LLMError(message=str(exc), retryable=exc.status_code in {429, 500, 502, 503})
            except anthropic.APIConnectionError as exc:
                return LLMError(message=str(exc), retryable=True)

            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if isinstance(block, TextBlock):
                        return LLMResponse(
                            content=block.text,
                            input_tokens=total_input,
                            output_tokens=total_output,
                            model=resolved,
                        )
                return LLMError(
                    message="No text block in final tool-loop response", retryable=False
                )

            if response.stop_reason != "tool_use":
                return LLMError(
                    message=f"Unexpected stop_reason: {response.stop_reason}", retryable=False
                )

            tool_results = []
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    executor = executors.get(block.name)
                    if executor is None:
                        result = f"Error: unknown tool '{block.name}'"
                    else:
                        try:
                            result = executor(dict(block.input))
                        except Exception:
                            logger.exception(
                                "Tool execution failed: %s(%r)", block.name, block.input
                            )
                            result = f"Error executing {block.name}"
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )

            api_messages = [
                *api_messages,
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},  # type: ignore[typeddict-item]
            ]

        return LLMError(message=f"Tool loop exceeded max_rounds={max_rounds}", retryable=False)

    def _call_api(
        self,
        messages: list[LLMMessage],
        *,
        resolved: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
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
        model: LLMModel | str,
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
