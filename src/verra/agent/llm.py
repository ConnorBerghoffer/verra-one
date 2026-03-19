"""LiteLLM wrapper with streaming and tool-calling support."""


from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterator

# Suppress LiteLLM's verbose debug output unless the caller explicitly wants it.
os.environ.setdefault("LITELLM_LOG", "ERROR")


# Tool definitions

RETRIEVAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the ingested business data. Use this to find emails, documents, "
            "contracts, meeting notes, and other business information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["email", "contract", "policy", "meeting", "invoice", "any"],
                    "description": "Filter by document type",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Filter by a specific person, company, or project name",
                },
            },
            "required": ["query"],
        },
    },
}


# LLM Client


class LLMClient:
    """Thin wrapper around LiteLLM for completions, streaming, and tool calling."""

    def __init__(
        self,
        model: str = "ollama/llama3.2",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra = kwargs

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Return a full completion string (non-streaming)."""
        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._extra,
            )
        except litellm.AuthenticationError as exc:
            raise RuntimeError(
                "API key not configured or invalid. "
                "Set ANTHROPIC_API_KEY (or the relevant provider key) "
                "or run 'verra setup'."
            ) from exc
        except litellm.APIConnectionError as exc:
            raise RuntimeError(
                "Cannot reach the LLM API. Check your internet connection "
                "and verify the model endpoint is reachable."
            ) from exc
        return response.choices[0].message.content or ""  # type: ignore[attr-defined]

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Yield response text chunks as they stream in."""
        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self._extra,
            )
        except litellm.AuthenticationError as exc:
            raise RuntimeError(
                "API key not configured or invalid. "
                "Set ANTHROPIC_API_KEY (or the relevant provider key) "
                "or run 'verra setup'."
            ) from exc
        except litellm.APIConnectionError as exc:
            raise RuntimeError(
                "Cannot reach the LLM API. Check your internet connection "
                "and verify the model endpoint is reachable."
            ) from exc
        for chunk in response:
            delta = chunk.choices[0].delta  # type: ignore[attr-defined]
            content = getattr(delta, "content", None)
            if content:
                yield content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handler: Callable[[str, dict[str, Any]], str],
        max_rounds: int = 3,
    ) -> str:
        """Call LLM with tool-calling support. Returns final text response.

        The LLM can call tools up to max_rounds times before producing
        a final text response. Each tool call result is fed back into
        the conversation.

        Parameters
        ----------
        messages:
            Initial message list (system + history + user question).
        tools:
            List of tool definitions in OpenAI function-calling format.
        tool_handler:
            Callable(tool_name, tool_args) -> result_string.
            Called for each tool invocation the LLM makes.
        max_rounds:
            Maximum number of tool-call rounds before forcing a final answer.

        Returns
        -------
        The LLM's final text response (after all tool calls are resolved).
        """
        import litellm

        current_messages = list(messages)

        for _round in range(max_rounds):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=current_messages,
                    tools=tools,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **self._extra,
                )
            except litellm.AuthenticationError as exc:
                raise RuntimeError(
                    "API key not configured or invalid. "
                    "Set ANTHROPIC_API_KEY (or the relevant provider key) "
                    "or run 'verra setup'."
                ) from exc
            except litellm.APIConnectionError as exc:
                raise RuntimeError(
                    "Cannot reach the LLM API. Check your internet connection "
                    "and verify the model endpoint is reachable."
                ) from exc

            choice = response.choices[0]  # type: ignore[attr-defined]
            message = choice.message

            # If the model produced a text response without tool calls, we're done
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return message.content or ""

            # Add the assistant's tool-call turn to the conversation
            current_messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each tool call and append results
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_args = {}

                try:
                    result = tool_handler(tool_name, tool_args)
                except Exception as exc:
                    result = f"[Tool error: {exc}]"

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # Max rounds reached — ask for a final answer without tools
        current_messages.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the search results above.",
            }
        )
        try:
            final = litellm.completion(
                model=self.model,
                messages=current_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._extra,
            )
        except litellm.AuthenticationError as exc:
            raise RuntimeError(
                "API key not configured or invalid. "
                "Set ANTHROPIC_API_KEY (or the relevant provider key) "
                "or run 'verra setup'."
            ) from exc
        except litellm.APIConnectionError as exc:
            raise RuntimeError(
                "Cannot reach the LLM API. Check your internet connection "
                "and verify the model endpoint is reachable."
            ) from exc
        return final.choices[0].message.content or ""  # type: ignore[attr-defined]

    def is_available(self) -> bool:
        """Quick liveness check — returns False if the model can't be reached."""
        try:
            self.complete([{"role": "user", "content": "ping"}])
            return True
        except Exception:
            return False
