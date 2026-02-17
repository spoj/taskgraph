"""OpenRouter API client with reasoning_effort and cache control support.

Features:
- Connection pooling (single httpx.AsyncClient per OpenRouterClient)
- Cache control blocks only for Anthropic models
- Retry with exponential backoff, respects Retry-After for 429
- Usage tracking

Example usage:

    async with OpenRouterClient() as client:
        call_model = create_model_callable(client, "anthropic/claude-opus-4.5", tools)
        message, usage = await call_model(messages)
"""

import os
import json
import asyncio
import random
import sys
import logging
from typing import Any, Awaitable, Callable

import httpx

DEFAULT_MODEL = "openai/gpt-5.2"
MAX_ERROR_DETAIL_CHARS = (
    500  # Truncation limit for error details in log/exception messages
)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

log = logging.getLogger(__name__)


def get_headers() -> dict[str, str]:
    """Get API request headers."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/taskgraph",
    }


def add_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add cache_control breakpoint to the last message's content block.

    Always placed on the very last message so the entire conversation prefix
    is cached across iterations. The bulk of the payload is in tool results,
    so we must include those — not just user/system messages.
    """
    messages = [m.copy() for m in messages]  # Don't mutate original

    # Walk backwards to find the last message with content
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content")
        if content is None:
            continue

        if isinstance(content, str):
            messages[i]["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list):
            # Add cache_control to last text block
            for j in range(len(content) - 1, -1, -1):
                if content[j].get("type") == "text":
                    content[j]["cache_control"] = {"type": "ephemeral"}
                    break
        break

    return messages


def _is_anthropic_model(model: str) -> bool:
    """Check if the model is an Anthropic model (cache_control is Anthropic-specific)."""
    return "anthropic" in model.lower()


class OpenRouterClient:
    """Async client for OpenRouter API with reasoning and caching.

    Must be used as an async context manager to ensure proper connection cleanup:

        async with OpenRouterClient() as client:
            response = await client.chat(model, messages)
    """

    def __init__(
        self,
        reasoning_effort: str | None = "low",
        timeout: float = 600.0,
        max_retries: int = 8,
    ):
        self.reasoning_effort = reasoning_effort or "low"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

        # Status codes that should NOT be retried
        self.no_retry_codes = {401, 403, 404}

    async def __aenter__(self) -> "OpenRouterClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "OpenRouterClient must be used as async context manager: "
                "async with OpenRouterClient() as client: ..."
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """
        Make a chat completion request.

        Args:
            model: Model identifier (e.g., "anthropic/claude-opus-4.5")
            messages: List of message dicts
            tools: Optional list of tool definitions
            tool_choice: "auto", "none", or "required"

        Returns:
            {
                "message": assistant message dict with 'content' and 'tool_calls',
                "usage": usage dict with token counts
            }
        """
        client = self._get_client()

        # Add cache control only for Anthropic models
        if _is_anthropic_model(model):
            messages = add_cache_control(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        payload["reasoning"] = {"effort": self.reasoning_effort}

        backoff = 3.0
        max_backoff = 30.0
        attempt = 0

        while True:
            try:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=get_headers(),
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt < self.max_retries:
                    jitter = random.uniform(0, 3)
                    log.warning(
                        "[Retry %d/%d] Network error: %s: %s",
                        attempt + 1,
                        self.max_retries,
                        type(e).__name__,
                        e,
                    )
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, max_backoff)
                    attempt += 1
                    continue
                raise RuntimeError(f"API error after {self.max_retries} retries: {e}")

            response_text = response.text
            try:
                response_data = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError:
                if attempt < self.max_retries:
                    jitter = random.uniform(0, 3)
                    log.warning(
                        "[Retry %d/%d] Invalid JSON (status %d): %s",
                        attempt + 1,
                        self.max_retries,
                        response.status_code,
                        response_text[:MAX_ERROR_DETAIL_CHARS],
                    )
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, max_backoff)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"Invalid JSON response (status {response.status_code}): {response_text[:MAX_ERROR_DETAIL_CHARS]}"
                )

            # Success
            if response.status_code == 200:
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                usage = response_data.get("usage", {})

                return {
                    "message": message,
                    "usage": usage,
                }

            # Don't retry certain client errors
            if response.status_code in self.no_retry_codes:
                error_detail = response_data.get("error", {}).get(
                    "message", response_text[:MAX_ERROR_DETAIL_CHARS]
                )
                raise RuntimeError(
                    f"OpenRouter API error: {response.status_code} - {response.reason_phrase}: {error_detail}"
                )

            # Retry all other errors
            if attempt < self.max_retries:
                error_detail = response_data.get("error", {}).get(
                    "message", response_text[:MAX_ERROR_DETAIL_CHARS]
                )

                # Respect Retry-After header for 429
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = backoff
                    else:
                        wait = backoff
                else:
                    wait = backoff

                jitter = random.uniform(0, 3)
                log.warning(
                    "[Retry %d/%d] HTTP %d: %s",
                    attempt + 1,
                    self.max_retries,
                    response.status_code,
                    error_detail,
                )
                await asyncio.sleep(wait + jitter)
                backoff = min(backoff * 2, max_backoff)
                attempt += 1
                continue

            # Exhausted retries
            error_detail = response_data.get("error", {}).get(
                "message", response_text[:MAX_ERROR_DETAIL_CHARS]
            )
            raise RuntimeError(
                f"OpenRouter API error after {self.max_retries} retries: {response.status_code}: {error_detail}"
            )


def create_model_callable(
    client: OpenRouterClient,
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> Callable[[list[dict[str, Any]]], Awaitable[tuple[dict[str, Any], dict[str, Any]]]]:
    """Create a model callable for use with run_agent_loop.

    Returns an async function (messages) -> (assistant_message, usage)
    """

    async def call_model(
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )
        return response["message"], response["usage"]

    return call_model
