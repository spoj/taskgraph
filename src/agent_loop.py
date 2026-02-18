"""Generic async agent loop with pluggable model calls and tool execution.

Design:
- ModelCallable: async function that takes messages and returns (assistant_message, usage)
- ToolExecutor: async function that executes a tool call and returns result string
- ValidationFn: optional function called when agent stops, returns (valid, error_msg)
- The loop continues until validation passes, max_iterations, or max_tokens reached
- Tools within the same round are executed concurrently via asyncio.gather

Example usage:

    async def call_model(messages: list[dict]) -> tuple[dict, dict]:
        # Call your LLM API
        return assistant_message, usage

    async def execute_tool(name: str, args: dict) -> str:
        if name == "run_sql":
            return json.dumps(run_sql(conn, args["query"]))
        return json.dumps({"error": f"Unknown tool: {name}"})

    def validate() -> tuple[bool, str]:
        result = validate_output(conn)
        return result["valid"], result.get("error", "")

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        validation_fn=validate,
        max_tokens=20_000_000,
    )
"""

import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# Default token budget per node (prompt + completion combined)
DEFAULT_MAX_TOKENS = 20_000_000
DEFAULT_MAX_ITERATIONS = 200


@dataclass
class AgentResult:
    """Result of an agent loop run."""

    success: bool
    final_message: str
    iterations: int
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
        }
    )
    tool_calls_count: int = 0


# Type aliases
ModelCallable = Callable[
    [list[dict[str, Any]]],
    Awaitable[tuple[dict[str, Any], dict[str, Any]]],
]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
ValidationFn = Callable[[], tuple[bool, str]]
OnIterationFn = Callable[
    [int, dict[str, Any] | None, list[dict[str, Any]] | None],
    None,
]


def accumulate_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    """Add usage from a response to the running total."""
    total["prompt_tokens"] += usage.get("prompt_tokens", 0)
    total["completion_tokens"] += usage.get("completion_tokens", 0)

    prompt_details = usage.get("prompt_tokens_details", {})
    total["cache_read_tokens"] += prompt_details.get("cached_tokens", 0)

    completion_details = usage.get("completion_tokens_details", {})
    total["reasoning_tokens"] += completion_details.get("reasoning_tokens", 0)


async def run_agent_loop(
    call_model: ModelCallable,
    tool_executor: ToolExecutor,
    initial_messages: list[dict[str, Any]],
    validation_fn: ValidationFn | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_iteration: OnIterationFn | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AgentResult:
    """
    Run an agent loop until completion or max iterations.

    Tools within the same round are executed concurrently.

    Args:
        call_model: Async function (messages) -> (assistant_message, usage)
            assistant_message should have 'content' and optionally 'tool_calls'
        tool_executor: Async function (tool_name, args) -> result_string
        initial_messages: Starting messages (system + user prompts)
        validation_fn: Optional () -> (valid, error_msg), called when agent stops
        max_iterations: Maximum iterations (0 = unlimited)
        max_tokens: Maximum total tokens (prompt + completion) before abort.
            Default 20M. Set to 0 to disable.
        on_iteration: Optional callback (iteration, assistant_msg, tool_results)

    Returns:
        AgentResult with success status, final message, usage stats
    """
    messages = list(initial_messages)
    total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
    }
    tool_calls_count = 0
    iteration = 0

    def _result(success: bool, message: str) -> AgentResult:
        return AgentResult(
            success=success,
            final_message=message,
            iterations=iteration,
            messages=messages,
            usage=total_usage,
            tool_calls_count=tool_calls_count,
        )

    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1

        # Call the model
        assistant_message, usage = await call_model(messages)
        accumulate_usage(total_usage, usage)

        # Check token budget
        if max_tokens > 0:
            total_tokens = (
                total_usage["prompt_tokens"] + total_usage["completion_tokens"]
            )
            if total_tokens > max_tokens:
                log.warning("Token limit exceeded: %d / %d", total_tokens, max_tokens)
                return _result(
                    False, f"Token limit exceeded ({total_tokens:,} / {max_tokens:,})"
                )

        # Add assistant message to history
        messages.append(assistant_message)

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            # Parse all tool calls
            parsed = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                parsed.append((tc["id"], func_name, args))

            # Execute all tools concurrently
            async def _exec(
                call_id: str, name: str, args: dict[str, Any]
            ) -> dict[str, Any]:
                result = await tool_executor(name, args)
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }

            tool_results = await asyncio.gather(
                *(_exec(cid, name, args) for cid, name, args in parsed)
            )
            tool_results = list(tool_results)
            tool_calls_count += len(tool_results)

            messages.extend(tool_results)

            if on_iteration:
                on_iteration(iteration, assistant_message, tool_results)
        else:
            # No tool calls - agent is done
            if on_iteration:
                on_iteration(iteration, assistant_message, None)

            content = assistant_message.get("content", "")

            # Run validation if provided
            if validation_fn:
                valid, error_msg = validation_fn()
                if valid:
                    return _result(True, content)
                # Validation failed - feed error back and continue
                messages.append(
                    {
                        "role": "user",
                        "content": f"VALIDATION FAILED:\n{error_msg}\n\nFix the issues and reply when done.",
                    }
                )
            else:
                return _result(True, content)

    # Max iterations reached
    return _result(False, "Max iterations reached without valid output")
