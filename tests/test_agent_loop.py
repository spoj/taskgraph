import asyncio
import json

from src.agent_loop import run_agent_loop, AgentResult, DEFAULT_MAX_TOKENS


class TestTokenCircuitBreaker:
    """Tests for the token budget enforcement in run_agent_loop()."""

    def test_default_max_tokens(self):
        """Default max tokens is 20M."""
        assert DEFAULT_MAX_TOKENS == 20_000_000

    def test_budget_exceeded_stops_agent(self):
        """Agent stops with success=False when token budget is exceeded."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 600_000, "completion_tokens": 500_000},
            )

        async def mock_tool_executor(name, args):
            return json.dumps({"result": "ok"})

        def mock_validation():
            return False, "not valid yet"  # Always fail to keep loop going

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                validation_fn=mock_validation,
                max_tokens=1_000_000,  # 1M budget
                max_iterations=100,
            )
        )
        assert not result.success
        assert "token limit" in result.final_message.lower()
        # With 1.1M tokens per call and 1M budget, should stop after 1 call
        assert call_count == 1

    def test_budget_not_exceeded_continues(self):
        """Agent continues normally when under budget."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            # First call: use tools. Second call: stop.
            if call_count == 1:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    {"prompt_tokens": 100, "completion_tokens": 50},
                )
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 100, "completion_tokens": 50},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=1_000_000,
                max_iterations=10,
            )
        )
        assert result.success
        assert call_count == 2

    def test_budget_disabled_when_zero(self):
        """Setting max_tokens=0 disables the token budget check."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"tc{call_count}",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000},
                )
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=0,  # disabled
                max_iterations=10,
            )
        )
        assert result.success  # Completes despite enormous token usage
        assert result.usage["prompt_tokens"] == 30_000_000

    def test_usage_accumulation(self):
        """Token usage is correctly accumulated across iterations."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"tc{call_count}",
                                "function": {"name": "t", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "prompt_tokens_details": {"cached_tokens": 200},
                        "completion_tokens_details": {"reasoning_tokens": 100},
                    },
                )
            return (
                {"role": "assistant", "content": "done"},
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "prompt_tokens_details": {"cached_tokens": 200},
                    "completion_tokens_details": {"reasoning_tokens": 100},
                },
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=0,
                max_iterations=10,
            )
        )
        assert result.usage["prompt_tokens"] == 4000
        assert result.usage["completion_tokens"] == 2000
        assert result.usage["cache_read_tokens"] == 800
        assert result.usage["reasoning_tokens"] == 400

    def test_max_iterations_exceeded(self):
        """Agent stops with success=False when max iterations exceeded."""

        async def mock_call_model(messages):
            return (
                {"role": "assistant", "content": "thinking..."},
                {"prompt_tokens": 100, "completion_tokens": 50},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        def mock_validation():
            return False, "still not valid"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                validation_fn=mock_validation,
                max_iterations=3,
                max_tokens=0,
            )
        )
        assert not result.success
        assert result.iterations == 3
        assert "max iterations" in result.final_message.lower()




class TestAgentLoopConcurrentTools:
    """Tests for run_agent_loop: concurrent tool execution."""

    def test_multiple_tools_executed_concurrently(self):
        """Multiple tool calls in one round are all executed."""
        call_count = 0

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return two tool calls
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "tool_a",
                                "arguments": '{"x": 1}',
                            },
                        },
                        {
                            "id": "tc2",
                            "function": {
                                "name": "tool_b",
                                "arguments": '{"y": 2}',
                            },
                        },
                    ],
                }, {"prompt_tokens": 100, "completion_tokens": 50}
            else:
                # Second call: done
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 100, "completion_tokens": 50}

        tool_calls_received = []

        async def mock_executor(name, args):
            tool_calls_received.append((name, args))
            return json.dumps({"result": f"ok from {name}"})

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )

        assert result.success is True
        assert result.tool_calls_count == 2
        assert len(tool_calls_received) == 2
        names = {tc[0] for tc in tool_calls_received}
        assert names == {"tool_a", "tool_b"}

    def test_tool_results_added_to_messages(self):
        """Tool results are appended as tool messages with correct IDs."""
        call_count = 0

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_abc",
                            "function": {
                                "name": "run_sql",
                                "arguments": '{"query": "SELECT 1"}',
                            },
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                # Verify tool result was in messages
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                assert len(tool_msgs) == 1
                assert tool_msgs[0]["tool_call_id"] == "tc_abc"
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            return '{"success": true}'

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )
        assert result.success is True


class TestAgentLoopJSONDecodeError:
    """Tests for agent_loop: malformed tool call arguments."""

    def test_malformed_json_args_fallback_to_empty(self):
        """Malformed JSON in tool arguments falls back to empty dict."""
        call_count = 0
        received_args = []

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "run_sql",
                                "arguments": "NOT VALID JSON {{{",
                            },
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            received_args.append(args)
            return '{"ok": true}'

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )

        assert result.success is True
        assert received_args[0] == {}


class TestAgentLoopOnIteration:
    """Tests for agent_loop: on_iteration callback."""

    def test_on_iteration_called_with_tool_results(self):
        """on_iteration receives assistant message and tool results."""
        call_count = 0
        callbacks = []

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "t", "arguments": "{}"},
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            return '{"ok": true}'

        def on_iter(iteration, assistant_msg, tool_results):
            callbacks.append((iteration, assistant_msg is not None, tool_results))

        asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
                on_iteration=on_iter,
            )
        )

        assert len(callbacks) == 2
        # First iteration: has tool results
        assert callbacks[0][0] == 1
        assert callbacks[0][1] is True  # assistant_msg not None
        assert callbacks[0][2] is not None  # tool_results present
        # Second iteration: no tool results (agent done)
        assert callbacks[1][0] == 2
        assert callbacks[1][2] is None


