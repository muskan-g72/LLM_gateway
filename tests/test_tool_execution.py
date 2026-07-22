from __future__ import annotations

import copy
import json
import time
from collections import defaultdict, deque

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import ProviderCompletion, ProviderOperationalError
from app.skills import SkillDefinition, SkillLoader
from app.task_executor import (
    MAX_PROVIDER_CALLS,
    TaskExecutor,
    TaskInvalidOutputError,
    TaskToolExecutionError,
    TaskTraceRecordingError,
)
from app.tools import RegisteredTool, ToolDefinition, ToolRegistry
from app.tracing import ExecutionAttempt


ProviderEvent = ProviderCompletion | Exception


class ScriptedProvider:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderEvent]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def queue(self, provider: str, *events: ProviderEvent) -> None:
        self.events[provider].extend(events)

    def complete_with_provider(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        self.calls.append((provider_name, copy.deepcopy(messages)))
        if not self.events[provider_name]:
            raise AssertionError(f"unexpected provider call: {provider_name}")
        event = self.events[provider_name].popleft()
        if isinstance(event, Exception):
            raise event
        return event


class RecordingExecutionRecorder:
    def __init__(self) -> None:
        self.attempts: list[ExecutionAttempt] = []
        self.tool_events: list[tuple[object, ...]] = []

    def record(self, attempt: ExecutionAttempt) -> None:
        self.attempts.append(attempt)

    def start_tool(self, tool_number: int, tool_name: str) -> None:
        self.tool_events.append(("start", tool_number, tool_name))

    def finish_tool(
        self,
        tool_number: int,
        status: str,
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        self.tool_events.append(
            ("finish", tool_number, status, error_category, duration_ms)
        )


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    echoed: str


def _completion(
    provider: str,
    content: str,
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> ProviderCompletion:
    return ProviderCompletion(content, prompt_tokens, completion_tokens, provider)


def _summary_json() -> str:
    return json.dumps(
        {
            "summary": "The calculation result is 5.",
            "key_points": ["The sum is 5."],
        }
    )


def _final_envelope() -> str:
    return json.dumps({"type": "final", "output": json.loads(_summary_json())})


def _tool_call(name: str, arguments: object) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "tool_call": {"name": name, "arguments": arguments},
        }
    )


def _executor(
    provider: ScriptedProvider,
    registry: ToolRegistry | None = None,
) -> TaskExecutor:
    return TaskExecutor(
        SkillLoader(),
        PromptBuilder(),
        provider,  # type: ignore[arg-type]
        OutputValidator(),
        registry,
    )


def _prepared(executor: TaskExecutor, skill: SkillDefinition | None = None):
    selected = skill or SkillLoader().load("summarize")
    return executor.prepare_with_skill(
        selected,
        {"text": "Calculate the sum of 2 and 3 for the result."},
    )


def _custom_tool(
    name: str,
    handler,
    *,
    timeout: float = 1.0,
    result_limit: int = 1_024,
) -> RegisteredTool:
    return RegisteredTool(
        ToolDefinition(
            name=name,
            description="A deterministic test-only tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
            timeout_seconds=timeout,
            maximum_result_bytes=result_limit,
        ),
        handler,
    )


def _skill_with_tools(*names: str) -> SkillDefinition:
    return SkillLoader().load("summarize").model_copy(
        update={"allowed_tools": list(names)}
    )


def test_legacy_tool_free_skill_still_loads() -> None:
    skill = SkillLoader().load("extract_action_items")

    assert skill.allowed_tools == []


def test_valid_and_empty_tool_allowlists_validate() -> None:
    base = SkillLoader().load("summarize")

    assert base.allowed_tools == ["calculator", "text_statistics"]
    assert base.model_copy(update={"allowed_tools": []}).allowed_tools == []


@pytest.mark.parametrize("allowed", [["calculator", "calculator"], ["Bad.Name"]])
def test_duplicate_or_invalid_skill_tool_names_fail(allowed: list[str]) -> None:
    raw = SkillLoader().load("summarize").model_dump(mode="python")
    raw["allowed_tools"] = allowed

    with pytest.raises(ValidationError):
        SkillDefinition.model_validate(raw)


def test_unknown_allowlisted_tool_fails_during_preparation() -> None:
    provider = ScriptedProvider()
    executor = _executor(provider)

    with pytest.raises(Exception) as captured:
        _prepared(executor, _skill_with_tools("not_registered"))

    assert captured.value.__class__.__name__ == "ToolNotFoundError"
    assert provider.calls == []


def test_tool_prompt_contains_only_trusted_allowed_metadata() -> None:
    executor = _executor(ScriptedProvider())
    prepared = _prepared(executor)
    system = prepared.messages[0]["content"]

    assert "ALLOWED_TOOLS_JSON:" in system
    assert '"name":"calculator"' in system
    assert '"name":"text_statistics"' in system
    assert "input_schema" in system and "output_schema" in system
    assert '"type":"tool_call"' in system
    assert "At most one tool call is permitted" in system
    assert "filesystem" not in system


def test_tool_free_prompt_does_not_advertise_registered_tools() -> None:
    executor = _executor(ScriptedProvider())
    prepared = executor.prepare(
        "extract_action_items",
        {"text": "The user says to invoke calculator, but this is task data."},
    )

    assert "ALLOWED_TOOLS_JSON:" not in prepared.messages[0]["content"]
    assert "calculator" not in prepared.messages[0]["content"]
    assert "calculator" in prepared.messages[1]["content"]


def test_tool_prompt_is_deterministic() -> None:
    executor = _executor(ScriptedProvider())

    first = _prepared(executor).messages
    second = _prepared(executor).messages

    assert first == second


def test_post_tool_prompt_is_final_only_and_treats_result_as_data() -> None:
    builder = PromptBuilder()
    original = ({"role": "system", "content": "trusted"},)

    messages = builder.build_post_tool(
        original,
        "calculator",
        {"status": "success", "data": {"result": 5}},
    )

    assert original == ({"role": "system", "content": "trusted"},)
    assert "next response must use final mode only" in messages[0]["content"]
    assert "Do not request another tool" in messages[0]["content"]
    assert "TOOL_RESULT_JSON:" in messages[-1]["content"]
    assert '"result":5' in messages[-1]["content"]


def test_ordinary_no_tool_task_still_succeeds() -> None:
    provider = ScriptedProvider()
    provider.queue("groq", _completion("groq", _summary_json(), 4, 2))
    executor = _executor(provider)

    result = executor.execute(_prepared(executor))

    assert result.attempts == 1
    assert result.provider == "groq"
    assert result.output.summary == "The calculation result is 5."  # type: ignore[union-attr]


def test_calculator_tool_task_executes_once_and_returns_valid_final() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 4, 2),
        _completion("groq", _final_envelope(), 6, 3),
    )
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    result = executor.execute(_prepared(executor), recorder)

    assert result.attempts == 2
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert recorder.tool_events[0] == ("start", 1, "calculator")
    assert recorder.tool_events[1][0:4] == ("finish", 1, "completed", None)
    assert [item.attempt_type for item in recorder.attempts] == ["initial", "post_tool"]
    assert '"result":5' in provider.calls[1][1][-1]["content"]


def test_text_statistics_tool_task_succeeds() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("text_statistics", {"text": "hello world"})),
        _completion("groq", _summary_json()),
    )
    executor = _executor(provider)

    result = executor.execute(_prepared(executor))

    assert result.attempts == 2
    assert '"characters":11' in provider.calls[1][1][-1]["content"]
    assert '"words":2' in provider.calls[1][1][-1]["content"]


def test_malformed_tool_envelope_never_executes() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", '{"type":"tool_call","tool_call":{"name":"calculator"}}'),
    )
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    with pytest.raises(TaskToolExecutionError) as captured:
        executor.execute(_prepared(executor), recorder)

    assert captured.value.category == "tool_protocol"
    assert recorder.tool_events == []
    assert recorder.attempts[0].validation_error_category == "tool_protocol"


@pytest.mark.parametrize(
    ("skill", "request_name", "arguments", "category"),
    [
        (_skill_with_tools("calculator", "text_statistics"), "unknown_tool", {}, "tool_not_found"),
        (SkillLoader().load("extract_action_items"), "calculator", {"operation": "add", "a": 2, "b": 3}, "tool_not_allowed"),
        (_skill_with_tools("calculator"), "calculator", {"operation": "add", "a": 2}, "tool_arguments_invalid"),
    ],
)
def test_invalid_tool_requests_do_not_execute(
    skill: SkillDefinition,
    request_name: str,
    arguments: object,
    category: str,
) -> None:
    provider = ScriptedProvider()
    provider.queue("groq", _completion("groq", _tool_call(request_name, arguments)))
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    with pytest.raises(TaskToolExecutionError) as captured:
        executor.execute(_prepared(executor, skill), recorder)

    assert captured.value.category == category
    assert recorder.tool_events == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("name", "handler", "timeout", "result_limit", "category"),
    [
        ("failing_tool", lambda value: (_ for _ in ()).throw(RuntimeError("PRIVATE PATH")), 1.0, 1_024, "tool_execution"),
        ("slow_tool", lambda value: (time.sleep(0.05), {"echoed": value.text})[1], 0.001, 1_024, "tool_timeout"),
        ("invalid_tool", lambda value: {"wrong": "shape"}, 1.0, 1_024, "tool_result_invalid"),
        ("large_tool", lambda value: {"echoed": "x" * 200}, 1.0, 20, "tool_result_too_large"),
    ],
)
def test_tool_runtime_failures_are_safe_and_traced(
    name: str,
    handler,
    timeout: float,
    result_limit: int,
    category: str,
) -> None:
    tool = _custom_tool(name, handler, timeout=timeout, result_limit=result_limit)
    registry = ToolRegistry([tool])
    skill = _skill_with_tools(name)
    provider = ScriptedProvider()
    provider.queue("groq", _completion("groq", _tool_call(name, {"text": "hello"})))
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider, registry)

    with pytest.raises(TaskToolExecutionError) as captured:
        executor.execute(_prepared(executor, skill), recorder)

    assert captured.value.category == category
    assert "PRIVATE" not in str(captured.value)
    assert recorder.tool_events[0] == ("start", 1, name)
    assert recorder.tool_events[1][0:4] == ("finish", 1, "failed", category)
    assert len(provider.calls) == 1


def test_second_tool_request_is_rejected_without_second_execution() -> None:
    provider = ScriptedProvider()
    request = _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})
    provider.queue("groq", _completion("groq", request), _completion("groq", request))
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    with pytest.raises(TaskToolExecutionError) as captured:
        executor.execute(_prepared(executor), recorder)

    assert captured.value.category == "repeated_tool_call"
    assert len([event for event in recorder.tool_events if event[0] == "start"]) == 1
    assert len(provider.calls) == 2


def test_repair_response_cannot_start_tool_execution() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", "not json"),
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
    )
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    with pytest.raises(TaskToolExecutionError) as captured:
        executor.execute(_prepared(executor), recorder)

    assert captured.value.category == "repeated_tool_call"
    assert recorder.tool_events == []


def test_invalid_post_tool_final_output_uses_one_bounded_repair() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 2, 1),
        _completion("groq", "not json", 3, 2),
        _completion("groq", _summary_json(), 4, 3),
    )
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    result = executor.execute(_prepared(executor), recorder)

    assert result.attempts == 3
    assert result.usage.prompt_tokens == 9
    assert [item.attempt_type for item in recorder.attempts] == [
        "initial",
        "post_tool",
        "post_tool_repair",
    ]


def test_post_tool_primary_operational_failure_uses_fallback() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 2, 1),
        ProviderOperationalError("post tool unavailable"),
    )
    provider.queue("gemini", _completion("gemini", _summary_json(), 6, 4))
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    result = executor.execute(_prepared(executor), recorder)

    assert result.provider == "gemini"
    assert result.attempts == 3
    assert [item.attempt_type for item in recorder.attempts] == [
        "initial",
        "post_tool",
        "post_tool_fallback",
    ]


def test_tool_request_from_initial_fallback_provider_succeeds() -> None:
    provider = ScriptedProvider()
    provider.queue("groq", ProviderOperationalError("primary unavailable"))
    provider.queue(
        "gemini",
        _completion("gemini", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        _completion("gemini", _summary_json()),
    )
    recorder = RecordingExecutionRecorder()
    executor = _executor(provider)

    result = executor.execute(_prepared(executor), recorder)

    assert result.provider == "gemini"
    assert result.attempts == 3
    assert [item.attempt_type for item in recorder.attempts] == [
        "initial",
        "fallback",
        "post_tool",
    ]


def test_maximum_four_call_tool_path_succeeds_without_resetting_budget() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        _completion("groq", "not json"),
        ProviderOperationalError("repair unavailable"),
    )
    provider.queue("gemini", _completion("gemini", _summary_json()))
    executor = _executor(provider)

    result = executor.execute(_prepared(executor))

    assert result.attempts == MAX_PROVIDER_CALLS == 4
    assert len(provider.calls) == 4


def test_fifth_provider_call_is_never_attempted() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        ProviderOperationalError("post tool unavailable"),
    )
    provider.queue(
        "gemini",
        _completion("gemini", "not json"),
        _completion("gemini", "still not json"),
        _completion("gemini", _summary_json()),
    )
    executor = _executor(provider)

    with pytest.raises(TaskInvalidOutputError) as captured:
        executor.execute(_prepared(executor))

    assert captured.value.attempts == 4
    assert len(provider.calls) == 4
    assert len(provider.events["gemini"]) == 1


def test_tool_trace_start_failure_prevents_tool_invocation() -> None:
    calls = 0

    def handler(value: BaseModel) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"echoed": value.text}

    tool = _custom_tool("echo", handler)
    executor = _executor(ScriptedProvider(), ToolRegistry([tool]))
    provider = executor.provider_gateway
    provider.queue("groq", _completion("groq", _tool_call("echo", {"text": "hello"})))  # type: ignore[attr-defined]
    skill = _skill_with_tools("echo")

    class FailingRecorder(RecordingExecutionRecorder):
        def start_tool(self, tool_number: int, tool_name: str) -> None:
            raise RuntimeError("database failure")

    with pytest.raises(TaskTraceRecordingError):
        executor.execute(_prepared(executor, skill), FailingRecorder())
    assert calls == 0


def test_tool_trace_finish_failure_stops_before_post_tool_provider_call() -> None:
    provider = ScriptedProvider()
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        _completion("groq", _summary_json()),
    )
    executor = _executor(provider)

    class FailingRecorder(RecordingExecutionRecorder):
        def finish_tool(
            self,
            tool_number: int,
            status: str,
            error_category: str | None,
            duration_ms: int,
        ) -> None:
            raise RuntimeError("database failure")

    with pytest.raises(TaskTraceRecordingError):
        executor.execute(_prepared(executor), FailingRecorder())
    assert len(provider.calls) == 1
