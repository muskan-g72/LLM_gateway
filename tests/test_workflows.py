from __future__ import annotations

import json
from collections import defaultdict, deque

import pytest
from pydantic import ValidationError

from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import ProviderCompletion, ProviderOperationalError
from app.skills import SkillLoader
from app.task_executor import TaskExecutor
from app.tools import build_builtin_tool_registry
from app.tracing import ExecutionAttempt
from app.workflow_executor import (
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowStepExecutionError,
)
from app.workflows import (
    MAX_WORKFLOW_STEPS,
    UnknownWorkflowError,
    WorkflowDefinition,
    WorkflowDefinitionError,
    WorkflowInputError,
    WorkflowRegistry,
    WorkflowStepDefinition,
    resolve_step_input,
)


ProviderEvent = ProviderCompletion | Exception


class WorkflowProvider:
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
        self.calls.append(
            (provider_name, [dict(message) for message in messages])
        )
        if not self.events[provider_name]:
            raise AssertionError(f"unexpected provider call: {provider_name}")
        event = self.events[provider_name].popleft()
        if isinstance(event, Exception):
            raise event
        return event


class MemoryTaskRecorder:
    def __init__(self) -> None:
        self.attempts: list[ExecutionAttempt] = []
        self.tools: list[dict[str, object]] = []

    def record(self, attempt: ExecutionAttempt) -> None:
        self.attempts.append(attempt)

    def start_tool(self, tool_number: int, tool_name: str) -> None:
        self.tools.append(
            {
                "tool_number": tool_number,
                "tool_name": tool_name,
                "status": "running",
            }
        )

    def finish_tool(
        self,
        tool_number: int,
        status: str,
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        tool = self.tools[tool_number - 1]
        tool.update(
            {
                "status": status,
                "error_category": error_category,
                "duration_ms": duration_ms,
            }
        )


class MemoryWorkflowRecorder:
    def __init__(self) -> None:
        self.steps: list[tuple[int, str, str, MemoryTaskRecorder]] = []

    def start_step(
        self,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> MemoryTaskRecorder:
        recorder = MemoryTaskRecorder()
        self.steps.append((step_order, task_id, skill, recorder))
        return recorder


def _completion(
    content: str,
    provider: str = "groq",
    prompt_tokens: int = 2,
    completion_tokens: int = 1,
) -> ProviderCompletion:
    return ProviderCompletion(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=provider,
    )


def _summary(
    summary: str = "The release includes notes for Friday.",
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "key_points": ["The release notes are due Friday."],
        }
    )


def _actions() -> str:
    return json.dumps(
        {
            "action_items": [
                {
                    "task": "Maya must publish the release notes",
                    "owner": "Maya",
                    "deadline": "Friday",
                }
            ]
        }
    )


def _tool_call() -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "tool_call": {
                "name": "text_statistics",
                "arguments": {"text": "release notes Friday"},
            },
        }
    )


def _executor(
    provider: WorkflowProvider,
) -> WorkflowExecutor:
    task_executor = TaskExecutor(
        SkillLoader(),
        PromptBuilder(),
        provider,  # type: ignore[arg-type]
        OutputValidator(),
    )
    return WorkflowExecutor(task_executor)


def _input() -> dict[str, object]:
    return {
        "text": (
            "The team approved the release. "
            "Maya must publish the release notes by Friday."
        )
    }


def _queue_success(
    provider: WorkflowProvider,
    *,
    tool: bool = False,
) -> None:
    provider.queue(
        "groq",
        _completion(_summary()),
        _completion(_actions()),
    )
    if tool:
        provider.queue(
            "groq",
            _completion(_tool_call()),
            _completion(_summary("The final release report includes statistics.")),
        )
    else:
        provider.queue(
            "groq",
            _completion(_summary("The final release report is ready.")),
        )


def test_builtin_workflow_is_fixed_bounded_and_ordered() -> None:
    executor = _executor(WorkflowProvider())
    definition = executor.registry.get("article_processing")

    assert executor.registry.names == ("article_processing",)
    assert [step.step_id for step in definition.steps] == [
        "summary",
        "action_items",
        "final_report",
    ]
    assert [step.skill for step in definition.steps] == [
        "summarize",
        "extract_action_items",
        "summarize",
    ]
    assert definition.steps[-1].tool == "text_statistics"
    assert len(definition.steps) <= MAX_WORKFLOW_STEPS


def test_unknown_workflow_fails_without_accepting_a_definition() -> None:
    executor = _executor(WorkflowProvider())

    with pytest.raises(UnknownWorkflowError):
        executor.registry.get("model_generated_workflow")


def test_registry_returns_detached_definition_copies() -> None:
    executor = _executor(WorkflowProvider())
    first = executor.registry.get("article_processing")
    first.steps[0].input_mapping["text"] = ("input.changed",)

    second = executor.registry.get("article_processing")

    assert second.steps[0].input_mapping["text"] == ("input.text",)


def test_duplicate_and_forward_step_references_are_rejected() -> None:
    loader = SkillLoader()
    schema = loader.load("summarize").output_schema
    step = WorkflowStepDefinition(
        step_id="duplicate",
        name="Duplicate",
        skill="summarize",
        input_mapping={"text": ("input.text",)},
        output_schema=schema,
    )
    with pytest.raises(ValidationError):
        WorkflowDefinition(
            id="bad_duplicate",
            name="Bad duplicate",
            description="Invalid.",
            steps=(step, step),
        )

    forward = WorkflowStepDefinition(
        step_id="forward",
        name="Forward",
        skill="summarize",
        input_mapping={"text": ("steps.later.summary",)},
        output_schema=schema,
    )
    with pytest.raises(ValidationError):
        WorkflowDefinition(
            id="bad_forward",
            name="Bad forward",
            description="Invalid.",
            steps=(forward,),
        )


def test_workflow_step_count_is_bounded_before_execution() -> None:
    loader = SkillLoader()
    schema = loader.load("summarize").output_schema
    steps = tuple(
        WorkflowStepDefinition(
            step_id=f"step_{index}",
            name=f"Step {index}",
            skill="summarize",
            input_mapping={"text": ("input.text",)},
            output_schema=schema,
        )
        for index in range(MAX_WORKFLOW_STEPS + 1)
    )

    with pytest.raises(ValidationError):
        WorkflowDefinition(
            id="too_many_steps",
            name="Too many steps",
            description="Invalid.",
            steps=steps,
        )


def test_registry_rejects_schema_mismatch_and_disallowed_tool() -> None:
    loader = SkillLoader()
    tools = build_builtin_tool_registry()
    bad_schema = WorkflowDefinition(
        id="bad_schema",
        name="Bad schema",
        description="Invalid.",
        steps=(
            WorkflowStepDefinition(
                step_id="one",
                name="One",
                skill="summarize",
                input_mapping={"text": ("input.text",)},
                output_schema=loader.load("extract_action_items").output_schema,
            ),
        ),
    )
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRegistry((bad_schema,), loader, tools)

    bad_tool = WorkflowDefinition(
        id="bad_tool",
        name="Bad tool",
        description="Invalid.",
        steps=(
            WorkflowStepDefinition(
                step_id="one",
                name="One",
                skill="extract_action_items",
                tool="calculator",
                input_mapping={"text": ("input.text",)},
                output_schema=loader.load("extract_action_items").output_schema,
            ),
        ),
    )
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRegistry((bad_tool,), loader, tools)


def test_mapping_is_deterministic_and_uses_only_declared_sources() -> None:
    executor = _executor(WorkflowProvider())
    definition = executor.registry.get("article_processing")
    original = _input()
    outputs = {
        "summary": {
            "summary": "Release notes are due Friday.",
            "key_points": ["Maya owns the notes."],
        }
    }

    first = resolve_step_input(definition.steps[1], original, outputs)
    second = resolve_step_input(definition.steps[1], original, outputs)

    assert first == second
    mapped = json.loads(first["text"])  # type: ignore[arg-type]
    assert mapped == {
        "input.text": original["text"],
        "steps.summary.summary": "Release notes are due Friday.",
    }
    assert original == _input()


def test_missing_original_mapping_input_fails_before_execution() -> None:
    executor = _executor(WorkflowProvider())

    with pytest.raises(WorkflowInputError):
        executor.prepare("article_processing", {})


def test_successful_workflow_runs_steps_sequentially() -> None:
    provider = WorkflowProvider()
    _queue_success(provider)
    executor = _executor(provider)
    recorder = MemoryWorkflowRecorder()

    result = executor.execute(
        executor.prepare("article_processing", _input()),
        recorder,
    )

    assert [step.step_id for step in result.steps] == [
        "summary",
        "action_items",
        "final_report",
    ]
    assert [item[0] for item in recorder.steps] == [1, 2, 3]
    assert result.output["summary"] == "The final release report is ready."
    assert result.attempts == 3
    assert result.usage.prompt_tokens == 6
    assert result.usage.completion_tokens == 3


def test_validated_previous_outputs_are_mapped_into_later_prompts() -> None:
    provider = WorkflowProvider()
    _queue_success(provider)
    executor = _executor(provider)

    executor.execute(
        executor.prepare("article_processing", _input()),
        MemoryWorkflowRecorder(),
    )

    second_prompt = provider.calls[1][1][-1]["content"]
    third_prompt = provider.calls[2][1][-1]["content"]
    assert "steps.summary.summary" in second_prompt
    assert "The release includes notes for Friday." in second_prompt
    assert "steps.action_items.action_items" in third_prompt
    assert "Maya must publish the release notes" in third_prompt


def test_stored_preferences_are_passed_to_every_step_without_mutation() -> None:
    provider = WorkflowProvider()
    _queue_success(provider)
    executor = _executor(provider)
    preferences = {"response_detail": "concise"}

    executor.execute(
        executor.prepare("article_processing", _input(), preferences),
        MemoryWorkflowRecorder(),
    )

    assert preferences == {"response_detail": "concise"}
    assert all(
        '"response_detail":"concise"' in messages[-1]["content"]
        for _, messages in provider.calls
    )


def test_repair_remains_bounded_inside_one_workflow_step() -> None:
    provider = WorkflowProvider()
    provider.queue(
        "groq",
        _completion("not json"),
        _completion(_summary()),
        _completion(_actions()),
        _completion(_summary("The final release report is ready.")),
    )
    executor = _executor(provider)

    result = executor.execute(
        executor.prepare("article_processing", _input()),
        MemoryWorkflowRecorder(),
    )

    assert result.steps[0].attempts == 2
    assert result.attempts == 4
    assert len(provider.calls) == 4


def test_operational_failure_uses_existing_fallback_inside_step() -> None:
    provider = WorkflowProvider()
    provider.queue(
        "groq",
        ProviderOperationalError("primary"),
        _completion(_actions()),
        _completion(_summary("The final release report is ready.")),
    )
    provider.queue("gemini", _completion(_summary(), provider="gemini"))
    executor = _executor(provider)

    result = executor.execute(
        executor.prepare("article_processing", _input()),
        MemoryWorkflowRecorder(),
    )

    assert result.steps[0].provider == "gemini"
    assert result.steps[0].attempts == 2
    assert result.steps[1].provider == "groq"


def test_tool_execution_is_limited_to_declared_workflow_step() -> None:
    provider = WorkflowProvider()
    _queue_success(provider, tool=True)
    executor = _executor(provider)
    recorder = MemoryWorkflowRecorder()

    result = executor.execute(
        executor.prepare("article_processing", _input()),
        recorder,
    )

    assert [step.tool_count for step in result.steps] == [0, 0, 1]
    assert result.tool_count == 1
    assert recorder.steps[2][3].tools[0]["tool_name"] == "text_statistics"
    assert "ALLOWED_TOOLS_JSON" not in provider.calls[0][1][0]["content"]
    assert "ALLOWED_TOOLS_JSON" not in provider.calls[1][1][0]["content"]
    assert '"name":"text_statistics"' in provider.calls[2][1][0]["content"]
    assert '"name":"calculator"' not in provider.calls[2][1][0]["content"]
    assert "TOOL_RESULT_JSON" in provider.calls[-1][1][-1]["content"]


def test_failed_step_cancels_all_later_steps() -> None:
    provider = WorkflowProvider()
    provider.queue(
        "groq",
        _completion(_summary()),
        _completion("not json"),
        _completion("still not json"),
    )
    executor = _executor(provider)
    recorder = MemoryWorkflowRecorder()

    with pytest.raises(WorkflowStepExecutionError) as captured:
        executor.execute(
            executor.prepare("article_processing", _input()),
            recorder,
        )

    assert [step.status for step in captured.value.steps] == [
        "completed",
        "failed",
    ]
    assert [item[0] for item in recorder.steps] == [1, 2]
    assert len(provider.calls) == 3


def test_four_call_limit_is_independent_for_each_step() -> None:
    provider = WorkflowProvider()
    provider.queue(
        "groq",
        _completion("not json"),
        ProviderOperationalError("repair failed"),
        _completion(_actions()),
        _completion(_summary("The final release report is ready.")),
    )
    provider.queue(
        "gemini",
        _completion("still invalid", provider="gemini"),
        _completion(_summary(), provider="gemini"),
    )
    executor = _executor(provider)

    result = executor.execute(
        executor.prepare("article_processing", _input()),
        MemoryWorkflowRecorder(),
    )

    assert result.steps[0].attempts == 4
    assert all(step.attempts <= 4 for step in result.steps)
    assert result.attempts == 6


def test_workflow_failure_exposes_only_safe_metadata() -> None:
    provider = WorkflowProvider()
    raw_secret = "RAW_MODEL_SECRET_C:\\private\\file"
    provider.queue(
        "groq",
        _completion(_summary()),
        _completion(raw_secret),
        _completion(raw_secret),
    )
    executor = _executor(provider)

    with pytest.raises(WorkflowExecutionError) as captured:
        executor.execute(
            executor.prepare("article_processing", _input()),
            MemoryWorkflowRecorder(),
        )

    assert raw_secret not in str(captured.value)
    assert all(step.output is None for step in captured.value.steps if step.status == "failed")
