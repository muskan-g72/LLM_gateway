from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID, uuid4

from app.prompt_builder import PromptBuildError
from app.skills import SkillDefinition, SkillError
from app.task_executor import (
    CompletionUsage,
    PreparedTask,
    TaskExecutionError,
    TaskExecutor,
    task_error_category,
)
from app.tracing import (
    AttemptRepository,
    ExecutionAttempt,
    ExecutionRecorder,
    StoreTraceRecorder,
    ToolTraceStatus,
    WorkflowStepSettlement,
)
from app.workflows import (
    WorkflowDefinition,
    WorkflowDefinitionError,
    WorkflowInputError,
    WorkflowRegistry,
    WorkflowStepDefinition,
    build_builtin_workflow_registry,
    resolve_step_input,
    validated_workflow_input,
)


@dataclass(frozen=True)
class WorkflowTokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class WorkflowStepResult:
    step_order: int
    step_id: str
    name: str
    skill: str
    task_id: UUID | None
    status: Literal["completed", "failed"]
    provider: str | None
    attempts: int
    tool_count: int
    usage: WorkflowTokenUsage
    error_category: str | None
    output: dict[str, object] | None

    def settlement(self) -> WorkflowStepSettlement:
        return WorkflowStepSettlement(
            step_order=self.step_order,
            task_id=str(self.task_id) if self.task_id is not None else None,
            status="completed" if self.status == "completed" else "failed",
            provider=self.provider,
            attempts=self.attempts,
            tool_count=self.tool_count,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            error_category=self.error_category,
        )


@dataclass(frozen=True)
class PreparedWorkflow:
    workflow_id: UUID
    definition: WorkflowDefinition
    original_input: dict[str, object]
    preferences: dict[str, object] | None
    first_task: PreparedTask


@dataclass(frozen=True)
class WorkflowExecutionResult:
    workflow_id: UUID
    workflow: str
    steps: tuple[WorkflowStepResult, ...]
    output: dict[str, object]
    usage: WorkflowTokenUsage
    attempts: int
    tool_count: int
    completion_usage: tuple[CompletionUsage, ...]


class WorkflowExecutionError(Exception):
    """Safe terminal workflow failure carrying settlement metadata."""

    def __init__(
        self,
        message: str,
        prepared: PreparedWorkflow,
        steps: tuple[WorkflowStepResult, ...],
        completion_usage: tuple[CompletionUsage, ...],
        error_category: str,
    ) -> None:
        self.workflow_id = prepared.workflow_id
        self.definition = prepared.definition
        self.steps = steps
        self.completion_usage = completion_usage
        self.error_category = error_category
        super().__init__(message)


class WorkflowStepExecutionError(WorkflowExecutionError):
    """One TaskExecutor step failed under its existing bounded policy."""


class WorkflowTraceRecordingError(WorkflowExecutionError):
    """Workflow or step trace metadata could not be recorded safely."""


class WorkflowMappingExecutionError(WorkflowExecutionError):
    """A trusted later-step mapping could not be resolved at runtime."""


class WorkflowRepository(AttemptRepository, Protocol):
    def start_workflow_step(
        self,
        workflow_id: str,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> None: ...


class WorkflowExecutionRecorder(Protocol):
    def start_step(
        self,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> ExecutionRecorder: ...


class StoreWorkflowRecorder:
    """Start each step trace atomically and return its normal task recorder."""

    def __init__(self, repository: WorkflowRepository, workflow_id: str) -> None:
        self._repository = repository
        self._workflow_id = workflow_id

    def start_step(
        self,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> ExecutionRecorder:
        self._repository.start_workflow_step(
            self._workflow_id,
            step_order,
            task_id,
            skill,
        )
        return StoreTraceRecorder(self._repository, task_id)


class _CountingRecorder:
    """Count safe tool starts while forwarding the complete task trace."""

    def __init__(self, delegate: ExecutionRecorder) -> None:
        self._delegate = delegate
        self.tool_count = 0

    def record(self, attempt: ExecutionAttempt) -> None:
        self._delegate.record(attempt)

    def start_tool(self, tool_number: int, tool_name: str) -> None:
        self._delegate.start_tool(tool_number, tool_name)
        self.tool_count += 1

    def finish_tool(
        self,
        tool_number: int,
        status: ToolTraceStatus,
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        self._delegate.finish_tool(
            tool_number,
            status,
            error_category,
            duration_ms,
        )


def _usage(completions: tuple[CompletionUsage, ...]) -> WorkflowTokenUsage:
    return WorkflowTokenUsage(
        prompt_tokens=sum(item.prompt_tokens for item in completions),
        completion_tokens=sum(item.completion_tokens for item in completions),
    )


class WorkflowExecutor:
    """Run a predeclared workflow sequentially by delegating every step."""

    def __init__(
        self,
        task_executor: TaskExecutor,
        registry: WorkflowRegistry | None = None,
    ) -> None:
        self.task_executor = task_executor
        self.registry = registry or build_builtin_workflow_registry(
            task_executor.skill_loader,
            task_executor.tool_registry,
        )

    def prepare(
        self,
        workflow_name: str,
        workflow_input: dict[str, object],
        preferences: dict[str, object] | None = None,
    ) -> PreparedWorkflow:
        definition = self.registry.get(workflow_name)
        original_input = validated_workflow_input(workflow_input)
        first_step = definition.steps[0]
        first_input = resolve_step_input(first_step, original_input, {})
        first_task = self._prepare_task(first_step, first_input, preferences)
        return PreparedWorkflow(
            workflow_id=uuid4(),
            definition=definition,
            original_input=original_input,
            preferences=deepcopy(preferences),
            first_task=first_task,
        )

    def _prepare_task(
        self,
        step: WorkflowStepDefinition,
        task_input: dict[str, object],
        preferences: dict[str, object] | None,
    ) -> PreparedTask:
        try:
            base_skill = self.task_executor.skill_loader.load(step.skill)
        except SkillError:
            raise WorkflowDefinitionError(
                "workflow step skill could not be loaded"
            ) from None

        skill_data = base_skill.model_dump(mode="python", by_alias=True)
        skill_data["allowed_tools"] = [step.tool] if step.tool is not None else []
        restricted_skill = SkillDefinition.model_validate(skill_data)
        try:
            return self.task_executor.prepare_with_skill(
                restricted_skill,
                task_input,
                preferences,
            )
        except PromptBuildError:
            raise WorkflowInputError(
                "mapped workflow input does not satisfy its skill"
            ) from None

    def execute(
        self,
        prepared: PreparedWorkflow,
        recorder: WorkflowExecutionRecorder,
    ) -> WorkflowExecutionResult:
        prior_outputs: dict[str, dict[str, object]] = {}
        step_results: list[WorkflowStepResult] = []
        completions: list[CompletionUsage] = []

        for index, step in enumerate(prepared.definition.steps):
            step_order = index + 1
            if index == 0:
                task = prepared.first_task
            else:
                try:
                    task_input = resolve_step_input(
                        step,
                        prepared.original_input,
                        prior_outputs,
                    )
                    task = self._prepare_task(
                        step,
                        task_input,
                        prepared.preferences,
                    )
                except (WorkflowInputError, WorkflowDefinitionError):
                    failed = self._failed_step(
                        step_order,
                        step,
                        None,
                        (),
                        0,
                        "workflow_mapping",
                    )
                    step_results.append(failed)
                    raise WorkflowMappingExecutionError(
                        "workflow step mapping failed",
                        prepared,
                        tuple(step_results),
                        tuple(completions),
                        "workflow_mapping",
                    ) from None

            try:
                step_recorder = recorder.start_step(
                    step_order,
                    str(task.task_id),
                    step.skill,
                )
            except Exception:
                failed = self._failed_step(
                    step_order,
                    step,
                    None,
                    (),
                    0,
                    "trace_recording",
                )
                step_results.append(failed)
                raise WorkflowTraceRecordingError(
                    "workflow step trace could not be started",
                    prepared,
                    tuple(step_results),
                    tuple(completions),
                    "trace_recording",
                ) from None

            counting_recorder = _CountingRecorder(step_recorder)
            try:
                result = self.task_executor.execute(task, counting_recorder)
            except TaskExecutionError as error:
                completions.extend(error.completion_usage)
                failed = self._failed_step(
                    step_order,
                    step,
                    task.task_id,
                    error.completion_usage,
                    counting_recorder.tool_count,
                    task_error_category(error),
                    attempts=error.attempts,
                )
                step_results.append(failed)
                raise WorkflowStepExecutionError(
                    "workflow step execution failed",
                    prepared,
                    tuple(step_results),
                    tuple(completions),
                    task_error_category(error),
                ) from None

            output = result.output.model_dump(mode="json")
            prior_outputs[step.step_id] = deepcopy(output)
            completions.extend(result.completion_usage)
            step_results.append(
                WorkflowStepResult(
                    step_order=step_order,
                    step_id=step.step_id,
                    name=step.name,
                    skill=step.skill,
                    task_id=result.task_id,
                    status="completed",
                    provider=result.provider,
                    attempts=result.attempts,
                    tool_count=counting_recorder.tool_count,
                    usage=WorkflowTokenUsage(
                        prompt_tokens=result.usage.prompt_tokens,
                        completion_tokens=result.usage.completion_tokens,
                    ),
                    error_category=None,
                    output=output,
                )
            )

        total_usage = _usage(tuple(completions))
        return WorkflowExecutionResult(
            workflow_id=prepared.workflow_id,
            workflow=prepared.definition.id,
            steps=tuple(step_results),
            output=deepcopy(step_results[-1].output or {}),
            usage=total_usage,
            attempts=sum(step.attempts for step in step_results),
            tool_count=sum(step.tool_count for step in step_results),
            completion_usage=tuple(completions),
        )

    @staticmethod
    def _failed_step(
        step_order: int,
        step: WorkflowStepDefinition,
        task_id: UUID | None,
        completion_usage: tuple[CompletionUsage, ...],
        tool_count: int,
        error_category: str,
        *,
        attempts: int = 0,
    ) -> WorkflowStepResult:
        return WorkflowStepResult(
            step_order=step_order,
            step_id=step.step_id,
            name=step.name,
            skill=step.skill,
            task_id=task_id,
            status="failed",
            provider=None,
            attempts=attempts,
            tool_count=tool_count,
            usage=_usage(completion_usage),
            error_category=error_category,
            output=None,
        )
