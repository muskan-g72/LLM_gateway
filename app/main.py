from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings
from app.db import GatewayStore, virtual_key_identifier
from app.memory import PreferenceError, PreferenceService
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuildError, PromptBuilder
from app.providers import ProviderError, ProviderGateway
from app.skills import SkillError, SkillLoader, UnknownSkillError
from app.task_executor import (
    CompletionUsage,
    TaskExecutionError,
    TaskExecutor,
    TaskInternalError,
    TaskInvalidOutputError,
    TaskProviderConfigurationError,
    TaskProvidersUnavailableError,
    TaskToolExecutionError,
    TaskTraceRecordingError,
    task_error_category,
)
from app.tools import ToolError
from app.tracing import StoreTraceRecorder
from app.workflow_executor import (
    StoreWorkflowRecorder,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowMappingExecutionError,
    WorkflowStepExecutionError,
    WorkflowStepResult,
    WorkflowTraceRecordingError,
)
from app.workflows import (
    UnknownWorkflowError,
    WorkflowDefinitionError,
    WorkflowInputError,
)


settings = get_settings()
store = GatewayStore(settings.database_url)
providers = ProviderGateway(settings)
task_executor = TaskExecutor(
    SkillLoader(),
    PromptBuilder(),
    providers,
    OutputValidator(),
)
workflow_executor = WorkflowExecutor(task_executor)


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.initialize()
    yield


app = FastAPI(
    title="Orchestrix",
    description="AI execution platform with virtual keys, provider fallback, budget enforcement, and persistent tracing.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain text")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def model_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must contain text")
        return value

    @model_validator(mode="after")
    def requires_a_user_message(self) -> "ChatRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("at least one user message is required")
        return self


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    content: str
    usage: TokenUsage


class UsageResponse(BaseModel):
    key: str
    requests: int
    tokens_in: int
    tokens_out: int
    spend: int
    budget: int
    remaining: int


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(min_length=1)
    input: dict[str, object]
    preferences: dict[str, object] | None = None

    @field_validator("skill")
    @classmethod
    def skill_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("skill must contain text")
        return value


class TaskResponse(BaseModel):
    task_id: UUID
    status: Literal["completed"]
    skill: str
    output: dict[str, object]
    provider: str
    attempts: int
    usage: TokenUsage


class TaskAttemptResponse(BaseModel):
    attempt_number: int
    provider: str
    attempt_type: Literal[
        "initial",
        "repair",
        "fallback",
        "fallback_repair",
        "post_tool",
        "post_tool_repair",
        "post_tool_fallback",
        "post_tool_fallback_repair",
    ]
    status: Literal[
        "completed",
        "validation_error",
        "operational_error",
        "configuration_error",
    ]
    usage: TokenUsage
    validation_error_category: Literal[
        "parsing",
        "structure",
        "semantic",
        "tool_protocol",
    ] | None
    provider_error_category: Literal["operational", "configuration"] | None
    created_at: str


class ToolTraceResponse(BaseModel):
    tool_number: int
    tool_name: str
    status: Literal["running", "completed", "failed"]
    error_category: str | None
    duration_ms: int
    created_at: str
    completed_at: str | None


class TaskTraceResponse(BaseModel):
    task_id: str
    status: Literal["running", "completed", "failed"]
    skill: str
    provider: str | None
    attempts: int
    usage: TokenUsage
    error_category: str | None
    created_at: str
    completed_at: str | None
    attempt_history: list[TaskAttemptResponse]
    tool_history: list[ToolTraceResponse] = Field(default_factory=list)


class PreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferences: dict[str, object]


class PreferencesResponse(BaseModel):
    preferences: dict[str, object]


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: str = Field(min_length=1)
    input: dict[str, object]
    preferences: dict[str, object] | None = None

    @field_validator("workflow")
    @classmethod
    def workflow_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow must contain text")
        return value


class WorkflowStepResponse(BaseModel):
    step_order: int
    step_id: str
    name: str
    skill: str
    status: Literal["completed"]
    provider: str
    attempts: int
    tool_count: int
    usage: TokenUsage


class WorkflowResponse(BaseModel):
    workflow_id: UUID
    status: Literal["completed"]
    workflow: str
    steps: list[WorkflowStepResponse]
    output: dict[str, object]
    usage: TokenUsage


class WorkflowTraceStepResponse(BaseModel):
    step_order: int
    step_id: str
    name: str
    skill: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    provider: str | None
    attempts: int
    tool_count: int
    usage: TokenUsage
    error_category: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class WorkflowTraceResponse(BaseModel):
    workflow_id: str
    workflow: str
    name: str
    description: str
    status: Literal["running", "completed", "failed"]
    step_count: int
    completed_steps: int
    attempts: int
    tool_count: int
    usage: TokenUsage
    error_category: str | None
    created_at: str
    completed_at: str | None
    steps: list[WorkflowTraceStepResponse]


def _virtual_key(authorization: str | None) -> str:
    if authorization is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")

    pieces = authorization.strip().split(None, 1)
    if len(pieces) != 2 or pieces[0].lower() != "bearer" or not pieces[1].strip():
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    return pieces[1].strip()


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(
    request: ChatRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    key = _virtual_key(authorization)
    reservation = store.reserve_request(key)
    if reservation == "unknown":
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    if reservation == "over_budget":
        raise HTTPException(status_code=429, detail="virtual key budget exhausted")

    messages = [message.model_dump() for message in request.messages]
    try:
        completion = providers.complete(messages)
    except ProviderError:
        store.release_request(key)
        raise HTTPException(status_code=502, detail="all providers failed") from None

    store.record_success(
        key=key,
        provider=completion.provider,
        tokens_in=completion.prompt_tokens,
        tokens_out=completion.completion_tokens,
    )
    return ChatResponse(
        content=completion.content,
        usage=TokenUsage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        ),
    )


def _usage_event_tuples(
    completion_usage: tuple[CompletionUsage, ...],
) -> list[tuple[str, int, int]]:
    return [
        (item.provider, item.prompt_tokens, item.completion_tokens)
        for item in completion_usage
    ]


def _task_error_category(error: TaskExecutionError) -> str:
    return task_error_category(error)


def _settle_failed_task(
    key: str,
    task_id: str,
    error: TaskExecutionError,
) -> None:
    try:
        if error.completion_usage:
            store.record_usage_events(
                key,
                _usage_event_tuples(error.completion_usage),
                task_id=task_id,
                trace_status="failed",
                attempts=error.attempts,
                error_category=_task_error_category(error),
            )
        else:
            store.finalize_failed_task_without_usage(
                key,
                task_id,
                error.attempts,
                _task_error_category(error),
            )
    except Exception:
        raise HTTPException(status_code=500, detail="task accounting failed") from None


@app.post("/v1/tasks/execute", response_model=TaskResponse)
def execute_task(
    request: TaskRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> TaskResponse:
    key = _virtual_key(authorization)
    if store.get_usage(key) is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")

    try:
        skill = task_executor.skill_loader.load(request.skill)
    except UnknownSkillError:
        raise HTTPException(status_code=404, detail="unknown skill") from None
    except SkillError:
        raise HTTPException(status_code=500, detail="skill configuration error") from None

    owner_id = virtual_key_identifier(key)
    preference_service = PreferenceService(store)
    try:
        stored_preferences = preference_service.get(owner_id)
        effective_preferences = preference_service.merge(
            stored_preferences,
            request.preferences,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="preference storage failed") from None

    prompt_preferences: dict[str, object] | None
    if effective_preferences or request.preferences is not None:
        prompt_preferences = effective_preferences
    else:
        prompt_preferences = None

    try:
        prepared = task_executor.prepare_with_skill(
            skill,
            request.input,
            prompt_preferences,
        )
    except PromptBuildError:
        raise HTTPException(status_code=422, detail="invalid task input") from None
    except ToolError:
        raise HTTPException(status_code=500, detail="skill configuration error") from None

    reservation = store.reserve_request(key)
    if reservation == "unknown":
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    if reservation == "over_budget":
        raise HTTPException(status_code=429, detail="virtual key budget exhausted")

    task_id = str(prepared.task_id)
    try:
        store.create_task_execution(task_id, owner_id, prepared.skill.name)
    except Exception:
        try:
            store.release_request(key)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="task tracing failed") from None

    try:
        result = task_executor.execute(
            prepared,
            StoreTraceRecorder(store, task_id),
        )
    except TaskExecutionError as error:
        _settle_failed_task(key, task_id, error)
        if isinstance(error, TaskProviderConfigurationError):
            raise HTTPException(
                status_code=500,
                detail="provider configuration error",
            ) from None
        if isinstance(error, TaskInternalError):
            raise HTTPException(
                status_code=500,
                detail="task configuration error",
            ) from None
        if isinstance(error, TaskInvalidOutputError):
            raise HTTPException(
                status_code=502,
                detail="provider output remained invalid",
            ) from None
        if isinstance(error, TaskProvidersUnavailableError):
            raise HTTPException(
                status_code=502,
                detail="all providers unavailable",
            ) from None
        if isinstance(error, TaskTraceRecordingError):
            raise HTTPException(
                status_code=500,
                detail="task tracing failed",
            ) from None
        if isinstance(error, TaskToolExecutionError):
            raise HTTPException(
                status_code=502,
                detail="tool execution failed",
            ) from None
        raise HTTPException(status_code=500, detail="task execution failed") from None

    try:
        store.record_usage_events(
            key,
            _usage_event_tuples(result.completion_usage),
            task_id=task_id,
            trace_status="completed",
            final_provider=result.provider,
            attempts=result.attempts,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="task accounting failed") from None

    return TaskResponse(
        task_id=result.task_id,
        status="completed",
        skill=result.skill,
        output=result.output.model_dump(mode="json"),
        provider=result.provider,
        attempts=result.attempts,
        usage=TokenUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
        ),
    )


def _workflow_step_response(step: WorkflowStepResult) -> WorkflowStepResponse:
    if step.status != "completed" or step.provider is None:
        raise RuntimeError("successful workflow contains an incomplete step")
    return WorkflowStepResponse(
        step_order=step.step_order,
        step_id=step.step_id,
        name=step.name,
        skill=step.skill,
        status="completed",
        provider=step.provider,
        attempts=step.attempts,
        tool_count=step.tool_count,
        usage=TokenUsage(
            prompt_tokens=step.usage.prompt_tokens,
            completion_tokens=step.usage.completion_tokens,
        ),
    )


@app.post("/v1/workflows/execute", response_model=WorkflowResponse)
def execute_workflow(
    request: WorkflowRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> WorkflowResponse:
    key = _virtual_key(authorization)
    if store.get_usage(key) is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")

    try:
        workflow_executor.registry.get(request.workflow)
    except UnknownWorkflowError:
        raise HTTPException(status_code=404, detail="unknown workflow") from None

    owner_id = virtual_key_identifier(key)
    preference_service = PreferenceService(store)
    try:
        stored_preferences = preference_service.get(owner_id)
        effective_preferences = preference_service.merge(
            stored_preferences,
            request.preferences,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="preference storage failed") from None

    prompt_preferences: dict[str, object] | None
    if effective_preferences or request.preferences is not None:
        prompt_preferences = effective_preferences
    else:
        prompt_preferences = None

    try:
        prepared = workflow_executor.prepare(
            request.workflow,
            request.input,
            prompt_preferences,
        )
    except WorkflowInputError:
        raise HTTPException(status_code=422, detail="invalid workflow input") from None
    except (WorkflowDefinitionError, ToolError):
        raise HTTPException(
            status_code=500,
            detail="workflow configuration error",
        ) from None

    reservation = store.reserve_request(key)
    if reservation == "unknown":
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    if reservation == "over_budget":
        raise HTTPException(status_code=429, detail="virtual key budget exhausted")

    workflow_id = str(prepared.workflow_id)
    try:
        store.create_workflow_execution(
            workflow_id,
            owner_id,
            prepared.definition.id,
            prepared.definition.name,
            prepared.definition.description,
            [
                (index, step.step_id, step.name, step.skill)
                for index, step in enumerate(prepared.definition.steps, start=1)
            ],
        )
    except Exception:
        try:
            store.release_request(key)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="workflow tracing failed") from None

    try:
        result = workflow_executor.execute(
            prepared,
            StoreWorkflowRecorder(store, workflow_id),
        )
    except WorkflowExecutionError as error:
        try:
            store.settle_workflow(
                key,
                workflow_id,
                "failed",
                [step.settlement() for step in error.steps],
                _usage_event_tuples(error.completion_usage),
                error.error_category,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="workflow accounting failed",
            ) from None

        if isinstance(
            error,
            (WorkflowTraceRecordingError, WorkflowMappingExecutionError),
        ) or error.error_category in {
            "configuration",
            "internal",
            "trace_recording",
        }:
            raise HTTPException(
                status_code=500,
                detail="workflow execution failed",
            ) from None
        if isinstance(error, WorkflowStepExecutionError):
            raise HTTPException(
                status_code=502,
                detail="workflow step failed",
            ) from None
        raise HTTPException(
            status_code=500,
            detail="workflow execution failed",
        ) from None

    try:
        store.settle_workflow(
            key,
            workflow_id,
            "completed",
            [step.settlement() for step in result.steps],
            _usage_event_tuples(result.completion_usage),
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="workflow accounting failed",
        ) from None

    return WorkflowResponse(
        workflow_id=result.workflow_id,
        status="completed",
        workflow=result.workflow,
        steps=[_workflow_step_response(step) for step in result.steps],
        output=result.output,
        usage=TokenUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
        ),
    )


@app.get("/v1/workflows/{workflow_id}", response_model=WorkflowTraceResponse)
def get_workflow_trace(
    workflow_id: str,
    authorization: Annotated[str | None, Header()] = None,
) -> WorkflowTraceResponse:
    key = _virtual_key(authorization)
    if store.get_usage(key) is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")

    try:
        trace = store.get_workflow_execution(
            workflow_id,
            virtual_key_identifier(key),
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="workflow trace lookup failed",
        ) from None
    if trace is None:
        raise HTTPException(status_code=404, detail="workflow not found")

    return WorkflowTraceResponse(
        workflow_id=trace.workflow_id,
        workflow=trace.definition_id,
        name=trace.name,
        description=trace.description,
        status=trace.status,
        step_count=trace.step_count,
        completed_steps=trace.completed_steps,
        attempts=trace.attempts,
        tool_count=trace.tool_count,
        usage=TokenUsage(
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
        ),
        error_category=trace.error_category,
        created_at=trace.created_at,
        completed_at=trace.completed_at,
        steps=[
            WorkflowTraceStepResponse(
                step_order=step.step_order,
                step_id=step.step_id,
                name=step.name,
                skill=step.skill,
                status=step.status,
                provider=step.provider,
                attempts=step.attempts,
                tool_count=step.tool_count,
                usage=TokenUsage(
                    prompt_tokens=step.prompt_tokens,
                    completion_tokens=step.completion_tokens,
                ),
                error_category=step.error_category,
                created_at=step.created_at,
                started_at=step.started_at,
                completed_at=step.completed_at,
            )
            for step in trace.steps
        ],
    )


@app.get(
    "/v1/tasks/{task_id}",
    response_model=TaskTraceResponse,
    response_model_exclude_defaults=True,
)
def get_task_trace(
    task_id: str,
    authorization: Annotated[str | None, Header()] = None,
) -> TaskTraceResponse:
    key = _virtual_key(authorization)
    if store.get_usage(key) is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")

    try:
        trace = store.get_task_execution(task_id, virtual_key_identifier(key))
    except Exception:
        raise HTTPException(status_code=500, detail="task trace lookup failed") from None
    if trace is None:
        raise HTTPException(status_code=404, detail="task not found")

    return TaskTraceResponse(
        task_id=trace.task_id,
        status=trace.status,
        skill=trace.skill,
        provider=trace.final_provider,
        attempts=trace.attempts,
        usage=TokenUsage(
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
        ),
        error_category=trace.error_category,
        created_at=trace.created_at,
        completed_at=trace.completed_at,
        attempt_history=[
            TaskAttemptResponse(
                attempt_number=attempt.attempt_number,
                provider=attempt.provider,
                attempt_type=attempt.attempt_type,
                status=attempt.status,
                usage=TokenUsage(
                    prompt_tokens=attempt.prompt_tokens,
                    completion_tokens=attempt.completion_tokens,
                ),
                validation_error_category=attempt.validation_error_category,
                provider_error_category=attempt.provider_error_category,
                created_at=attempt.created_at,
            )
            for attempt in trace.attempt_history
        ],
        tool_history=[
            ToolTraceResponse(
                tool_number=tool.tool_number,
                tool_name=tool.tool_name,
                status=tool.status,
                error_category=tool.error_category,
                duration_ms=tool.duration_ms,
                created_at=tool.created_at,
                completed_at=tool.completed_at,
            )
            for tool in trace.tool_history
        ],
    )


def _authenticated_owner_id(authorization: str | None) -> str:
    key = _virtual_key(authorization)
    if store.get_usage(key) is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    return virtual_key_identifier(key)


@app.get("/v1/preferences", response_model=PreferencesResponse)
def get_preferences(
    authorization: Annotated[str | None, Header()] = None,
) -> PreferencesResponse:
    owner_id = _authenticated_owner_id(authorization)
    try:
        preferences = PreferenceService(store).get(owner_id)
    except Exception:
        raise HTTPException(status_code=500, detail="preference storage failed") from None
    return PreferencesResponse(preferences=preferences)


@app.put("/v1/preferences", response_model=PreferencesResponse)
def put_preferences(
    request: PreferencesRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> PreferencesResponse:
    owner_id = _authenticated_owner_id(authorization)
    try:
        preferences = PreferenceService(store).put(
            owner_id,
            request.preferences,
        )
    except PreferenceError:
        raise HTTPException(status_code=422, detail="invalid preferences") from None
    except Exception:
        raise HTTPException(status_code=500, detail="preference storage failed") from None
    return PreferencesResponse(preferences=preferences)


@app.delete("/v1/preferences/{preference_key}", status_code=204)
def delete_preference(
    preference_key: str,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    owner_id = _authenticated_owner_id(authorization)
    try:
        PreferenceService(store).delete(owner_id, preference_key)
    except PreferenceError:
        raise HTTPException(status_code=422, detail="invalid preference name") from None
    except Exception:
        raise HTTPException(status_code=500, detail="preference storage failed") from None
    return Response(status_code=204)


@app.get("/usage", response_model=UsageResponse)
def usage(key: Annotated[str | None, Query()] = None) -> UsageResponse:
    if not key:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    stats = store.get_usage(key)
    if stats is None:
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    return UsageResponse(**stats.as_contract())


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
