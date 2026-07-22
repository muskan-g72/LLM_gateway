from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings
from app.db import GatewayStore
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
)


settings = get_settings()
store = GatewayStore(settings.database_path)
providers = ProviderGateway(settings)
task_executor = TaskExecutor(
    SkillLoader(),
    PromptBuilder(),
    providers,
    OutputValidator(),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.initialize()
    yield


app = FastAPI(
    title="Minimal LLM Gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
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


def _settle_failed_task(key: str, error: TaskExecutionError) -> None:
    try:
        if error.completion_usage:
            store.record_usage_events(
                key,
                _usage_event_tuples(error.completion_usage),
            )
        else:
            store.release_request(key)
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
        prepared = task_executor.prepare(
            request.skill,
            request.input,
            request.preferences,
        )
    except UnknownSkillError:
        raise HTTPException(status_code=404, detail="unknown skill") from None
    except PromptBuildError:
        raise HTTPException(status_code=422, detail="invalid task input") from None
    except SkillError:
        raise HTTPException(status_code=500, detail="skill configuration error") from None

    reservation = store.reserve_request(key)
    if reservation == "unknown":
        raise HTTPException(status_code=401, detail="missing or unknown virtual key")
    if reservation == "over_budget":
        raise HTTPException(status_code=429, detail="virtual key budget exhausted")

    try:
        result = task_executor.execute(prepared)
    except TaskExecutionError as error:
        _settle_failed_task(key, error)
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
        raise HTTPException(status_code=500, detail="task execution failed") from None

    try:
        store.record_usage_events(
            key,
            _usage_event_tuples(result.completion_usage),
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
