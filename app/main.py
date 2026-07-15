from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings
from app.db import GatewayStore
from app.providers import ProviderError, ProviderGateway


settings = get_settings()
store = GatewayStore(settings.database_path)
providers = ProviderGateway(settings)


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