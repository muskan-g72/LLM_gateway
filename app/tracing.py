from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


AttemptType = Literal[
    "initial",
    "repair",
    "fallback",
    "fallback_repair",
    "post_tool",
    "post_tool_repair",
    "post_tool_fallback",
    "post_tool_fallback_repair",
]
AttemptStatus = Literal[
    "completed",
    "validation_error",
    "operational_error",
    "configuration_error",
]
ValidationErrorCategory = Literal[
    "parsing",
    "structure",
    "semantic",
    "tool_protocol",
]
ProviderErrorCategory = Literal["operational", "configuration"]
TraceStatus = Literal["running", "completed", "failed"]
ToolTraceStatus = Literal["running", "completed", "failed"]


@dataclass(frozen=True)
class ExecutionAttempt:
    """Safe metadata for one provider invocation; model content is excluded."""

    attempt_number: int
    provider: str
    attempt_type: AttemptType
    status: AttemptStatus
    prompt_tokens: int = 0
    completion_tokens: int = 0
    validation_error_category: ValidationErrorCategory | None = None
    provider_error_category: ProviderErrorCategory | None = None


@dataclass(frozen=True)
class StoredAttempt:
    attempt_number: int
    provider: str
    attempt_type: AttemptType
    status: AttemptStatus
    prompt_tokens: int
    completion_tokens: int
    validation_error_category: ValidationErrorCategory | None
    provider_error_category: ProviderErrorCategory | None
    created_at: str


@dataclass(frozen=True)
class StoredToolExecution:
    tool_number: int
    tool_name: str
    status: ToolTraceStatus
    error_category: str | None
    duration_ms: int
    created_at: str
    completed_at: str | None


@dataclass(frozen=True)
class StoredTaskTrace:
    task_id: str
    status: TraceStatus
    skill: str
    final_provider: str | None
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    error_category: str | None
    created_at: str
    completed_at: str | None
    attempt_history: tuple[StoredAttempt, ...]
    tool_history: tuple[StoredToolExecution, ...]


class AttemptRecorder(Protocol):
    """Persistence boundary used by TaskExecutor without coupling it to HTTP."""

    def record(self, attempt: ExecutionAttempt) -> None: ...


class ExecutionRecorder(AttemptRecorder, Protocol):
    def start_tool(self, tool_number: int, tool_name: str) -> None: ...

    def finish_tool(
        self,
        tool_number: int,
        status: Literal["completed", "failed"],
        error_category: str | None,
        duration_ms: int,
    ) -> None: ...


class AttemptRepository(Protocol):
    def append_task_attempt(self, task_id: str, attempt: ExecutionAttempt) -> None: ...

    def create_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        tool_name: str,
    ) -> None: ...

    def finalize_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        status: Literal["completed", "failed"],
        error_category: str | None,
        duration_ms: int,
    ) -> None: ...


class StoreTraceRecorder:
    """Bind one task ID to a repository that persists safe attempt metadata."""

    def __init__(self, repository: AttemptRepository, task_id: str) -> None:
        self._repository = repository
        self._task_id = task_id

    def record(self, attempt: ExecutionAttempt) -> None:
        self._repository.append_task_attempt(self._task_id, attempt)

    def start_tool(self, tool_number: int, tool_name: str) -> None:
        self._repository.create_task_tool_execution(
            self._task_id,
            tool_number,
            tool_name,
        )

    def finish_tool(
        self,
        tool_number: int,
        status: Literal["completed", "failed"],
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        self._repository.finalize_task_tool_execution(
            self._task_id,
            tool_number,
            status,
            error_category,
            duration_ms,
        )
