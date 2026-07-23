from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Literal, cast

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.migrations import upgrade_database
from app.tracing import (
    AttemptStatus,
    AttemptType,
    ExecutionAttempt,
    ProviderErrorCategory,
    StoredAttempt,
    StoredTaskTrace,
    StoredToolExecution,
    StoredWorkflowStep,
    StoredWorkflowTrace,
    ToolTraceStatus,
    TraceStatus,
    ValidationErrorCategory,
    WorkflowStepSettlement,
    WorkflowStepTraceStatus,
    WorkflowTraceStatus,
)


ReservationResult = Literal["reserved", "unknown", "over_budget"]

SEEDED_KEYS = {
    "vk_open": 50,
    "vk_tiny": 2,
    "vk_edge": 1,
}

def virtual_key_identifier(key: str) -> str:
    """Return a stable pseudonymous owner ID without persisting a plaintext key."""
    return sha256(key.encode("utf-8")).hexdigest()


def _timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        rendered = value.isoformat()
        return rendered.replace("+00:00", "Z")
    return str(value)


class StoreConsistencyError(RuntimeError):
    """An expected running trace or owner-scoped row could not be updated."""


class DatabaseInitializationError(RuntimeError):
    """PostgreSQL could not be verified or initialized safely."""


@dataclass(frozen=True)
class UsageStats:
    key: str
    requests: int
    tokens_in: int
    tokens_out: int
    budget: int

    def as_contract(self) -> dict[str, int | str]:
        return {
            "key": self.key,
            "requests": self.requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "spend": self.requests,
            "budget": self.budget,
            "remaining": max(self.budget - self.requests, 0),
        }


class GatewayStore:
    """PostgreSQL repository preserving the gateway's existing storage boundary."""

    def __init__(self, database_url: str) -> None:
        url = make_url(database_url)
        if url.get_backend_name() != "postgresql":
            raise ValueError("GatewayStore requires a PostgreSQL DATABASE_URL")
        self.database_url = database_url
        self.engine: Engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
        self._sessions = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def initialize(self) -> None:
        """Apply Alembic migrations, verify connectivity, and idempotently seed keys."""
        try:
            upgrade_database(self.database_url)
            with self._sessions.begin() as session:
                session.execute(text("SELECT 1"))
                session.execute(
                    text(
                        """
                        INSERT INTO virtual_keys (key, budget)
                        VALUES (:key, :budget)
                        ON CONFLICT (key) DO NOTHING
                        """
                    ),
                    [
                        {"key": key, "budget": budget}
                        for key, budget in SEEDED_KEYS.items()
                    ],
                )
        except Exception:
            raise DatabaseInitializationError(
                "PostgreSQL is unavailable or could not be initialized"
            ) from None

    def dispose(self) -> None:
        """Release pooled connections, primarily for deterministic test cleanup."""
        self.engine.dispose()

    def reserve_request(self, key: str) -> ReservationResult:
        """Atomically admit one request without overspending its final unit."""
        with self._sessions.begin() as session:
            admitted = session.execute(
                text(
                    """
                    UPDATE virtual_keys
                    SET requests = requests + 1
                    WHERE key = :key AND requests < budget
                    RETURNING key
                    """
                ),
                {"key": key},
            ).first()
            if admitted is not None:
                return "reserved"
            exists = session.execute(
                text("SELECT 1 FROM virtual_keys WHERE key = :key"),
                {"key": key},
            ).first()
            return "over_budget" if exists is not None else "unknown"

    def release_request(self, key: str) -> None:
        """Undo a reservation when no provider produced a billable completion."""
        with self._sessions.begin() as session:
            session.execute(
                text(
                    """
                    UPDATE virtual_keys
                    SET requests = GREATEST(requests - 1, 0)
                    WHERE key = :key
                    """
                ),
                {"key": key},
            )

    def record_success(
        self,
        key: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        self.record_usage_events(key, [(provider, tokens_in, tokens_out)])

    def record_usage_events(
        self,
        key: str,
        events: Sequence[tuple[str, int, int]],
        *,
        task_id: str | None = None,
        trace_status: Literal["completed", "failed"] | None = None,
        final_provider: str | None = None,
        attempts: int | None = None,
        error_category: str | None = None,
    ) -> None:
        """Record completion usage and optionally finalize its task trace atomically."""
        if not events and task_id is None:
            return
        if (task_id is None) != (trace_status is None):
            raise ValueError("task trace finalization fields must be supplied together")
        if task_id is not None and attempts is None:
            raise ValueError("task trace finalization requires an attempt count")

        token_input_total = sum(tokens_in for _, tokens_in, _ in events)
        token_output_total = sum(tokens_out for _, _, tokens_out in events)
        with self._sessions.begin() as session:
            if events:
                session.execute(
                    text(
                        """
                        UPDATE virtual_keys
                        SET tokens_in = tokens_in + :tokens_in,
                            tokens_out = tokens_out + :tokens_out
                        WHERE key = :key
                        """
                    ),
                    {
                        "tokens_in": token_input_total,
                        "tokens_out": token_output_total,
                        "key": key,
                    },
                )
                session.execute(
                    text(
                        """
                        INSERT INTO usage_events (
                            virtual_key, provider, tokens_in, tokens_out
                        )
                        VALUES (
                            :virtual_key, :provider, :tokens_in, :tokens_out
                        )
                        """
                    ),
                    [
                        {
                            "virtual_key": key,
                            "provider": provider,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        }
                        for provider, tokens_in, tokens_out in events
                    ],
                )

            if task_id is not None and trace_status is not None:
                cursor = session.execute(
                    text(
                        """
                        UPDATE task_executions
                        SET status = :status,
                            final_provider = :final_provider,
                            attempts = :attempts,
                            prompt_tokens = :prompt_tokens,
                            completion_tokens = :completion_tokens,
                            error_category = :error_category,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE task_id = :task_id
                          AND virtual_key_id = :owner_id
                          AND status = 'running'
                        """
                    ),
                    {
                        "status": trace_status,
                        "final_provider": final_provider,
                        "attempts": attempts,
                        "prompt_tokens": token_input_total,
                        "completion_tokens": token_output_total,
                        "error_category": error_category,
                        "task_id": task_id,
                        "owner_id": virtual_key_identifier(key),
                    },
                )
                if cursor.rowcount != 1:
                    raise StoreConsistencyError("task trace could not be finalized")

    def create_task_execution(
        self,
        task_id: str,
        virtual_key_id: str,
        skill: str,
    ) -> None:
        with self._sessions.begin() as session:
            session.execute(
                text(
                    """
                    INSERT INTO task_executions (
                        task_id, virtual_key_id, skill, status
                    )
                    VALUES (:task_id, :virtual_key_id, :skill, 'running')
                    """
                ),
                {
                    "task_id": task_id,
                    "virtual_key_id": virtual_key_id,
                    "skill": skill,
                },
            )

    def append_task_attempt(
        self,
        task_id: str,
        attempt: ExecutionAttempt,
    ) -> None:
        with self._sessions.begin() as session:
            cursor = session.execute(
                text(
                    """
                    INSERT INTO task_attempts (
                        task_id, attempt_number, provider, attempt_type, status,
                        prompt_tokens, completion_tokens,
                        validation_error_category, provider_error_category
                    )
                    SELECT
                        :task_id, :attempt_number, :provider, :attempt_type,
                        :status, :prompt_tokens, :completion_tokens,
                        :validation_error_category, :provider_error_category
                    FROM task_executions
                    WHERE task_id = :running_task_id AND status = 'running'
                    """
                ),
                {
                    "task_id": task_id,
                    "attempt_number": attempt.attempt_number,
                    "provider": attempt.provider,
                    "attempt_type": attempt.attempt_type,
                    "status": attempt.status,
                    "prompt_tokens": attempt.prompt_tokens,
                    "completion_tokens": attempt.completion_tokens,
                    "validation_error_category": attempt.validation_error_category,
                    "provider_error_category": attempt.provider_error_category,
                    "running_task_id": task_id,
                },
            )
            if cursor.rowcount != 1:
                raise StoreConsistencyError("task attempt could not be recorded")

    def create_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        tool_name: str,
    ) -> None:
        with self._sessions.begin() as session:
            cursor = session.execute(
                text(
                    """
                    INSERT INTO task_tool_executions (
                        task_id, tool_number, tool_name, status
                    )
                    SELECT :task_id, :tool_number, :tool_name, 'running'
                    FROM task_executions
                    WHERE task_id = :running_task_id AND status = 'running'
                    """
                ),
                {
                    "task_id": task_id,
                    "tool_number": tool_number,
                    "tool_name": tool_name,
                    "running_task_id": task_id,
                },
            )
            if cursor.rowcount != 1:
                raise StoreConsistencyError("tool execution could not be recorded")

    def finalize_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        status: Literal["completed", "failed"],
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        with self._sessions.begin() as session:
            cursor = session.execute(
                text(
                    """
                    UPDATE task_tool_executions
                    SET status = :status,
                        error_category = :error_category,
                        duration_ms = :duration_ms,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = :task_id
                      AND tool_number = :tool_number
                      AND status = 'running'
                    """
                ),
                {
                    "status": status,
                    "error_category": error_category,
                    "duration_ms": duration_ms,
                    "task_id": task_id,
                    "tool_number": tool_number,
                },
            )
            if cursor.rowcount != 1:
                raise StoreConsistencyError("tool execution could not be finalized")

    def finalize_failed_task_without_usage(
        self,
        key: str,
        task_id: str,
        attempts: int,
        error_category: str,
    ) -> None:
        """Atomically release an unbilled reservation and fail its running trace."""
        with self._sessions.begin() as session:
            cursor = session.execute(
                text(
                    """
                    UPDATE task_executions
                    SET status = 'failed',
                        attempts = :attempts,
                        error_category = :error_category,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = :task_id
                      AND virtual_key_id = :owner_id
                      AND status = 'running'
                    """
                ),
                {
                    "attempts": attempts,
                    "error_category": error_category,
                    "task_id": task_id,
                    "owner_id": virtual_key_identifier(key),
                },
            )
            if cursor.rowcount != 1:
                raise StoreConsistencyError("task trace could not be finalized")
            session.execute(
                text(
                    """
                    UPDATE virtual_keys
                    SET requests = GREATEST(requests - 1, 0)
                    WHERE key = :key
                    """
                ),
                {"key": key},
            )

    def get_task_execution(
        self,
        task_id: str,
        virtual_key_id: str,
    ) -> StoredTaskTrace | None:
        with self._sessions() as session:
            row = session.execute(
                text(
                    """
                    SELECT task_id, status, skill, final_provider, attempts,
                           prompt_tokens, completion_tokens, error_category,
                           created_at, completed_at
                    FROM task_executions
                    WHERE task_id = :task_id AND virtual_key_id = :virtual_key_id
                    """
                ),
                {"task_id": task_id, "virtual_key_id": virtual_key_id},
            ).mappings().first()
            if row is None:
                return None
            attempt_rows = session.execute(
                text(
                    """
                    SELECT attempt_number, provider, attempt_type, status,
                           prompt_tokens, completion_tokens,
                           validation_error_category, provider_error_category,
                           created_at
                    FROM task_attempts
                    WHERE task_id = :task_id
                    ORDER BY attempt_number ASC
                    """
                ),
                {"task_id": task_id},
            ).mappings().all()
            tool_rows = session.execute(
                text(
                    """
                    SELECT tool_number, tool_name, status, error_category,
                           duration_ms, created_at, completed_at
                    FROM task_tool_executions
                    WHERE task_id = :task_id
                    ORDER BY tool_number ASC
                    """
                ),
                {"task_id": task_id},
            ).mappings().all()

        attempts = tuple(
            StoredAttempt(
                attempt_number=item["attempt_number"],
                provider=item["provider"],
                attempt_type=cast(AttemptType, item["attempt_type"]),
                status=cast(AttemptStatus, item["status"]),
                prompt_tokens=item["prompt_tokens"],
                completion_tokens=item["completion_tokens"],
                validation_error_category=cast(
                    ValidationErrorCategory | None,
                    item["validation_error_category"],
                ),
                provider_error_category=cast(
                    ProviderErrorCategory | None,
                    item["provider_error_category"],
                ),
                created_at=cast(str, _timestamp(item["created_at"])),
            )
            for item in attempt_rows
        )
        tools = tuple(
            StoredToolExecution(
                tool_number=item["tool_number"],
                tool_name=item["tool_name"],
                status=cast(ToolTraceStatus, item["status"]),
                error_category=item["error_category"],
                duration_ms=item["duration_ms"],
                created_at=cast(str, _timestamp(item["created_at"])),
                completed_at=_timestamp(item["completed_at"]),
            )
            for item in tool_rows
        )
        return StoredTaskTrace(
            task_id=row["task_id"],
            status=cast(TraceStatus, row["status"]),
            skill=row["skill"],
            final_provider=row["final_provider"],
            attempts=row["attempts"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            error_category=row["error_category"],
            created_at=cast(str, _timestamp(row["created_at"])),
            completed_at=_timestamp(row["completed_at"]),
            attempt_history=attempts,
            tool_history=tools,
        )

    def create_workflow_execution(
        self,
        workflow_id: str,
        virtual_key_id: str,
        definition_id: str,
        name: str,
        description: str,
        steps: Sequence[tuple[int, str, str, str]],
    ) -> None:
        """Create one running workflow and all fixed pending steps atomically."""
        if not steps:
            raise ValueError("workflow must contain at least one step")
        with self._sessions.begin() as session:
            session.execute(
                text(
                    """
                    INSERT INTO workflow_executions (
                        workflow_id, virtual_key_id, definition_id, name,
                        description, status, step_count
                    )
                    VALUES (
                        :workflow_id, :virtual_key_id, :definition_id, :name,
                        :description, 'running', :step_count
                    )
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "virtual_key_id": virtual_key_id,
                    "definition_id": definition_id,
                    "name": name,
                    "description": description,
                    "step_count": len(steps),
                },
            )
            session.execute(
                text(
                    """
                    INSERT INTO workflow_steps (
                        workflow_id, step_order, step_id, name, skill, status
                    )
                    VALUES (
                        :workflow_id, :step_order, :step_id,
                        :name, :skill, 'pending'
                    )
                    """
                ),
                [
                    {
                        "workflow_id": workflow_id,
                        "step_order": step_order,
                        "step_id": step_id,
                        "name": step_name,
                        "skill": skill,
                    }
                    for step_order, step_id, step_name, skill in steps
                ],
            )

    def start_workflow_step(
        self,
        workflow_id: str,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> None:
        """Atomically mark one declared step running and create its task trace."""
        with self._sessions.begin() as session:
            row = session.execute(
                text(
                    """
                    SELECT workflow_executions.virtual_key_id
                    FROM workflow_executions
                    JOIN workflow_steps
                      ON workflow_steps.workflow_id =
                         workflow_executions.workflow_id
                    WHERE workflow_executions.workflow_id = :workflow_id
                      AND workflow_executions.status = 'running'
                      AND workflow_steps.step_order = :step_order
                      AND workflow_steps.skill = :skill
                      AND workflow_steps.status = 'pending'
                    FOR UPDATE OF workflow_steps
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "step_order": step_order,
                    "skill": skill,
                },
            ).mappings().first()
            if row is None:
                raise StoreConsistencyError("workflow is not running")
            session.execute(
                text(
                    """
                    INSERT INTO task_executions (
                        task_id, virtual_key_id, skill, status
                    )
                    VALUES (:task_id, :virtual_key_id, :skill, 'running')
                    """
                ),
                {
                    "task_id": task_id,
                    "virtual_key_id": row["virtual_key_id"],
                    "skill": skill,
                },
            )
            cursor = session.execute(
                text(
                    """
                    UPDATE workflow_steps
                    SET status = 'running',
                        task_id = :task_id,
                        started_at = CURRENT_TIMESTAMP
                    WHERE workflow_id = :workflow_id
                      AND step_order = :step_order
                      AND skill = :skill
                      AND status = 'pending'
                    """
                ),
                {
                    "task_id": task_id,
                    "workflow_id": workflow_id,
                    "step_order": step_order,
                    "skill": skill,
                },
            )
            if cursor.rowcount != 1:
                raise StoreConsistencyError("workflow step could not be started")

    def settle_workflow(
        self,
        key: str,
        workflow_id: str,
        status: Literal["completed", "failed"],
        steps: Sequence[WorkflowStepSettlement],
        events: Sequence[tuple[str, int, int]],
        error_category: str | None = None,
    ) -> None:
        """Atomically settle usage, step task traces, and the workflow trace."""
        prompt_total = sum(tokens_in for _, tokens_in, _ in events)
        completion_total = sum(tokens_out for _, _, tokens_out in events)
        if prompt_total != sum(step.prompt_tokens for step in steps):
            raise ValueError("workflow prompt usage does not match its steps")
        if completion_total != sum(step.completion_tokens for step in steps):
            raise ValueError("workflow completion usage does not match its steps")

        owner_id = virtual_key_identifier(key)
        with self._sessions.begin() as session:
            workflow_row = session.execute(
                text(
                    """
                    SELECT step_count
                    FROM workflow_executions
                    WHERE workflow_id = :workflow_id
                      AND virtual_key_id = :owner_id
                      AND status = 'running'
                    FOR UPDATE
                    """
                ),
                {"workflow_id": workflow_id, "owner_id": owner_id},
            ).first()
            if workflow_row is None:
                raise StoreConsistencyError("workflow trace could not be finalized")

            if events:
                session.execute(
                    text(
                        """
                        UPDATE virtual_keys
                        SET tokens_in = tokens_in + :tokens_in,
                            tokens_out = tokens_out + :tokens_out
                        WHERE key = :key
                        """
                    ),
                    {
                        "tokens_in": prompt_total,
                        "tokens_out": completion_total,
                        "key": key,
                    },
                )
                session.execute(
                    text(
                        """
                        INSERT INTO usage_events (
                            virtual_key, provider, tokens_in, tokens_out
                        )
                        VALUES (
                            :virtual_key, :provider, :tokens_in, :tokens_out
                        )
                        """
                    ),
                    [
                        {
                            "virtual_key": key,
                            "provider": provider,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        }
                        for provider, tokens_in, tokens_out in events
                    ],
                )

            for step in steps:
                cursor = session.execute(
                    text(
                        """
                        UPDATE workflow_steps
                        SET status = :status,
                            provider = :provider,
                            attempts = :attempts,
                            tool_count = :tool_count,
                            prompt_tokens = :prompt_tokens,
                            completion_tokens = :completion_tokens,
                            error_category = :error_category,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE workflow_id = :workflow_id
                          AND step_order = :step_order
                          AND status IN ('pending', 'running')
                        """
                    ),
                    {
                        "status": step.status,
                        "provider": step.provider,
                        "attempts": step.attempts,
                        "tool_count": step.tool_count,
                        "prompt_tokens": step.prompt_tokens,
                        "completion_tokens": step.completion_tokens,
                        "error_category": step.error_category,
                        "workflow_id": workflow_id,
                        "step_order": step.step_order,
                    },
                )
                if cursor.rowcount != 1:
                    raise StoreConsistencyError(
                        "workflow step could not be finalized"
                    )

                if step.task_id is not None:
                    task_cursor = session.execute(
                        text(
                            """
                            UPDATE task_executions
                            SET status = :status,
                                final_provider = :provider,
                                attempts = :attempts,
                                prompt_tokens = :prompt_tokens,
                                completion_tokens = :completion_tokens,
                                error_category = :error_category,
                                completed_at = CURRENT_TIMESTAMP
                            WHERE task_id = :task_id
                              AND virtual_key_id = :owner_id
                              AND status = 'running'
                            """
                        ),
                        {
                            "status": step.status,
                            "provider": step.provider,
                            "attempts": step.attempts,
                            "prompt_tokens": step.prompt_tokens,
                            "completion_tokens": step.completion_tokens,
                            "error_category": step.error_category,
                            "task_id": step.task_id,
                            "owner_id": owner_id,
                        },
                    )
                    if task_cursor.rowcount != 1:
                        raise StoreConsistencyError(
                            "workflow task trace could not be finalized"
                        )

            if status == "failed":
                session.execute(
                    text(
                        """
                        UPDATE workflow_steps
                        SET status = 'skipped',
                            completed_at = CURRENT_TIMESTAMP
                        WHERE workflow_id = :workflow_id
                          AND status = 'pending'
                        """
                    ),
                    {"workflow_id": workflow_id},
                )

            attempts = sum(step.attempts for step in steps)
            tool_count = sum(step.tool_count for step in steps)
            completed_steps = sum(step.status == "completed" for step in steps)
            workflow_cursor = session.execute(
                text(
                    """
                    UPDATE workflow_executions
                    SET status = :status,
                        completed_steps = :completed_steps,
                        attempts = :attempts,
                        tool_count = :tool_count,
                        prompt_tokens = :prompt_tokens,
                        completion_tokens = :completion_tokens,
                        error_category = :error_category,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE workflow_id = :workflow_id
                      AND virtual_key_id = :owner_id
                      AND status = 'running'
                    """
                ),
                {
                    "status": status,
                    "completed_steps": completed_steps,
                    "attempts": attempts,
                    "tool_count": tool_count,
                    "prompt_tokens": prompt_total,
                    "completion_tokens": completion_total,
                    "error_category": error_category,
                    "workflow_id": workflow_id,
                    "owner_id": owner_id,
                },
            )
            if workflow_cursor.rowcount != 1:
                raise StoreConsistencyError("workflow trace could not be finalized")

            if not events:
                session.execute(
                    text(
                        """
                        UPDATE virtual_keys
                        SET requests = GREATEST(requests - 1, 0)
                        WHERE key = :key
                        """
                    ),
                    {"key": key},
                )

    def get_workflow_execution(
        self,
        workflow_id: str,
        virtual_key_id: str,
    ) -> StoredWorkflowTrace | None:
        with self._sessions() as session:
            row = session.execute(
                text(
                    """
                    SELECT workflow_id, definition_id, name, description, status,
                           step_count, completed_steps, attempts, tool_count,
                           prompt_tokens, completion_tokens, error_category,
                           created_at, completed_at
                    FROM workflow_executions
                    WHERE workflow_id = :workflow_id
                      AND virtual_key_id = :virtual_key_id
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "virtual_key_id": virtual_key_id,
                },
            ).mappings().first()
            if row is None:
                return None
            step_rows = session.execute(
                text(
                    """
                    SELECT step_order, step_id, name, skill, status, provider,
                           attempts, tool_count, prompt_tokens, completion_tokens,
                           error_category, created_at, started_at, completed_at
                    FROM workflow_steps
                    WHERE workflow_id = :workflow_id
                    ORDER BY step_order ASC
                    """
                ),
                {"workflow_id": workflow_id},
            ).mappings().all()

        steps = tuple(
            StoredWorkflowStep(
                step_order=item["step_order"],
                step_id=item["step_id"],
                name=item["name"],
                skill=item["skill"],
                status=cast(WorkflowStepTraceStatus, item["status"]),
                provider=item["provider"],
                attempts=item["attempts"],
                tool_count=item["tool_count"],
                prompt_tokens=item["prompt_tokens"],
                completion_tokens=item["completion_tokens"],
                error_category=item["error_category"],
                created_at=cast(str, _timestamp(item["created_at"])),
                started_at=_timestamp(item["started_at"]),
                completed_at=_timestamp(item["completed_at"]),
            )
            for item in step_rows
        )
        return StoredWorkflowTrace(
            workflow_id=row["workflow_id"],
            definition_id=row["definition_id"],
            name=row["name"],
            description=row["description"],
            status=cast(WorkflowTraceStatus, row["status"]),
            step_count=row["step_count"],
            completed_steps=row["completed_steps"],
            attempts=row["attempts"],
            tool_count=row["tool_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            error_category=row["error_category"],
            created_at=cast(str, _timestamp(row["created_at"])),
            completed_at=_timestamp(row["completed_at"]),
            steps=steps,
        )

    def get_preference_values(self, virtual_key_id: str) -> dict[str, str]:
        with self._sessions() as session:
            rows = session.execute(
                text(
                    """
                    SELECT preference_key, preference_value_json
                    FROM user_preferences
                    WHERE virtual_key_id = :virtual_key_id
                    ORDER BY preference_key ASC
                    """
                ),
                {"virtual_key_id": virtual_key_id},
            ).mappings().all()
        return {
            row["preference_key"]: row["preference_value_json"] for row in rows
        }

    def upsert_preference_values(
        self,
        virtual_key_id: str,
        values: dict[str, str],
    ) -> None:
        if not values:
            return
        with self._sessions.begin() as session:
            session.execute(
                text(
                    """
                    INSERT INTO user_preferences (
                        virtual_key_id, preference_key, preference_value_json
                    )
                    VALUES (
                        :virtual_key_id, :preference_key, :preference_value_json
                    )
                    ON CONFLICT (virtual_key_id, preference_key) DO UPDATE SET
                        preference_value_json =
                            EXCLUDED.preference_value_json,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                [
                    {
                        "virtual_key_id": virtual_key_id,
                        "preference_key": name,
                        "preference_value_json": value,
                    }
                    for name, value in values.items()
                ],
            )

    def delete_preference_value(
        self,
        virtual_key_id: str,
        preference_key: str,
    ) -> bool:
        with self._sessions.begin() as session:
            cursor = session.execute(
                text(
                    """
                    DELETE FROM user_preferences
                    WHERE virtual_key_id = :virtual_key_id
                      AND preference_key = :preference_key
                    """
                ),
                {
                    "virtual_key_id": virtual_key_id,
                    "preference_key": preference_key,
                },
            )
            return cursor.rowcount == 1

    def get_usage(self, key: str) -> UsageStats | None:
        with self._sessions() as session:
            row = session.execute(
                text(
                    """
                    SELECT key, requests, tokens_in, tokens_out, budget
                    FROM virtual_keys
                    WHERE key = :key
                    """
                ),
                {"key": key},
            ).mappings().first()
        if row is None:
            return None
        return UsageStats(
            key=row["key"],
            requests=row["requests"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            budget=row["budget"],
        )
