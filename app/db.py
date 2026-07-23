from __future__ import annotations

import sqlite3
from hashlib import sha256
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=5,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        database = Path(self.database_path)
        database.parent.mkdir(parents=True, exist_ok=True)

        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS virtual_keys (
                    key TEXT PRIMARY KEY,
                    budget INTEGER NOT NULL CHECK (budget >= 0),
                    requests INTEGER NOT NULL DEFAULT 0 CHECK (requests >= 0),
                    tokens_in INTEGER NOT NULL DEFAULT 0 CHECK (tokens_in >= 0),
                    tokens_out INTEGER NOT NULL DEFAULT 0 CHECK (tokens_out >= 0)
                );

                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    virtual_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL CHECK (tokens_in >= 0),
                    tokens_out INTEGER NOT NULL CHECK (tokens_out >= 0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (virtual_key) REFERENCES virtual_keys(key)
                );

                CREATE TABLE IF NOT EXISTS task_executions (
                    task_id TEXT PRIMARY KEY,
                    virtual_key_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'completed', 'failed')
                    ),
                    final_provider TEXT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
                    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        completion_tokens >= 0
                    ),
                    error_category TEXT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    completed_at TEXT NULL
                );

                CREATE TABLE IF NOT EXISTS task_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                    provider TEXT NOT NULL,
                    attempt_type TEXT NOT NULL CHECK (
                        attempt_type IN (
                            'initial', 'repair', 'fallback', 'fallback_repair',
                            'post_tool', 'post_tool_repair',
                            'post_tool_fallback', 'post_tool_fallback_repair'
                        )
                    ),
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'completed', 'validation_error',
                            'operational_error', 'configuration_error'
                        )
                    ),
                    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        prompt_tokens >= 0
                    ),
                    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        completion_tokens >= 0
                    ),
                    validation_error_category TEXT NULL CHECK (
                        validation_error_category IS NULL OR
                        validation_error_category IN (
                            'parsing', 'structure', 'semantic', 'tool_protocol'
                        )
                    ),
                    provider_error_category TEXT NULL CHECK (
                        provider_error_category IS NULL OR
                        provider_error_category IN ('operational', 'configuration')
                    ),
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    FOREIGN KEY (task_id) REFERENCES task_executions(task_id),
                    UNIQUE (task_id, attempt_number)
                );

                CREATE INDEX IF NOT EXISTS idx_task_executions_owner
                ON task_executions (virtual_key_id, task_id);

                CREATE INDEX IF NOT EXISTS idx_task_attempts_task
                ON task_attempts (task_id, attempt_number);

                CREATE TABLE IF NOT EXISTS user_preferences (
                    virtual_key_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    PRIMARY KEY (virtual_key_id, preference_key)
                );

                CREATE TABLE IF NOT EXISTS task_tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    tool_number INTEGER NOT NULL CHECK (tool_number = 1),
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'completed', 'failed')
                    ),
                    error_category TEXT NULL,
                    duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    completed_at TEXT NULL,
                    FOREIGN KEY (task_id) REFERENCES task_executions(task_id),
                    UNIQUE (task_id, tool_number)
                );

                CREATE INDEX IF NOT EXISTS idx_task_tools_task
                ON task_tool_executions (task_id, tool_number);

                CREATE TABLE IF NOT EXISTS workflow_executions (
                    workflow_id TEXT PRIMARY KEY,
                    virtual_key_id TEXT NOT NULL,
                    definition_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'completed', 'failed')
                    ),
                    step_count INTEGER NOT NULL CHECK (step_count > 0),
                    completed_steps INTEGER NOT NULL DEFAULT 0 CHECK (
                        completed_steps >= 0 AND completed_steps <= step_count
                    ),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    tool_count INTEGER NOT NULL DEFAULT 0 CHECK (tool_count >= 0),
                    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        prompt_tokens >= 0
                    ),
                    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        completion_tokens >= 0
                    ),
                    error_category TEXT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    completed_at TEXT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    step_order INTEGER NOT NULL CHECK (step_order > 0),
                    step_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    task_id TEXT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'pending', 'running', 'completed', 'failed', 'skipped'
                        )
                    ),
                    provider TEXT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    tool_count INTEGER NOT NULL DEFAULT 0 CHECK (tool_count >= 0),
                    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        prompt_tokens >= 0
                    ),
                    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        completion_tokens >= 0
                    ),
                    error_category TEXT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    started_at TEXT NULL,
                    completed_at TEXT NULL,
                    FOREIGN KEY (workflow_id)
                        REFERENCES workflow_executions(workflow_id),
                    FOREIGN KEY (task_id) REFERENCES task_executions(task_id),
                    UNIQUE (workflow_id, step_order),
                    UNIQUE (workflow_id, step_id)
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_executions_owner
                ON workflow_executions (virtual_key_id, workflow_id);

                CREATE INDEX IF NOT EXISTS idx_workflow_steps_execution
                ON workflow_steps (workflow_id, step_order);
                """
            )
            self._upgrade_task_attempt_schema(connection)
            for key, budget in SEEDED_KEYS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO virtual_keys (key, budget) VALUES (?, ?)",
                    (key, budget),
                )
        finally:
            connection.close()

    def reserve_request(self, key: str) -> ReservationResult:
        """Atomically admit one request or reject it before provider work begins."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT requests, budget FROM virtual_keys WHERE key = ?",
                (key,),
            ).fetchone()

            if row is None:
                connection.rollback()
                return "unknown"
            if row["requests"] >= row["budget"]:
                connection.rollback()
                return "over_budget"

            connection.execute(
                "UPDATE virtual_keys SET requests = requests + 1 WHERE key = ?",
                (key,),
            )
            connection.commit()
            return "reserved"
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release_request(self, key: str) -> None:
        """Undo a reservation when no provider produced a billable response."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE virtual_keys
                SET requests = CASE WHEN requests > 0 THEN requests - 1 ELSE 0 END
                WHERE key = ?
                """,
                (key,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

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
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if events:
                connection.execute(
                    """
                    UPDATE virtual_keys
                    SET tokens_in = tokens_in + ?, tokens_out = tokens_out + ?
                    WHERE key = ?
                    """,
                    (token_input_total, token_output_total, key),
                )
                connection.executemany(
                    """
                    INSERT INTO usage_events
                        (virtual_key, provider, tokens_in, tokens_out)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (key, provider, tokens_in, tokens_out)
                        for provider, tokens_in, tokens_out in events
                    ],
                )

            if task_id is not None and trace_status is not None:
                cursor = connection.execute(
                    """
                    UPDATE task_executions
                    SET status = ?, final_provider = ?, attempts = ?,
                        prompt_tokens = ?, completion_tokens = ?,
                        error_category = ?,
                        completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE task_id = ? AND virtual_key_id = ? AND status = 'running'
                    """,
                    (
                        trace_status,
                        final_provider,
                        attempts,
                        token_input_total,
                        token_output_total,
                        error_category,
                        task_id,
                        virtual_key_identifier(key),
                    ),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.IntegrityError("task trace could not be finalized")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_task_execution(
        self,
        task_id: str,
        virtual_key_id: str,
        skill: str,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO task_executions
                    (task_id, virtual_key_id, skill, status)
                VALUES (?, ?, ?, 'running')
                """,
                (task_id, virtual_key_id, skill),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _upgrade_task_attempt_schema(connection: sqlite3.Connection) -> None:
        """Idempotently widen the Phase 3 CHECK constraint for bounded tool phases."""
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("task_attempts",),
        ).fetchone()
        if row is None or "post_tool" in row["sql"]:
            return

        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "ALTER TABLE task_attempts RENAME TO task_attempts_phase3"
            )
            connection.execute("DROP INDEX IF EXISTS idx_task_attempts_task")
            connection.execute(
                """
                CREATE TABLE task_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                    provider TEXT NOT NULL,
                    attempt_type TEXT NOT NULL CHECK (
                        attempt_type IN (
                            'initial', 'repair', 'fallback', 'fallback_repair',
                            'post_tool', 'post_tool_repair',
                            'post_tool_fallback', 'post_tool_fallback_repair'
                        )
                    ),
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'completed', 'validation_error',
                            'operational_error', 'configuration_error'
                        )
                    ),
                    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        prompt_tokens >= 0
                    ),
                    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
                        completion_tokens >= 0
                    ),
                    validation_error_category TEXT NULL CHECK (
                        validation_error_category IS NULL OR
                        validation_error_category IN (
                            'parsing', 'structure', 'semantic', 'tool_protocol'
                        )
                    ),
                    provider_error_category TEXT NULL CHECK (
                        provider_error_category IS NULL OR
                        provider_error_category IN ('operational', 'configuration')
                    ),
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    FOREIGN KEY (task_id) REFERENCES task_executions(task_id),
                    UNIQUE (task_id, attempt_number)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO task_attempts (
                    id, task_id, attempt_number, provider, attempt_type, status,
                    prompt_tokens, completion_tokens,
                    validation_error_category, provider_error_category, created_at
                )
                SELECT id, task_id, attempt_number, provider, attempt_type, status,
                       prompt_tokens, completion_tokens,
                       validation_error_category, provider_error_category, created_at
                FROM task_attempts_phase3
                """
            )
            connection.execute("DROP TABLE task_attempts_phase3")
            connection.execute(
                """
                CREATE INDEX idx_task_attempts_task
                ON task_attempts (task_id, attempt_number)
                """
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def append_task_attempt(
        self,
        task_id: str,
        attempt: ExecutionAttempt,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO task_attempts (
                    task_id, attempt_number, provider, attempt_type, status,
                    prompt_tokens, completion_tokens,
                    validation_error_category, provider_error_category
                )
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
                FROM task_executions
                WHERE task_id = ? AND status = 'running'
                """,
                (
                    task_id,
                    attempt.attempt_number,
                    attempt.provider,
                    attempt.attempt_type,
                    attempt.status,
                    attempt.prompt_tokens,
                    attempt.completion_tokens,
                    attempt.validation_error_category,
                    attempt.provider_error_category,
                    task_id,
                ),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("task attempt could not be recorded")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        tool_name: str,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO task_tool_executions (
                    task_id, tool_number, tool_name, status
                )
                SELECT ?, ?, ?, 'running'
                FROM task_executions
                WHERE task_id = ? AND status = 'running'
                """,
                (task_id, tool_number, tool_name, task_id),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("tool execution could not be recorded")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finalize_task_tool_execution(
        self,
        task_id: str,
        tool_number: int,
        status: Literal["completed", "failed"],
        error_category: str | None,
        duration_ms: int,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE task_tool_executions
                SET status = ?, error_category = ?, duration_ms = ?,
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE task_id = ? AND tool_number = ? AND status = 'running'
                """,
                (status, error_category, duration_ms, task_id, tool_number),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("tool execution could not be finalized")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finalize_failed_task_without_usage(
        self,
        key: str,
        task_id: str,
        attempts: int,
        error_category: str,
    ) -> None:
        """Atomically release an unbilled reservation and fail its running trace."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE task_executions
                SET status = 'failed', attempts = ?, error_category = ?,
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE task_id = ? AND virtual_key_id = ? AND status = 'running'
                """,
                (
                    attempts,
                    error_category,
                    task_id,
                    virtual_key_identifier(key),
                ),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("task trace could not be finalized")
            connection.execute(
                """
                UPDATE virtual_keys
                SET requests = CASE WHEN requests > 0 THEN requests - 1 ELSE 0 END
                WHERE key = ?
                """,
                (key,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_task_execution(
        self,
        task_id: str,
        virtual_key_id: str,
    ) -> StoredTaskTrace | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT task_id, status, skill, final_provider, attempts,
                       prompt_tokens, completion_tokens, error_category,
                       created_at, completed_at
                FROM task_executions
                WHERE task_id = ? AND virtual_key_id = ?
                """,
                (task_id, virtual_key_id),
            ).fetchone()
            if row is None:
                return None
            attempt_rows = connection.execute(
                """
                SELECT attempt_number, provider, attempt_type, status,
                       prompt_tokens, completion_tokens,
                       validation_error_category, provider_error_category,
                       created_at
                FROM task_attempts
                WHERE task_id = ?
                ORDER BY attempt_number ASC
                """,
                (task_id,),
            ).fetchall()
            tool_rows = connection.execute(
                """
                SELECT tool_number, tool_name, status, error_category,
                       duration_ms, created_at, completed_at
                FROM task_tool_executions
                WHERE task_id = ?
                ORDER BY tool_number ASC
                """,
                (task_id,),
            ).fetchall()
        finally:
            connection.close()

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
                created_at=item["created_at"],
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
                created_at=item["created_at"],
                completed_at=item["completed_at"],
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
            created_at=row["created_at"],
            completed_at=row["completed_at"],
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
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO workflow_executions (
                    workflow_id, virtual_key_id, definition_id, name,
                    description, status, step_count
                )
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    workflow_id,
                    virtual_key_id,
                    definition_id,
                    name,
                    description,
                    len(steps),
                ),
            )
            connection.executemany(
                """
                INSERT INTO workflow_steps (
                    workflow_id, step_order, step_id, name, skill, status
                )
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                [
                    (
                        workflow_id,
                        step_order,
                        step_id,
                        step_name,
                        skill,
                    )
                    for step_order, step_id, step_name, skill in steps
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def start_workflow_step(
        self,
        workflow_id: str,
        step_order: int,
        task_id: str,
        skill: str,
    ) -> None:
        """Atomically mark one declared step running and create its task trace."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT workflow_executions.virtual_key_id
                FROM workflow_executions
                JOIN workflow_steps
                  ON workflow_steps.workflow_id = workflow_executions.workflow_id
                WHERE workflow_executions.workflow_id = ?
                  AND workflow_executions.status = 'running'
                  AND workflow_steps.step_order = ?
                  AND workflow_steps.skill = ?
                  AND workflow_steps.status = 'pending'
                """,
                (workflow_id, step_order, skill),
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("workflow is not running")

            connection.execute(
                """
                INSERT INTO task_executions (
                    task_id, virtual_key_id, skill, status
                )
                VALUES (?, ?, ?, 'running')
                """,
                (task_id, row["virtual_key_id"], skill),
            )
            cursor = connection.execute(
                """
                UPDATE workflow_steps
                SET status = 'running', task_id = ?,
                    started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE workflow_id = ? AND step_order = ?
                  AND skill = ? AND status = 'pending'
                """,
                (task_id, workflow_id, step_order, skill),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("workflow step could not be started")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

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

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            owner_id = virtual_key_identifier(key)
            workflow_row = connection.execute(
                """
                SELECT step_count
                FROM workflow_executions
                WHERE workflow_id = ? AND virtual_key_id = ? AND status = 'running'
                """,
                (workflow_id, owner_id),
            ).fetchone()
            if workflow_row is None:
                raise sqlite3.IntegrityError("workflow trace could not be finalized")

            if events:
                connection.execute(
                    """
                    UPDATE virtual_keys
                    SET tokens_in = tokens_in + ?, tokens_out = tokens_out + ?
                    WHERE key = ?
                    """,
                    (prompt_total, completion_total, key),
                )
                connection.executemany(
                    """
                    INSERT INTO usage_events (
                        virtual_key, provider, tokens_in, tokens_out
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (key, provider, tokens_in, tokens_out)
                        for provider, tokens_in, tokens_out in events
                    ],
                )

            for step in steps:
                cursor = connection.execute(
                    """
                    UPDATE workflow_steps
                    SET status = ?, provider = ?, attempts = ?, tool_count = ?,
                        prompt_tokens = ?, completion_tokens = ?,
                        error_category = ?,
                        completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE workflow_id = ? AND step_order = ?
                      AND status IN ('pending', 'running')
                    """,
                    (
                        step.status,
                        step.provider,
                        step.attempts,
                        step.tool_count,
                        step.prompt_tokens,
                        step.completion_tokens,
                        step.error_category,
                        workflow_id,
                        step.step_order,
                    ),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "workflow step could not be finalized"
                    )

                if step.task_id is not None:
                    task_cursor = connection.execute(
                        """
                        UPDATE task_executions
                        SET status = ?, final_provider = ?, attempts = ?,
                            prompt_tokens = ?, completion_tokens = ?,
                            error_category = ?,
                            completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE task_id = ? AND virtual_key_id = ?
                          AND status = 'running'
                        """,
                        (
                            step.status,
                            step.provider,
                            step.attempts,
                            step.prompt_tokens,
                            step.completion_tokens,
                            step.error_category,
                            step.task_id,
                            owner_id,
                        ),
                    )
                    if task_cursor.rowcount != 1:
                        raise sqlite3.IntegrityError(
                            "workflow task trace could not be finalized"
                        )

            if status == "failed":
                connection.execute(
                    """
                    UPDATE workflow_steps
                    SET status = 'skipped',
                        completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE workflow_id = ? AND status = 'pending'
                    """,
                    (workflow_id,),
                )

            attempts = sum(step.attempts for step in steps)
            tool_count = sum(step.tool_count for step in steps)
            completed_steps = sum(step.status == "completed" for step in steps)
            workflow_cursor = connection.execute(
                """
                UPDATE workflow_executions
                SET status = ?, completed_steps = ?, attempts = ?,
                    tool_count = ?, prompt_tokens = ?, completion_tokens = ?,
                    error_category = ?,
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE workflow_id = ? AND virtual_key_id = ? AND status = 'running'
                """,
                (
                    status,
                    completed_steps,
                    attempts,
                    tool_count,
                    prompt_total,
                    completion_total,
                    error_category,
                    workflow_id,
                    owner_id,
                ),
            )
            if workflow_cursor.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "workflow trace could not be finalized"
                )

            if not events:
                connection.execute(
                    """
                    UPDATE virtual_keys
                    SET requests = CASE
                        WHEN requests > 0 THEN requests - 1 ELSE 0
                    END
                    WHERE key = ?
                    """,
                    (key,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_workflow_execution(
        self,
        workflow_id: str,
        virtual_key_id: str,
    ) -> StoredWorkflowTrace | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT workflow_id, definition_id, name, description, status,
                       step_count, completed_steps, attempts, tool_count,
                       prompt_tokens, completion_tokens, error_category,
                       created_at, completed_at
                FROM workflow_executions
                WHERE workflow_id = ? AND virtual_key_id = ?
                """,
                (workflow_id, virtual_key_id),
            ).fetchone()
            if row is None:
                return None
            step_rows = connection.execute(
                """
                SELECT step_order, step_id, name, skill, status, provider,
                       attempts, tool_count, prompt_tokens, completion_tokens,
                       error_category, created_at, started_at, completed_at
                FROM workflow_steps
                WHERE workflow_id = ?
                ORDER BY step_order ASC
                """,
                (workflow_id,),
            ).fetchall()
        finally:
            connection.close()

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
                created_at=item["created_at"],
                started_at=item["started_at"],
                completed_at=item["completed_at"],
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
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            steps=steps,
        )

    def get_preference_values(self, virtual_key_id: str) -> dict[str, str]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT preference_key, preference_value_json
                FROM user_preferences
                WHERE virtual_key_id = ?
                ORDER BY preference_key ASC
                """,
                (virtual_key_id,),
            ).fetchall()
        finally:
            connection.close()
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
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO user_preferences (
                    virtual_key_id, preference_key, preference_value_json
                )
                VALUES (?, ?, ?)
                ON CONFLICT (virtual_key_id, preference_key) DO UPDATE SET
                    preference_value_json = excluded.preference_value_json,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                [
                    (virtual_key_id, name, value)
                    for name, value in values.items()
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def delete_preference_value(
        self,
        virtual_key_id: str,
        preference_key: str,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM user_preferences
                WHERE virtual_key_id = ? AND preference_key = ?
                """,
                (virtual_key_id, preference_key),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_usage(self, key: str) -> UsageStats | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT key, requests, tokens_in, tokens_out, budget
                FROM virtual_keys
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            return None
        return UsageStats(
            key=row["key"],
            requests=row["requests"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            budget=row["budget"],
        )
