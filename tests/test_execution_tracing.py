from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path
from threading import Lock

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import GatewayStore, virtual_key_identifier
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import (
    ProviderCompletion,
    ProviderConfigurationError,
    ProviderOperationalError,
)
from app.skills import SkillLoader
from app.task_executor import TaskExecutor
from app.tracing import ExecutionAttempt


ProviderEvent = ProviderCompletion | Exception


class TraceProviderGateway:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderEvent]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.before_return: object | None = None
        self._lock = Lock()

    def queue(self, provider_name: str, *events: ProviderEvent) -> None:
        self.events[provider_name].extend(events)

    def complete_with_provider(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        with self._lock:
            self.calls.append((provider_name, [dict(item) for item in messages]))
            if not self.events[provider_name]:
                raise AssertionError(f"unexpected provider call: {provider_name}")
            event = self.events[provider_name].popleft()
        if callable(self.before_return):
            self.before_return()
        if isinstance(event, Exception):
            raise event
        return event

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        return _completion("fake-chat", "unchanged", 2, 1)


@pytest.fixture
def trace_api(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> Iterator[tuple[TestClient, TraceProviderGateway]]:
    provider = TraceProviderGateway()
    executor = TaskExecutor(
        SkillLoader(),
        PromptBuilder(),
        provider,  # type: ignore[arg-type]
        OutputValidator(),
    )
    monkeypatch.setattr(main_module, "store", test_store)
    monkeypatch.setattr(main_module, "providers", provider)
    monkeypatch.setattr(main_module, "task_executor", executor)
    with TestClient(main_module.app) as client:
        yield client, provider


def _completion(
    provider: str,
    content: str,
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> ProviderCompletion:
    return ProviderCompletion(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=provider,
    )


def _summary_json() -> str:
    return json.dumps(
        {
            "summary": "The release was approved.",
            "key_points": ["Friday is the release date."],
        }
    )


def _post_task(client: TestClient, key: str = "vk_open"):
    return client.post(
        "/v1/tasks/execute",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "skill": "summarize",
            "input": {"text": "The team approved the Friday release."},
        },
    )


def _get_trace(client: TestClient, task_id: str, key: str = "vk_open"):
    return client.get(
        f"/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {key}"},
    )


def _table_columns(store: GatewayStore, table: str) -> set[str]:
    connection = sqlite3.connect(store.database_path)
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        connection.close()
    return {str(row[1]) for row in rows}


def test_trace_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    store = GatewayStore(str(tmp_path / "idempotent.db"))
    store.initialize()
    store.initialize()

    connection = sqlite3.connect(store.database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        connection.close()

    assert {"task_executions", "task_attempts", "user_preferences"} <= tables
    assert "sqlite_autoindex_task_attempts_1" in indexes


def test_successful_task_persists_completed_trace(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue("groq", _completion("groq", _summary_json(), 7, 4))

    task = _post_task(client)
    trace = _get_trace(client, task.json()["task_id"])

    assert task.status_code == trace.status_code == 200
    assert trace.json() == {
        "task_id": task.json()["task_id"],
        "status": "completed",
        "skill": "summarize",
        "provider": "groq",
        "attempts": 1,
        "usage": {"prompt_tokens": 7, "completion_tokens": 4},
        "error_category": None,
        "created_at": trace.json()["created_at"],
        "completed_at": trace.json()["completed_at"],
        "attempt_history": [
            {
                "attempt_number": 1,
                "provider": "groq",
                "attempt_type": "initial",
                "status": "completed",
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
                "validation_error_category": None,
                "provider_error_category": None,
                "created_at": trace.json()["attempt_history"][0]["created_at"],
            }
        ],
    }
    assert trace.json()["completed_at"] is not None


def test_repaired_task_stores_two_ordered_attempts(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue(
        "groq",
        _completion("groq", "not json", 3, 1),
        _completion("groq", _summary_json(), 6, 4),
    )

    task = _post_task(client)
    body = _get_trace(client, task.json()["task_id"]).json()

    assert [item["attempt_number"] for item in body["attempt_history"]] == [1, 2]
    assert [item["attempt_type"] for item in body["attempt_history"]] == [
        "initial",
        "repair",
    ]
    assert body["attempt_history"][0]["status"] == "validation_error"
    assert body["attempt_history"][0]["validation_error_category"] == "parsing"
    assert body["attempt_history"][1]["status"] == "completed"
    assert body["usage"] == {"prompt_tokens": 9, "completion_tokens": 5}


def test_fallback_records_operational_failure_then_success(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue("groq", ProviderOperationalError("PRIVATE_FAILURE"))
    provider.queue("gemini", _completion("gemini", _summary_json(), 8, 5))

    task = _post_task(client)
    body = _get_trace(client, task.json()["task_id"]).json()

    assert [item["attempt_type"] for item in body["attempt_history"]] == [
        "initial",
        "fallback",
    ]
    assert body["attempt_history"][0]["status"] == "operational_error"
    assert body["attempt_history"][0]["provider_error_category"] == "operational"
    assert body["attempt_history"][0]["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    assert body["provider"] == "gemini"
    assert "PRIVATE_FAILURE" not in json.dumps(body)


def test_four_call_path_records_exact_attempt_types(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue(
        "groq",
        _completion("groq", "not json", 2, 1),
        ProviderOperationalError("repair unavailable"),
    )
    provider.queue(
        "gemini",
        _completion("gemini", "still not json", 3, 2),
        _completion("gemini", _summary_json(), 5, 3),
    )

    task = _post_task(client)
    body = _get_trace(client, task.json()["task_id"]).json()

    assert body["attempts"] == 4
    assert [item["attempt_type"] for item in body["attempt_history"]] == [
        "initial",
        "repair",
        "fallback",
        "fallback_repair",
    ]
    assert len(provider.calls) == 4


def test_invalid_after_repair_stores_failed_trace(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = trace_api
    provider.queue(
        "groq",
        _completion("groq", "FULL_RAW_OUTPUT_ONE", 3, 2),
        _completion("groq", "FULL_RAW_OUTPUT_TWO", 4, 1),
    )

    response = _post_task(client)
    connection = sqlite3.connect(test_store.database_path)
    try:
        task_id = connection.execute(
            "SELECT task_id FROM task_executions"
        ).fetchone()[0]
    finally:
        connection.close()
    body = _get_trace(client, task_id).json()

    assert response.status_code == 502
    assert body["status"] == "failed"
    assert body["error_category"] == "validation"
    assert body["attempts"] == 2
    assert body["completed_at"] is not None
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}


@pytest.mark.parametrize(
    ("events", "expected_category", "expected_attempts"),
    [
        (
            [
                ("groq", ProviderOperationalError("primary")),
                ("gemini", ProviderOperationalError("fallback")),
            ],
            "operational",
            2,
        ),
        (
            [("groq", ProviderConfigurationError("SECRET_CONFIG"))],
            "configuration",
            1,
        ),
    ],
)
def test_provider_failures_store_safe_terminal_categories(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
    events: list[tuple[str, Exception]],
    expected_category: str,
    expected_attempts: int,
) -> None:
    client, provider = trace_api
    for provider_name, event in events:
        provider.queue(provider_name, event)

    response = _post_task(client)
    connection = sqlite3.connect(test_store.database_path)
    try:
        task_id = connection.execute(
            "SELECT task_id FROM task_executions"
        ).fetchone()[0]
    finally:
        connection.close()
    body = _get_trace(client, task_id).json()

    assert response.status_code in {500, 502}
    assert body["status"] == "failed"
    assert body["error_category"] == expected_category
    assert body["attempts"] == expected_attempts
    assert "SECRET_CONFIG" not in json.dumps(body)


def test_running_trace_exists_while_provider_is_active(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = trace_api
    observed: list[tuple[str, int]] = []

    def observe() -> None:
        connection = sqlite3.connect(test_store.database_path)
        try:
            row = connection.execute(
                "SELECT status, attempts FROM task_executions"
            ).fetchone()
        finally:
            connection.close()
        observed.append((row[0], row[1]))

    provider.before_return = observe
    provider.queue("groq", _completion("groq", _summary_json()))

    assert _post_task(client).status_code == 200
    assert observed == [("running", 0)]


def test_duplicate_attempt_number_is_rejected(test_store: GatewayStore) -> None:
    task_id = "duplicate-attempt-test"
    test_store.create_task_execution(
        task_id,
        virtual_key_identifier("vk_open"),
        "summarize",
    )
    attempt = ExecutionAttempt(1, "groq", "initial", "completed", 2, 1)
    test_store.append_task_attempt(task_id, attempt)

    with pytest.raises(sqlite3.IntegrityError):
        test_store.append_task_attempt(task_id, attempt)


def test_trace_owner_is_enforced_without_existence_disclosure(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue("groq", _completion("groq", _summary_json()))
    task_id = _post_task(client).json()["task_id"]

    assert _get_trace(client, task_id, "vk_tiny").status_code == 404
    assert _get_trace(client, "unknown-task").status_code == 404


@pytest.mark.parametrize("authorization", [None, "vk_open", "Basic vk_open", "Bearer"])
def test_trace_endpoint_requires_valid_bearer_authentication(
    trace_api: tuple[TestClient, TraceProviderGateway],
    authorization: str | None,
) -> None:
    client, _ = trace_api
    headers = {"Authorization": authorization} if authorization else {}

    response = client.get("/v1/tasks/unknown", headers=headers)

    assert response.status_code == 401


def test_trace_tables_exclude_prompts_outputs_and_plaintext_keys(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = trace_api
    raw_secret = "RAW_PROVIDER_RESPONSE_SENTINEL"
    provider.queue(
        "groq",
        _completion("groq", raw_secret),
        _completion("groq", _summary_json()),
    )
    _post_task(client)

    execution_columns = _table_columns(test_store, "task_executions")
    attempt_columns = _table_columns(test_store, "task_attempts")
    forbidden_columns = {"prompt", "input", "output", "raw_output", "virtual_key"}
    assert forbidden_columns.isdisjoint(execution_columns | attempt_columns)

    connection = sqlite3.connect(test_store.database_path)
    try:
        execution_values = connection.execute(
            "SELECT * FROM task_executions"
        ).fetchall()
        attempt_values = connection.execute("SELECT * FROM task_attempts").fetchall()
    finally:
        connection.close()
    persisted = repr(execution_values) + repr(attempt_values)
    assert raw_secret not in persisted
    assert "vk_open" not in persisted
    assert "Authorization" not in persisted


def test_trace_usage_matches_usage_accounting(
    trace_api: tuple[TestClient, TraceProviderGateway],
) -> None:
    client, provider = trace_api
    provider.queue(
        "groq",
        _completion("groq", "not json", 3, 1),
        _completion("groq", _summary_json(), 6, 4),
    )
    task = _post_task(client)

    trace_usage = _get_trace(client, task.json()["task_id"]).json()["usage"]
    gateway_usage = client.get("/usage", params={"key": "vk_open"}).json()

    assert trace_usage == {"prompt_tokens": 9, "completion_tokens": 5}
    assert gateway_usage["tokens_in"] == trace_usage["prompt_tokens"]
    assert gateway_usage["tokens_out"] == trace_usage["completion_tokens"]
    assert gateway_usage["requests"] == 1


def test_trace_creation_failure_releases_before_provider_work(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = trace_api

    def fail_creation(*args: object) -> None:
        raise sqlite3.OperationalError("PRIVATE DATABASE PATH")

    monkeypatch.setattr(test_store, "create_task_execution", fail_creation)

    response = _post_task(client, "vk_edge")

    assert response.status_code == 500
    assert response.json() == {"detail": "task tracing failed"}
    assert provider.calls == []
    stats = test_store.get_usage("vk_edge")
    assert stats is not None and stats.requests == 0
    assert "PRIVATE DATABASE PATH" not in response.text


def test_attempt_recording_failure_stops_and_safely_fails_trace(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = trace_api
    provider.queue("groq", _completion("groq", _summary_json(), 5, 3))

    def fail_attempt(*args: object) -> None:
        raise sqlite3.OperationalError("SQL SECRET")

    monkeypatch.setattr(test_store, "append_task_attempt", fail_attempt)

    response = _post_task(client)

    assert response.status_code == 500
    assert response.json() == {"detail": "task tracing failed"}
    assert len(provider.calls) == 1
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 5, 3)
    connection = sqlite3.connect(test_store.database_path)
    try:
        row = connection.execute(
            "SELECT status, error_category FROM task_executions"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("failed", "trace_recording")


def test_terminal_accounting_failure_never_marks_trace_completed(
    trace_api: tuple[TestClient, TraceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = trace_api
    provider.queue("groq", _completion("groq", _summary_json()))

    def fail_finalization(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("controlled")

    monkeypatch.setattr(test_store, "record_usage_events", fail_finalization)

    response = _post_task(client)

    assert response.status_code == 500
    connection = sqlite3.connect(test_store.database_path)
    try:
        status = connection.execute(
            "SELECT status FROM task_executions"
        ).fetchone()[0]
        usage_events = connection.execute(
            "SELECT COUNT(*) FROM usage_events"
        ).fetchone()[0]
    finally:
        connection.close()
    assert status == "running"
    assert usage_events == 0
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 1
