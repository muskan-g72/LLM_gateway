from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import GatewayStore
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import ProviderCompletion, ProviderOperationalError
from app.skills import SkillLoader
from app.task_executor import TaskExecutor
from app.workflow_executor import WorkflowExecutor


ProviderEvent = ProviderCompletion | Exception


class EndpointWorkflowProvider:
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

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        return ProviderCompletion("unchanged chat", 5, 3, "fake-chat")


@pytest.fixture
def workflow_api(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> Iterator[tuple[TestClient, EndpointWorkflowProvider]]:
    provider = EndpointWorkflowProvider()
    task_executor = TaskExecutor(
        SkillLoader(),
        PromptBuilder(),
        provider,  # type: ignore[arg-type]
        OutputValidator(),
    )
    workflow_executor = WorkflowExecutor(task_executor)
    monkeypatch.setattr(main_module, "store", test_store)
    monkeypatch.setattr(main_module, "providers", provider)
    monkeypatch.setattr(main_module, "task_executor", task_executor)
    monkeypatch.setattr(main_module, "workflow_executor", workflow_executor)
    with TestClient(main_module.app) as client:
        yield client, provider


def _completion(
    content: str,
    provider: str = "groq",
    prompt_tokens: int = 2,
    completion_tokens: int = 1,
) -> ProviderCompletion:
    return ProviderCompletion(
        content,
        prompt_tokens,
        completion_tokens,
        provider,
    )


def _summary(text: str = "The release notes are due Friday.") -> str:
    return json.dumps(
        {
            "summary": text,
            "key_points": ["Maya owns the Friday release notes."],
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


def _queue_success(
    provider: EndpointWorkflowProvider,
    *,
    tool: bool = True,
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


def _post(
    client: TestClient,
    *,
    key: str = "vk_open",
    workflow: str = "article_processing",
    task_input: dict[str, object] | None = None,
    preferences: dict[str, object] | None = None,
):
    body: dict[str, object] = {
        "workflow": workflow,
        "input": task_input
        if task_input is not None
        else {
            "text": (
                "The team approved the release. "
                "Maya must publish the release notes by Friday."
            )
        },
    }
    if preferences is not None:
        body["preferences"] = preferences
    return client.post(
        "/v1/workflows/execute",
        headers={"Authorization": f"Bearer {key}"},
        json=body,
    )


def _get_trace(client: TestClient, workflow_id: str, key: str = "vk_open"):
    return client.get(
        f"/v1/workflows/{workflow_id}",
        headers={"Authorization": f"Bearer {key}"},
    )


def _only_workflow_id(store: GatewayStore) -> str:
    connection = sqlite3.connect(store.database_path)
    try:
        row = connection.execute(
            "SELECT workflow_id FROM workflow_executions LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return str(row[0])


def _usage_event_count(store: GatewayStore) -> int:
    connection = sqlite3.connect(store.database_path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0])
    finally:
        connection.close()


def test_workflow_success_response_and_tool_contract(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, provider = workflow_api
    _queue_success(provider)

    response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "workflow_id": body["workflow_id"],
        "status": "completed",
        "workflow": "article_processing",
        "steps": [
            {
                "step_order": 1,
                "step_id": "summary",
                "name": "Summarize article",
                "skill": "summarize",
                "status": "completed",
                "provider": "groq",
                "attempts": 1,
                "tool_count": 0,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            {
                "step_order": 2,
                "step_id": "action_items",
                "name": "Extract action items",
                "skill": "extract_action_items",
                "status": "completed",
                "provider": "groq",
                "attempts": 1,
                "tool_count": 0,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            {
                "step_order": 3,
                "step_id": "final_report",
                "name": "Generate statistics-assisted report",
                "skill": "summarize",
                "status": "completed",
                "provider": "groq",
                "attempts": 2,
                "tool_count": 1,
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ],
        "output": {
            "summary": "The final release report includes statistics.",
            "key_points": ["Maya owns the Friday release notes."],
        },
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
    }


def test_workflow_trace_is_persistent_ordered_and_owned(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, provider = workflow_api
    _queue_success(provider)
    workflow_id = _post(client).json()["workflow_id"]

    response = _get_trace(client, workflow_id)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["workflow"] == "article_processing"
    assert body["step_count"] == 3
    assert body["completed_steps"] == 3
    assert body["attempts"] == 4
    assert body["tool_count"] == 1
    assert body["usage"] == {"prompt_tokens": 8, "completion_tokens": 4}
    assert [step["step_order"] for step in body["steps"]] == [1, 2, 3]
    assert all(step["completed_at"] is not None for step in body["steps"])
    assert _get_trace(client, workflow_id, "vk_tiny").status_code == 404


def test_each_workflow_step_retains_its_normal_task_trace(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    _queue_success(provider)
    assert _post(client).status_code == 200

    connection = sqlite3.connect(test_store.database_path)
    try:
        rows = connection.execute(
            """
            SELECT task_id
            FROM workflow_steps
            WHERE task_id IS NOT NULL
            ORDER BY step_order
            """
        ).fetchall()
    finally:
        connection.close()

    traces = [
        client.get(
            f"/v1/tasks/{row[0]}",
            headers={"Authorization": "Bearer vk_open"},
        ).json()
        for row in rows
    ]
    assert [trace["status"] for trace in traces] == [
        "completed",
        "completed",
        "completed",
    ]
    assert [trace["attempts"] for trace in traces] == [1, 1, 2]


def test_one_workflow_reservation_accounts_every_provider_completion(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    _queue_success(provider)

    assert _post(client).status_code == 200

    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 8, 4)
    assert _usage_event_count(test_store) == 4


def test_workflow_budget_is_one_unit_not_one_per_step(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    _queue_success(provider, tool=False)

    first = _post(client, key="vk_edge")
    second = _post(client, key="vk_edge")

    assert first.status_code == 200
    assert second.status_code == 429
    assert len(provider.calls) == 3
    stats = test_store.get_usage("vk_edge")
    assert stats is not None and stats.requests == 1


def test_parallel_vk_edge_workflows_admit_exactly_one(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    _, provider = workflow_api
    _queue_success(provider, tool=False)
    contenders = 10
    start_together = Barrier(contenders)
    request = main_module.WorkflowRequest.model_validate(
        {
            "workflow": "article_processing",
            "input": {
                "text": (
                    "The team approved the release. "
                    "Maya must publish the release notes by Friday."
                )
            },
        }
    )

    def execute() -> int:
        start_together.wait()
        try:
            main_module.execute_workflow(request, "Bearer vk_edge")
        except HTTPException as error:
            return error.status_code
        return 200

    with ThreadPoolExecutor(max_workers=contenders) as executor:
        statuses = list(executor.map(lambda _: execute(), range(contenders)))

    assert Counter(statuses) == Counter({429: 9, 200: 1})
    assert len(provider.calls) == 3


def test_unknown_workflow_and_invalid_input_do_not_consume_budget(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api

    assert _post(client, workflow="unknown").status_code == 404
    assert _post(client, task_input={}).status_code == 422

    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0
    assert provider.calls == []


@pytest.mark.parametrize(
    "authorization",
    [None, "vk_open", "Basic vk_open", "Bearer"],
)
def test_workflow_endpoints_require_existing_bearer_authentication(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    authorization: str | None,
) -> None:
    client, _ = workflow_api
    headers = {"Authorization": authorization} if authorization is not None else {}

    post = client.post(
        "/v1/workflows/execute",
        headers=headers,
        json={"workflow": "article_processing", "input": {"text": "release"}},
    )
    get = client.get("/v1/workflows/not-found", headers=headers)

    assert post.status_code == 401
    assert get.status_code == 401


def test_unknown_workflow_trace_returns_404(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, _ = workflow_api

    assert _get_trace(client, "not-found").status_code == 404


def test_unexpected_request_fields_are_rejected(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, _ = workflow_api
    response = client.post(
        "/v1/workflows/execute",
        headers={"Authorization": "Bearer vk_open"},
        json={
            "workflow": "article_processing",
            "input": {"text": "release"},
            "steps": [{"arbitrary": "model plan"}],
        },
    )

    assert response.status_code == 422


def test_stored_and_request_preferences_are_merged_for_all_steps(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, provider = workflow_api
    stored = client.put(
        "/v1/preferences",
        headers={"Authorization": "Bearer vk_open"},
        json={
            "preferences": {
                "preferred_language": "English",
                "response_detail": "concise",
            }
        },
    )
    assert stored.status_code == 200
    _queue_success(provider, tool=False)

    response = _post(
        client,
        preferences={"response_detail": "detailed"},
    )

    assert response.status_code == 200
    assert all(
        '"preferred_language":"English"' in messages[-1]["content"]
        and '"response_detail":"detailed"' in messages[-1]["content"]
        for _, messages in provider.calls
    )
    persisted = client.get(
        "/v1/preferences",
        headers={"Authorization": "Bearer vk_open"},
    ).json()
    assert persisted["preferences"]["response_detail"] == "concise"


def test_failed_step_is_recorded_and_later_steps_are_skipped(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    raw_secret = "RAW_PROVIDER_SECRET_C:\\private\\path"
    provider.queue(
        "groq",
        _completion(_summary()),
        _completion(raw_secret),
        _completion(raw_secret),
    )

    response = _post(client)
    workflow_id = _only_workflow_id(test_store)
    trace = _get_trace(client, workflow_id).json()

    assert response.status_code == 502
    assert response.json() == {"detail": "workflow step failed"}
    assert raw_secret not in response.text
    assert trace["status"] == "failed"
    assert [step["status"] for step in trace["steps"]] == [
        "completed",
        "failed",
        "skipped",
    ]
    assert len(provider.calls) == 3
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 6, 3)


def test_all_provider_failure_before_completion_releases_workflow_reservation(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    provider.queue("groq", ProviderOperationalError("primary"))
    provider.queue("gemini", ProviderOperationalError("fallback"))

    response = _post(client, key="vk_edge")
    trace = _get_trace(
        client,
        _only_workflow_id(test_store),
        "vk_edge",
    ).json()

    assert response.status_code == 502
    assert trace["status"] == "failed"
    assert [step["status"] for step in trace["steps"]] == [
        "failed",
        "skipped",
        "skipped",
    ]
    stats = test_store.get_usage("vk_edge")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (0, 0, 0)


def test_repair_fallback_and_tool_usage_are_all_aggregated(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    provider.queue(
        "groq",
        _completion("not json"),
        _completion(_summary()),
        ProviderOperationalError("step two primary"),
        _completion(_tool_call()),
        _completion(_summary("The final release report includes statistics.")),
    )
    provider.queue("gemini", _completion(_actions(), provider="gemini"))

    response = _post(client)

    assert response.status_code == 200
    assert [step["attempts"] for step in response.json()["steps"]] == [2, 2, 2]
    assert response.json()["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    assert _usage_event_count(test_store) == 5
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 10, 5)


def test_workflow_tables_store_no_input_output_prompt_or_credentials(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = workflow_api
    secret = "UNPERSISTED_WORKFLOW_INPUT_SECRET"
    _queue_success(provider, tool=False)
    assert _post(
        client,
        task_input={
            "text": (
                f"The release contains {secret}. "
                "Maya must publish release notes by Friday."
            )
        },
    ).status_code == 200

    connection = sqlite3.connect(test_store.database_path)
    try:
        columns = {
            row[1]
            for table in ("workflow_executions", "workflow_steps")
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        values = connection.execute("SELECT * FROM workflow_executions").fetchall()
        step_values = connection.execute("SELECT * FROM workflow_steps").fetchall()
    finally:
        connection.close()

    assert {
        "input",
        "output",
        "prompt",
        "raw_output",
        "arguments",
        "tool_result",
        "virtual_key",
    }.isdisjoint(columns)
    assert secret not in repr(values)
    assert secret not in repr(step_values)


def test_workflow_schema_initialization_is_idempotent(
    test_store: GatewayStore,
) -> None:
    test_store.initialize()
    connection = sqlite3.connect(test_store.database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                  AND name IN ('workflow_executions', 'workflow_steps')
                """
            ).fetchall()
        }
    finally:
        connection.close()

    assert tables == {"workflow_executions", "workflow_steps"}


def test_trace_creation_failure_releases_before_provider_work(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = workflow_api

    def fail_create(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("PRIVATE DATABASE PATH")

    monkeypatch.setattr(test_store, "create_workflow_execution", fail_create)

    response = _post(client, key="vk_edge")

    assert response.status_code == 500
    assert response.json() == {"detail": "workflow tracing failed"}
    assert "PRIVATE" not in response.text
    assert provider.calls == []
    stats = test_store.get_usage("vk_edge")
    assert stats is not None and stats.requests == 0


def test_atomic_settlement_failure_never_marks_workflow_completed(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = workflow_api
    _queue_success(provider, tool=False)
    original = test_store.settle_workflow

    def fail_settlement(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("PRIVATE SQL")

    monkeypatch.setattr(test_store, "settle_workflow", fail_settlement)
    response = _post(client)
    monkeypatch.setattr(test_store, "settle_workflow", original)
    trace = _get_trace(client, _only_workflow_id(test_store)).json()

    assert response.status_code == 500
    assert response.json() == {"detail": "workflow accounting failed"}
    assert "PRIVATE" not in response.text
    assert trace["status"] == "running"
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 0, 0)
    assert _usage_event_count(test_store) == 0


def test_existing_endpoint_contracts_remain_unchanged(
    workflow_api: tuple[TestClient, EndpointWorkflowProvider],
) -> None:
    client, _ = workflow_api
    chat = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_open"},
        json={"model": "client", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert chat.json() == {
        "content": "unchanged chat",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get(
        "/v1/preferences",
        headers={"Authorization": "Bearer vk_open"},
    ).json() == {"preferences": {}}
    usage = client.get("/usage", params={"key": "vk_open"}).json()
    assert set(usage) == {
        "key",
        "requests",
        "tokens_in",
        "tokens_out",
        "spend",
        "budget",
        "remaining",
    }
