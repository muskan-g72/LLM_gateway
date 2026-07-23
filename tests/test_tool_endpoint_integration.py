from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from threading import Lock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect

from app import main as main_module
from app.db import GatewayStore
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import ProviderCompletion, ProviderOperationalError
from app.skills import SkillLoader
from app.task_executor import TaskExecutor
from app.tracing import ExecutionAttempt
from app.tools import (
    CalculatorInput,
    CalculatorOutput,
    RegisteredTool,
    ToolDefinition,
    ToolRegistry,
    build_builtin_tool_registry,
)
from tests.database_helpers import fetch_all, fetch_scalar, table_columns, table_names


ProviderEvent = ProviderCompletion | Exception


class EndpointToolProvider:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderEvent]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self._lock = Lock()

    def queue(self, provider: str, *events: ProviderEvent) -> None:
        self.events[provider].extend(events)

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
        if isinstance(event, Exception):
            raise event
        return event

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        return ProviderCompletion("unchanged chat response", 5, 3, "fake-chat")


@pytest.fixture
def tool_api(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> Iterator[tuple[TestClient, EndpointToolProvider]]:
    provider = EndpointToolProvider()
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
    return ProviderCompletion(content, prompt_tokens, completion_tokens, provider)


def _summary_json() -> str:
    return json.dumps(
        {
            "summary": "The calculation result is 5.",
            "key_points": ["The sum is 5."],
        }
    )


def _tool_call(name: str, arguments: object) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "tool_call": {"name": name, "arguments": arguments},
        }
    )


def _post_task(
    client: TestClient,
    *,
    skill: str = "summarize",
    text: str = "Calculate the sum of 2 and 3 for the result.",
    key: str = "vk_open",
):
    return client.post(
        "/v1/tasks/execute",
        headers={"Authorization": f"Bearer {key}"},
        json={"skill": skill, "input": {"text": text}},
    )


def _trace(client: TestClient, task_id: str, key: str = "vk_open"):
    return client.get(
        f"/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {key}"},
    )


def _only_task_id(store: GatewayStore) -> str:
    task_id = fetch_scalar(
        store,
        "SELECT task_id FROM task_executions ORDER BY created_at DESC LIMIT 1",
    )
    assert task_id is not None
    return str(task_id)


def _usage_event_count(store: GatewayStore) -> int:
    count = fetch_scalar(store, "SELECT COUNT(*) FROM usage_events")
    assert count is not None
    return int(count)


def test_calculator_endpoint_success_preserves_task_response_contract(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 4, 2),
        _completion("groq", _summary_json(), 6, 3),
    )

    response = _post_task(client)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "task_id": body["task_id"],
        "status": "completed",
        "skill": "summarize",
        "output": {
            "summary": "The calculation result is 5.",
            "key_points": ["The sum is 5."],
        },
        "provider": "groq",
        "attempts": 2,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_text_statistics_endpoint_executes_registered_tool(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("text_statistics", {"text": "hello world"})),
        _completion("groq", _summary_json()),
    )

    response = _post_task(client)

    assert response.status_code == 200
    assert '"characters":11' in provider.calls[1][1][-1]["content"]


def test_successful_tool_trace_is_additive_and_safe(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        _completion("groq", _summary_json()),
    )
    task = _post_task(client)

    trace = _trace(client, task.json()["task_id"])

    assert trace.status_code == 200
    body = trace.json()
    assert [item["attempt_type"] for item in body["attempt_history"]] == [
        "initial",
        "post_tool",
    ]
    assert body["tool_history"] == [
        {
            "tool_number": 1,
            "tool_name": "calculator",
            "status": "completed",
            "error_category": None,
            "duration_ms": body["tool_history"][0]["duration_ms"],
            "created_at": body["tool_history"][0]["created_at"],
            "completed_at": body["tool_history"][0]["completed_at"],
        }
    ]
    assert type(body["tool_history"][0]["duration_ms"]) is int
    assert body["tool_history"][0]["completed_at"] is not None


def test_no_tool_trace_response_remains_exactly_backward_compatible(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, provider = tool_api
    provider.queue("groq", _completion("groq", _summary_json()))
    task = _post_task(client)

    body = _trace(client, task.json()["task_id"]).json()

    assert "tool_history" not in body
    assert body["attempt_history"][0]["attempt_type"] == "initial"


def test_tool_task_charges_one_request_and_records_only_provider_usage(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 4, 2),
        _completion("groq", _summary_json(), 6, 3),
    )

    response = _post_task(client)
    usage = client.get("/usage", params={"key": "vk_open"}).json()

    assert response.status_code == 200
    assert usage["requests"] == 1
    assert usage["tokens_in"] == 10
    assert usage["tokens_out"] == 5
    assert _usage_event_count(test_store) == 2


@pytest.mark.parametrize(
    ("skill", "tool_name", "arguments", "category"),
    [
        ("summarize", "not_registered", {}, "tool_not_found"),
        ("extract_action_items", "calculator", {"operation": "add", "a": 2, "b": 3}, "tool_not_allowed"),
        ("summarize", "calculator", {"operation": "add", "a": 2}, "tool_arguments_invalid"),
    ],
)
def test_invalid_tool_request_stays_charged_without_tool_row(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
    skill: str,
    tool_name: str,
    arguments: object,
    category: str,
) -> None:
    client, provider = tool_api
    provider.queue("groq", _completion("groq", _tool_call(tool_name, arguments), 4, 2))

    response = _post_task(client, skill=skill)
    task_id = _only_task_id(test_store)
    trace = _trace(client, task_id).json()

    assert response.status_code == 502
    assert response.json() == {"detail": "tool execution failed"}
    assert trace["status"] == "failed"
    assert trace["error_category"] == category
    assert "tool_history" not in trace
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 4, 2)


def test_second_tool_request_fails_after_exactly_one_execution(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = tool_api
    request = _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})
    provider.queue(
        "groq",
        _completion("groq", request, 3, 1),
        _completion("groq", request, 4, 2),
    )

    response = _post_task(client)
    trace = _trace(client, _only_task_id(test_store)).json()

    assert response.status_code == 502
    assert trace["error_category"] == "repeated_tool_call"
    assert len(trace["tool_history"]) == 1
    assert trace["tool_history"][0]["status"] == "completed"
    assert trace["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}


def _replace_executor_registry(
    provider: EndpointToolProvider,
    registry: ToolRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "task_executor",
        TaskExecutor(
            SkillLoader(),
            PromptBuilder(),
            provider,  # type: ignore[arg-type]
            OutputValidator(),
            registry,
        ),
    )


def _calculator_registry(handler, *, timeout: float = 1.0) -> ToolRegistry:
    builtin = build_builtin_tool_registry()
    calculator = RegisteredTool(
        ToolDefinition(
            name="calculator",
            description="Test calculator implementation.",
            input_model=CalculatorInput,
            output_model=CalculatorOutput,
            timeout_seconds=timeout,
        ),
        handler,
    )
    return ToolRegistry([calculator, builtin.get("text_statistics")])


@pytest.mark.parametrize(
    ("handler", "timeout", "category"),
    [
        (lambda value: (_ for _ in ()).throw(RuntimeError("C:\\private\\secret")), 1.0, "tool_execution"),
        (lambda value: (time.sleep(0.05), {"result": 5})[1], 0.001, "tool_timeout"),
        (lambda value: {"wrong": "shape", "secret": "RAW_RESULT"}, 1.0, "tool_result_invalid"),
    ],
)
def test_tool_runtime_failure_is_safe_charged_and_traced(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
    handler,
    timeout: float,
    category: str,
) -> None:
    client, provider = tool_api
    _replace_executor_registry(provider, _calculator_registry(handler, timeout=timeout), monkeypatch)
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 4, 2),
    )

    response = _post_task(client)
    trace = _trace(client, _only_task_id(test_store)).json()

    assert response.status_code == 502
    assert response.json() == {"detail": "tool execution failed"}
    assert "private" not in response.text
    assert "RAW_RESULT" not in response.text
    assert trace["tool_history"][0]["status"] == "failed"
    assert trace["tool_history"][0]["error_category"] == category
    assert trace["error_category"] == category
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 1


def test_tool_trace_does_not_persist_arguments_results_or_exception_text(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = tool_api
    argument_secret = "RAW_ARGUMENT_SECRET"
    provider.queue(
        "groq",
        _completion(
            "groq",
            _tool_call("text_statistics", {"text": argument_secret}),
        ),
        _completion("groq", _summary_json()),
    )
    assert _post_task(client).status_code == 200

    columns = table_columns(test_store, "task_tool_executions")
    values = fetch_all(test_store, "SELECT * FROM task_tool_executions")

    assert {"arguments", "result", "prompt", "raw_output"}.isdisjoint(columns)
    assert argument_secret not in repr(values)
    assert "characters" not in repr(values)


def test_tool_trace_is_hidden_from_non_owner(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3})),
        _completion("groq", _summary_json()),
    )
    task_id = _post_task(client).json()["task_id"]

    assert _trace(client, task_id, "vk_tiny").status_code == 404


def test_all_provider_failure_before_completion_still_releases_reservation(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
) -> None:
    client, provider = tool_api
    provider.queue("groq", ProviderOperationalError("primary"))
    provider.queue("gemini", ProviderOperationalError("fallback"))

    response = _post_task(client, key="vk_edge")

    assert response.status_code == 502
    stats = test_store.get_usage("vk_edge")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (0, 0, 0)


def test_tool_recording_failure_stops_before_post_tool_call(
    tool_api: tuple[TestClient, EndpointToolProvider],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = tool_api
    provider.queue(
        "groq",
        _completion("groq", _tool_call("calculator", {"operation": "add", "a": 2, "b": 3}), 4, 2),
        _completion("groq", _summary_json()),
    )

    def fail_tool_finish(*args: object, **kwargs: object) -> None:
        raise RuntimeError("PRIVATE SQL")

    monkeypatch.setattr(test_store, "finalize_task_tool_execution", fail_tool_finish)

    response = _post_task(client)
    task_id = _only_task_id(test_store)
    trace = _trace(client, task_id).json()

    assert response.status_code == 500
    assert response.json() == {"detail": "task tracing failed"}
    assert len(provider.calls) == 1
    assert trace["status"] == "failed"
    assert trace["error_category"] == "trace_recording"
    assert trace["tool_history"][0]["status"] == "running"
    assert "PRIVATE SQL" not in response.text


def test_tool_schema_initialization_is_idempotent_and_attempt_types_are_widened(
    test_store: GatewayStore,
) -> None:
    test_store.initialize()
    checks = inspect(test_store.engine).get_check_constraints("task_attempts")
    assert any(
        "post_tool_fallback_repair" in (constraint["sqltext"] or "")
        for constraint in checks
    )
    assert "task_tool_executions" in table_names(test_store)


def test_existing_attempt_rows_survive_idempotent_alembic_initialization(
    test_store: GatewayStore,
) -> None:
    test_store.create_task_execution("old-task", "internal-owner", "summarize")
    test_store.append_task_attempt(
        "old-task",
        ExecutionAttempt(1, "groq", "initial", "completed", 2, 3),
    )
    test_store.initialize()

    retained = fetch_all(
        test_store,
        "SELECT task_id, attempt_number, attempt_type FROM task_attempts",
    )
    checks = inspect(test_store.engine).get_check_constraints("task_attempts")

    assert [tuple(row) for row in retained] == [("old-task", 1, "initial")]
    assert any(
        "post_tool_fallback_repair" in (constraint["sqltext"] or "")
        for constraint in checks
    )


def test_chat_usage_health_and_preferences_contracts_remain_unchanged(
    tool_api: tuple[TestClient, EndpointToolProvider],
) -> None:
    client, _ = tool_api
    chat = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_open"},
        json={"model": "client", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert chat.json() == {
        "content": "unchanged chat response",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get(
        "/v1/preferences",
        headers={"Authorization": "Bearer vk_open"},
    ).json() == {"preferences": {}}
    usage = client.get("/usage", params={"key": "vk_open"}).json()
    assert set(usage) == {
        "key", "requests", "tokens_in", "tokens_out", "spend", "budget", "remaining"
    }
