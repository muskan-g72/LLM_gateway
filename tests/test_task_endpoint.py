from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import GatewayStore
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import (
    ProviderCompletion,
    ProviderConfigurationError,
    ProviderOperationalError,
)
from app.skills import SkillLoader
from app.task_executor import TaskExecutor
from tests.database_helpers import fetch_scalar


ProviderEvent = ProviderCompletion | Exception


class EndpointProviderGateway:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderEvent]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self._lock = Lock()

    def queue(self, provider_name: str, *events: ProviderEvent) -> None:
        self.events[provider_name].extend(events)

    def complete_with_provider(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        with self._lock:
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
        return ProviderCompletion(
            content="unchanged chat response",
            prompt_tokens=5,
            completion_tokens=3,
            provider="fake-chat",
        )


@pytest.fixture
def task_api(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> Iterator[tuple[TestClient, EndpointProviderGateway]]:
    provider = EndpointProviderGateway()
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


def _task_payload(
    preferences: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "skill": "summarize",
        "input": {"text": "The team approved the Friday release."},
    }
    if preferences is not None:
        payload["preferences"] = preferences
    return payload


def _post_task(
    client: TestClient,
    payload: dict[str, object] | None = None,
    authorization: str = "Bearer vk_open",
):
    return client.post(
        "/v1/tasks/execute",
        headers={"Authorization": authorization},
        json=payload or _task_payload(),
    )


def _usage_event_count(store: GatewayStore) -> int:
    count = fetch_scalar(store, "SELECT COUNT(*) FROM usage_events")
    assert count is not None
    return int(count)


def test_valid_summarize_endpoint_contract(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, provider = task_api
    provider.queue("groq", _completion("groq", _summary_json(), 7, 4))

    response = _post_task(
        client,
        _task_payload(
            {
                "response_detail": "concise",
                "preferred_language": "English",
                "include_key_points": True,
            }
        ),
    )

    assert response.status_code == 200
    body = response.json()
    UUID(body["task_id"])
    assert body == {
        "task_id": body["task_id"],
        "status": "completed",
        "skill": "summarize",
        "output": {
            "summary": "The release was approved.",
            "key_points": ["Friday is the release date."],
        },
        "provider": "groq",
        "attempts": 1,
        "usage": {"prompt_tokens": 7, "completion_tokens": 4},
    }


def test_valid_extract_action_items_endpoint_contract(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, provider = task_api
    raw_output = json.dumps(
        {
            "action_items": [
                {
                    "task": "Submit the report",
                    "owner": "Priya",
                    "deadline": "Friday",
                }
            ]
        }
    )
    provider.queue("groq", _completion("groq", raw_output))

    response = _post_task(
        client,
        {
            "skill": "extract_action_items",
            "input": {"text": "Priya will submit the report by Friday."},
        },
    )

    assert response.status_code == 200
    assert response.json()["output"] == {
        "action_items": [
            {"task": "Submit the report", "owner": "Priya", "deadline": "Friday"}
        ]
    }


def test_unexpected_top_level_request_field_is_rejected(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, provider = task_api
    payload = _task_payload()
    payload["unexpected"] = True

    response = _post_task(client, payload)

    assert response.status_code == 422
    assert provider.calls == []


@pytest.mark.parametrize(
    "authorization",
    [None, "vk_open", "Basic vk_open", "Bearer", "Bearer   "],
)
def test_task_endpoint_rejects_missing_or_malformed_authorization(
    task_api: tuple[TestClient, EndpointProviderGateway],
    authorization: str | None,
) -> None:
    client, provider = task_api
    headers = {"Authorization": authorization} if authorization is not None else {}

    response = client.post(
        "/v1/tasks/execute",
        headers=headers,
        json=_task_payload(),
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing or unknown virtual key"}
    assert provider.calls == []


def test_task_endpoint_rejects_unknown_key_before_skill_lookup(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, provider = task_api

    response = _post_task(
        client,
        {"skill": "not_registered", "input": {"text": "Example"}},
        authorization="Bearer vk_unknown",
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing or unknown virtual key"}
    assert provider.calls == []


def test_exhausted_task_key_returns_429_before_provider_call(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    assert test_store.reserve_request("vk_edge") == "reserved"

    response = _post_task(client, authorization="Bearer vk_edge")

    assert response.status_code == 429
    assert response.json() == {"detail": "virtual key budget exhausted"}
    assert provider.calls == []


def test_unknown_skill_returns_404_without_consuming_budget(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api

    response = _post_task(
        client,
        {"skill": "translate", "input": {"text": "Example"}},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown skill"}
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0
    assert provider.calls == []


@pytest.mark.parametrize("task_input", [{}, {"text": 7}, {"text": ""}])
def test_invalid_local_task_input_returns_422_without_consuming_budget(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
    task_input: dict[str, object],
) -> None:
    client, provider = task_api

    response = _post_task(
        client,
        {"skill": "summarize", "input": task_input},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid task input"}
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0
    assert provider.calls == []


def test_preferences_are_optional_and_not_added_when_missing(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, provider = task_api
    provider.queue("groq", _completion("groq", _summary_json()))

    response = _post_task(client)

    assert response.status_code == 200
    initial_user_message = next(
        message["content"]
        for message in provider.calls[0][1]
        if message["role"] == "user"
    )
    assert "PREFERENCES_JSON:" not in initial_user_message


def test_successful_task_consumes_one_request_and_records_one_completion(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue("groq", _completion("groq", _summary_json(), 7, 4))

    response = _post_task(client)
    usage_response = client.get("/usage", params={"key": "vk_open"})

    assert response.status_code == 200
    assert usage_response.status_code == 200
    assert usage_response.json() == {
        "key": "vk_open",
        "requests": 1,
        "tokens_in": 7,
        "tokens_out": 4,
        "spend": 1,
        "budget": 50,
        "remaining": 49,
    }
    assert _usage_event_count(test_store) == 1


def test_primary_repair_uses_one_request_and_aggregates_two_completions(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue(
        "groq",
        _completion("groq", "not json", 3, 1),
        _completion("groq", _summary_json(), 6, 4),
    )

    response = _post_task(client)

    assert response.status_code == 200
    assert response.json()["attempts"] == 2
    assert response.json()["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 5,
    }
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 9, 5)
    assert _usage_event_count(test_store) == 2


def test_fallback_uses_one_request_and_only_reported_completion_usage(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue("groq", ProviderOperationalError("primary unavailable"))
    provider.queue("gemini", _completion("gemini", _summary_json(), 8, 5))

    response = _post_task(client)

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert response.json()["attempts"] == 2
    assert response.json()["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
    }
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 1
    assert _usage_event_count(test_store) == 1


def test_fallback_repair_aggregates_both_reported_fallback_completions(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue("groq", ProviderOperationalError("primary unavailable"))
    provider.queue(
        "gemini",
        _completion("gemini", "not json", 4, 2),
        _completion("gemini", _summary_json(), 7, 3),
    )

    response = _post_task(client)

    assert response.status_code == 200
    assert response.json()["attempts"] == 3
    assert response.json()["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
    }
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 11, 5)
    assert _usage_event_count(test_store) == 2


def test_four_call_flow_still_consumes_one_request(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
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

    response = _post_task(client)

    assert response.status_code == 200
    assert response.json()["attempts"] == 4
    assert response.json()["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 6,
    }
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 1
    assert _usage_event_count(test_store) == 3


def test_complete_operational_failure_releases_reservation(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue("groq", ProviderOperationalError("primary unavailable"))
    provider.queue("gemini", ProviderOperationalError("fallback unavailable"))

    response = _post_task(client, authorization="Bearer vk_edge")

    assert response.status_code == 502
    assert response.json() == {"detail": "all providers unavailable"}
    stats = test_store.get_usage("vk_edge")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (0, 0, 0)


def test_billable_invalid_output_remains_charged_and_is_never_returned(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    raw_secret = "FULL_INVALID_MODEL_OUTPUT_SECRET"
    provider.queue(
        "groq",
        _completion("groq", raw_secret, 3, 2),
        _completion("groq", raw_secret, 4, 1),
    )

    response = _post_task(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "provider output remained invalid"}
    assert raw_secret not in response.text
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert (stats.requests, stats.tokens_in, stats.tokens_out) == (1, 7, 3)
    assert _usage_event_count(test_store) == 2


def test_provider_configuration_error_is_safe_500_and_releases_without_usage(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = task_api
    provider.queue(
        "groq",
        ProviderConfigurationError(
            "GROQ_API_KEY=provider-secret Authorization: Bearer vk_open"
        ),
    )

    response = _post_task(client)

    assert response.status_code == 500
    assert response.json() == {"detail": "provider configuration error"}
    assert "provider-secret" not in response.text
    assert "vk_open" not in response.text
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0


def test_accounting_failure_returns_safe_500_and_keeps_billable_reservation(
    task_api: tuple[TestClient, EndpointProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = task_api
    provider.queue("groq", _completion("groq", _summary_json()))

    def fail_accounting(
        key: str,
        events: list[tuple[str, int, int]],
    ) -> None:
        raise RuntimeError("controlled accounting failure")

    monkeypatch.setattr(test_store, "record_usage_events", fail_accounting)

    response = _post_task(client)

    assert response.status_code == 500
    assert response.json() == {"detail": "task accounting failed"}
    stats = test_store.get_usage("vk_open")
    assert stats is not None
    assert stats.requests == 1


def test_existing_chat_endpoint_contract_remains_unchanged(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    client, _ = task_api

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_open"},
        json={
            "model": "client-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "content": "unchanged chat response",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def test_parallel_vk_edge_tasks_admit_exactly_one(
    task_api: tuple[TestClient, EndpointProviderGateway],
) -> None:
    _, provider = task_api
    provider.queue("groq", _completion("groq", _summary_json()))
    contenders = 10
    start_together = Barrier(contenders)
    request = main_module.TaskRequest.model_validate(_task_payload())

    def execute() -> int:
        start_together.wait()
        try:
            main_module.execute_task(request, "Bearer vk_edge")
        except HTTPException as error:
            return error.status_code
        return 200

    with ThreadPoolExecutor(max_workers=contenders) as executor:
        statuses = list(executor.map(lambda _: execute(), range(contenders)))

    assert Counter(statuses) == Counter({429: 9, 200: 1})
    assert len(provider.calls) == 1
