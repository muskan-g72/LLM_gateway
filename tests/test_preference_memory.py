from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Lock

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import GatewayStore, virtual_key_identifier
from app.memory import (
    MAX_PREFERENCE_VALUE_BYTES,
    InvalidPreferenceNameError,
    InvalidPreferenceValueError,
    PreferenceService,
)
from app.output_validation import OutputValidator
from app.prompt_builder import PromptBuilder
from app.providers import ProviderCompletion
from app.skills import SkillLoader
from app.task_executor import TaskExecutor


class PreferenceProviderGateway:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderCompletion]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self._lock = Lock()

    def queue(self, provider_name: str, *events: ProviderCompletion) -> None:
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
            return self.events[provider_name].popleft()

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        return _completion()


@pytest.fixture
def preference_api(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> Iterator[tuple[TestClient, PreferenceProviderGateway]]:
    provider = PreferenceProviderGateway()
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


def _completion() -> ProviderCompletion:
    return ProviderCompletion(
        content=json.dumps(
            {
                "summary": "The release was approved.",
                "key_points": ["Friday is the release date."],
            }
        ),
        prompt_tokens=5,
        completion_tokens=3,
        provider="groq",
    )


def _headers(key: str = "vk_open") -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _post_task(
    client: TestClient,
    preferences: dict[str, object] | None = None,
):
    payload: dict[str, object] = {
        "skill": "summarize",
        "input": {"text": "The team approved the Friday release."},
    }
    if preferences is not None:
        payload["preferences"] = preferences
    return client.post("/v1/tasks/execute", headers=_headers(), json=payload)


def _prompt_preferences(
    provider: PreferenceProviderGateway,
    call_index: int = 0,
) -> dict[str, object] | None:
    user_content = next(
        message["content"]
        for message in provider.calls[call_index][1]
        if message["role"] == "user"
    )
    marker = "PREFERENCES_JSON:\n"
    if marker not in user_content:
        return None
    return json.loads(user_content.split(marker, 1)[1])


def test_service_accepts_primitive_and_nested_preferences(
    test_store: GatewayStore,
) -> None:
    service = PreferenceService(test_store)
    owner = virtual_key_identifier("vk_open")
    values = {
        "response_detail": "concise",
        "include_key_points": True,
        "options": {"count": 3, "labels": ["a", None, 1.5]},
    }

    assert service.put(owner, values) == values
    assert service.get(owner) == values


@pytest.mark.parametrize(
    "name",
    ["_private", "Uppercase", "has space", "path/name", "path\\name", "", "a" * 65],
)
def test_invalid_preference_names_are_rejected(
    test_store: GatewayStore,
    name: str,
) -> None:
    with pytest.raises(InvalidPreferenceNameError):
        PreferenceService(test_store).put(
            virtual_key_identifier("vk_open"),
            {name: "value"},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_preference_values_are_rejected(
    test_store: GatewayStore,
    value: float,
) -> None:
    with pytest.raises(InvalidPreferenceValueError):
        PreferenceService(test_store).put(
            virtual_key_identifier("vk_open"),
            {"number": value},
        )


def test_non_json_python_value_is_rejected_at_service_boundary(
    test_store: GatewayStore,
) -> None:
    with pytest.raises(InvalidPreferenceValueError):
        PreferenceService(test_store).put(
            virtual_key_identifier("vk_open"),
            {"created_on": date(2026, 7, 23)},
        )


def test_oversized_serialized_preference_value_is_rejected(
    test_store: GatewayStore,
) -> None:
    value = "é" * (MAX_PREFERENCE_VALUE_BYTES // 2 + 1)

    with pytest.raises(InvalidPreferenceValueError, match="4096 bytes"):
        PreferenceService(test_store).put(
            virtual_key_identifier("vk_open"),
            {"large_value": value},
        )


def test_all_values_are_validated_before_atomic_upsert(
    test_store: GatewayStore,
) -> None:
    owner = virtual_key_identifier("vk_open")
    service = PreferenceService(test_store)

    with pytest.raises(InvalidPreferenceNameError):
        service.put(owner, {"valid_name": "value", "invalid name": "bad"})

    assert service.get(owner) == {}


def test_preference_json_is_canonical_and_round_trips(
    test_store: GatewayStore,
) -> None:
    owner = virtual_key_identifier("vk_open")
    service = PreferenceService(test_store)
    service.put(owner, {"options": {"z": 2, "a": 1}})

    raw = test_store.get_preference_values(owner)

    assert raw == {"options": '{"a":1,"z":2}'}
    assert service.get(owner) == {"options": {"a": 1, "z": 2}}


def test_merge_copies_sources_and_request_values_override() -> None:
    stored = {"language": "English", "options": {"detail": "concise"}}
    requested = {"language": "French", "flags": [True]}

    merged = PreferenceService.merge(stored, requested)
    merged["options"]["detail"] = "changed"  # type: ignore[index]
    merged["flags"].append(False)  # type: ignore[union-attr]

    assert stored == {"language": "English", "options": {"detail": "concise"}}
    assert requested == {"language": "French", "flags": [True]}
    assert merged["language"] == "French"


def test_concurrent_disjoint_upserts_preserve_all_keys(
    test_store: GatewayStore,
) -> None:
    owner = virtual_key_identifier("vk_open")

    def write(index: int) -> None:
        PreferenceService(test_store).put(owner, {f"setting_{index}": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(8)))

    assert PreferenceService(test_store).get(owner) == {
        f"setting_{index}": index for index in range(8)
    }


def test_get_starts_empty_and_put_returns_complete_set(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, _ = preference_api

    assert client.get("/v1/preferences", headers=_headers()).json() == {
        "preferences": {}
    }
    response = client.put(
        "/v1/preferences",
        headers=_headers(),
        json={
            "preferences": {
                "response_detail": "concise",
                "preferred_language": "English",
                "include_key_points": True,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["preferences"] == {
        "response_detail": "concise",
        "preferred_language": "English",
        "include_key_points": True,
    }


def test_put_upserts_supplied_keys_and_preserves_unspecified_keys(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, _ = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"response_detail": "concise", "language": "English"}},
    )

    response = client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"response_detail": "detailed"}},
    )

    assert response.json() == {
        "preferences": {"language": "English", "response_detail": "detailed"}
    }


def test_delete_is_idempotent_and_removes_only_one_owned_key(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, _ = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"language": "English", "detail": "concise"}},
    )

    first = client.delete("/v1/preferences/language", headers=_headers())
    second = client.delete("/v1/preferences/language", headers=_headers())

    assert first.status_code == second.status_code == 204
    assert client.get("/v1/preferences", headers=_headers()).json() == {
        "preferences": {"detail": "concise"}
    }


def test_preference_ownership_isolated_for_read_and_delete(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, _ = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers("vk_open"),
        json={"preferences": {"language": "English"}},
    )

    assert client.get(
        "/v1/preferences",
        headers=_headers("vk_tiny"),
    ).json() == {"preferences": {}}
    assert client.delete(
        "/v1/preferences/language",
        headers=_headers("vk_tiny"),
    ).status_code == 204
    assert client.get("/v1/preferences", headers=_headers("vk_open")).json() == {
        "preferences": {"language": "English"}
    }


@pytest.mark.parametrize("authorization", [None, "vk_open", "Basic vk_open", "Bearer"])
def test_preference_endpoints_match_bearer_authentication(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
    authorization: str | None,
) -> None:
    client, _ = preference_api
    headers = {"Authorization": authorization} if authorization else {}

    assert client.get("/v1/preferences", headers=headers).status_code == 401


def test_preference_endpoint_rejects_bad_name_and_non_finite_value(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, _ = preference_api
    bad_name = client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"bad name": "value"}},
    )
    non_finite = client.put(
        "/v1/preferences",
        headers={**_headers(), "Content-Type": "application/json"},
        content='{"preferences":{"number":NaN}}',
    )

    assert bad_name.status_code == 422
    assert non_finite.status_code == 422
    assert bad_name.json() == {"detail": "invalid preferences"}


def test_database_failure_returns_safe_preference_error(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = preference_api

    def fail_lookup(owner: str) -> dict[str, str]:
        raise sqlite3.OperationalError("C:\\private\\gateway.db SQL SELECT")

    monkeypatch.setattr(test_store, "get_preference_values", fail_lookup)

    response = client.get("/v1/preferences", headers=_headers())

    assert response.status_code == 500
    assert response.json() == {"detail": "preference storage failed"}
    assert "gateway.db" not in response.text
    assert "SELECT" not in response.text


def test_stored_preferences_are_added_to_task_prompt(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, provider = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"preferred_language": "English"}},
    )
    provider.queue("groq", _completion())

    assert _post_task(client).status_code == 200
    assert _prompt_preferences(provider) == {"preferred_language": "English"}


def test_request_preferences_override_stored_without_persisting(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, provider = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={
            "preferences": {
                "preferred_language": "English",
                "response_detail": "concise",
            }
        },
    )
    provider.queue("groq", _completion())

    response = _post_task(
        client,
        {"response_detail": "detailed", "temporary": True},
    )

    assert response.status_code == 200
    assert _prompt_preferences(provider) == {
        "preferred_language": "English",
        "response_detail": "detailed",
        "temporary": True,
    }
    assert client.get("/v1/preferences", headers=_headers()).json() == {
        "preferences": {
            "preferred_language": "English",
            "response_detail": "concise",
        }
    }


def test_explicit_empty_request_preferences_retain_stored_values(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, provider = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"response_detail": "concise"}},
    )
    provider.queue("groq", _completion())

    assert _post_task(client, {}).status_code == 200
    assert _prompt_preferences(provider) == {"response_detail": "concise"}


def test_empty_memory_keeps_original_prompt_behavior(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
) -> None:
    client, provider = preference_api
    provider.queue("groq", _completion())

    assert _post_task(client).status_code == 200
    assert _prompt_preferences(provider) is None


def test_preference_lookup_failure_happens_before_reservation(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = preference_api

    def fail_lookup(owner: str) -> dict[str, str]:
        raise sqlite3.OperationalError("controlled")

    monkeypatch.setattr(test_store, "get_preference_values", fail_lookup)

    response = _post_task(client)

    assert response.status_code == 500
    assert provider.calls == []
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0


def test_unknown_skill_still_fails_before_memory_and_budget(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
    test_store: GatewayStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider = preference_api
    lookups: list[str] = []

    def record_lookup(owner: str) -> dict[str, str]:
        lookups.append(owner)
        return {}

    monkeypatch.setattr(test_store, "get_preference_values", record_lookup)
    response = client.post(
        "/v1/tasks/execute",
        headers=_headers(),
        json={"skill": "unknown", "input": {"text": "Example"}},
    )

    assert response.status_code == 404
    assert lookups == []
    assert provider.calls == []
    stats = test_store.get_usage("vk_open")
    assert stats is not None and stats.requests == 0


def test_new_tables_never_store_plaintext_virtual_keys(
    preference_api: tuple[TestClient, PreferenceProviderGateway],
    test_store: GatewayStore,
) -> None:
    client, provider = preference_api
    client.put(
        "/v1/preferences",
        headers=_headers(),
        json={"preferences": {"language": "English"}},
    )
    provider.queue("groq", _completion())
    _post_task(client)

    connection = sqlite3.connect(test_store.database_path)
    try:
        owners = connection.execute(
            "SELECT virtual_key_id FROM user_preferences UNION ALL "
            "SELECT virtual_key_id FROM task_executions"
        ).fetchall()
    finally:
        connection.close()

    assert owners
    assert all(owner[0] == virtual_key_identifier("vk_open") for owner in owners)
    assert all(owner[0] != "vk_open" for owner in owners)
