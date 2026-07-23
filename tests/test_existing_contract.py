from __future__ import annotations

from typing import Any, Protocol

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings
from app.db import GatewayStore
from app.providers import ProviderCompletion, ProviderError, ProviderGateway


class RecordingProvider(Protocol):
    calls: list[list[dict[str, str]]]


def _chat_payload(model: str = "client-selected-model") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }


def _settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://unused:unused@127.0.0.1:1/unused",
        groq_api_key="fake-groq-key",
        groq_model="gateway-owned-groq-model",
        gemini_api_key="fake-gemini-key",
        gemini_model="gateway-owned-gemini-model",
        provider_timeout_seconds=1.0,
        force_primary_fail=False,
    )


def test_chat_rejects_missing_authorization(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=_chat_payload())

    assert response.status_code == 401
    assert response.json() == {"detail": "missing or unknown virtual key"}


@pytest.mark.parametrize(
    "authorization",
    ["vk_open", "Basic vk_open", "Bearer", "Bearer   "],
)
def test_chat_rejects_malformed_bearer_header(
    client: TestClient,
    authorization: str,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": authorization},
        json=_chat_payload(),
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing or unknown virtual key"}


def test_chat_rejects_unknown_virtual_key(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_unknown"},
        json=_chat_payload(),
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing or unknown virtual key"}


def test_valid_chat_preserves_response_contract(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_open"},
        json=_chat_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "content": "deterministic completion",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    assert type(response.json()["usage"]["prompt_tokens"]) is int
    assert type(response.json()["usage"]["completion_tokens"]) is int


def test_route_does_not_forward_client_model_to_provider(
    client: TestClient,
    fake_providers: RecordingProvider,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_open"},
        json=_chat_payload(model="untrusted-client-model"),
    )

    assert response.status_code == 200
    assert fake_providers.calls == [
        [{"role": "user", "content": "Hello"}],
    ]


def test_groq_uses_gateway_owned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: dict[str, Any] = {}

    class FakeResponse:
        is_success = True

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "provider response"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured_request["url"] = url
        captured_request.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    gateway = ProviderGateway(_settings())

    gateway._complete_with_groq([{"role": "user", "content": "Hello"}])

    body = captured_request["json"]
    assert isinstance(body, dict)
    assert body["model"] == "gateway-owned-groq-model"
    assert body["model"] != "untrusted-client-model"


def test_vk_tiny_succeeds_twice_then_returns_429(
    client: TestClient,
    fake_providers: RecordingProvider,
) -> None:
    responses = [
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer vk_tiny"},
            json=_chat_payload(),
        )
        for _ in range(3)
    ]

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert responses[2].json() == {"detail": "virtual key budget exhausted"}
    assert len(fake_providers.calls) == 2


def test_usage_matches_request_budget_spend_and_remaining(
    client: TestClient,
) -> None:
    for _ in range(2):
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer vk_tiny"},
            json=_chat_payload(),
        )
        assert response.status_code == 200

    response = client.get("/usage", params={"key": "vk_tiny"})

    assert response.status_code == 200
    assert response.json() == {
        "key": "vk_tiny",
        "requests": 2,
        "tokens_in": 10,
        "tokens_out": 6,
        "spend": 2,
        "budget": 2,
        "remaining": 0,
    }


def test_primary_provider_error_invokes_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = ProviderGateway(_settings())
    called: list[str] = []
    fallback_completion = ProviderCompletion(
        content="fallback response",
        prompt_tokens=6,
        completion_tokens=4,
        provider="gemini",
    )

    def failing_primary(messages: list[dict[str, str]]) -> ProviderCompletion:
        called.append("primary")
        raise ProviderError("controlled primary failure")

    def successful_fallback(
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        called.append("fallback")
        return fallback_completion

    monkeypatch.setattr(gateway, "_complete_with_groq", failing_primary)
    monkeypatch.setattr(gateway, "_complete_with_gemini", successful_fallback)

    completion = gateway.complete([{"role": "user", "content": "Hello"}])

    assert called == ["primary", "fallback"]
    assert completion == fallback_completion


def test_both_provider_failures_release_the_reservation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
) -> None:
    gateway = ProviderGateway(_settings())
    called: list[str] = []

    def failing_primary(messages: list[dict[str, str]]) -> ProviderCompletion:
        called.append("primary")
        raise ProviderError("controlled primary failure")

    def failing_fallback(messages: list[dict[str, str]]) -> ProviderCompletion:
        called.append("fallback")
        raise ProviderError("controlled fallback failure")

    monkeypatch.setattr(gateway, "_complete_with_groq", failing_primary)
    monkeypatch.setattr(gateway, "_complete_with_gemini", failing_fallback)
    monkeypatch.setattr(main_module, "providers", gateway)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk_edge"},
        json=_chat_payload(),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "all providers failed"}
    assert called == ["primary", "fallback"]
    stats = test_store.get_usage("vk_edge")
    assert stats is not None
    assert stats.requests == 0
    assert stats.tokens_in == 0
    assert stats.tokens_out == 0
    assert stats.as_contract()["remaining"] == 1


def test_seeded_key_budgets_remain_unchanged(test_store: GatewayStore) -> None:
    expected_budgets = {"vk_open": 50, "vk_tiny": 2, "vk_edge": 1}

    actual_budgets = {
        key: stats.budget
        for key in expected_budgets
        if (stats := test_store.get_usage(key)) is not None
    }

    assert actual_budgets == expected_budgets
