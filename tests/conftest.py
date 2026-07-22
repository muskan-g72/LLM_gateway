from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import GatewayStore
from app.providers import ProviderCompletion


class FakeProviderGateway:
    """Return a fixed completion while recording the messages received."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        self.calls.append(messages)
        return ProviderCompletion(
            content="deterministic completion",
            prompt_tokens=5,
            completion_tokens=3,
            provider="fake-primary",
        )


@pytest.fixture(autouse=True)
def block_real_provider_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test immediately if production provider code reaches the network."""

    def blocked_http_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tests must not make real provider HTTP calls")

    monkeypatch.setattr("app.providers.httpx.post", blocked_http_call)


@pytest.fixture
def test_store(tmp_path: Path) -> GatewayStore:
    store = GatewayStore(str(tmp_path / "gateway-test.db"))
    store.initialize()
    return store


@pytest.fixture
def fake_providers() -> FakeProviderGateway:
    return FakeProviderGateway()


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    test_store: GatewayStore,
    fake_providers: FakeProviderGateway,
) -> Iterator[TestClient]:
    monkeypatch.setattr(main_module, "store", test_store)
    monkeypatch.setattr(main_module, "providers", fake_providers)

    with TestClient(main_module.app) as test_client:
        yield test_client
