from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema


_configured_test_database = (
    os.getenv("TEST_DATABASE_URL", "").strip()
    or os.getenv("DATABASE_URL", "").strip()
)
os.environ.setdefault(
    "DATABASE_URL",
    _configured_test_database
    or "postgresql+psycopg://invalid:invalid@127.0.0.1:1/unconfigured",
)

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
def postgres_database_url() -> str:
    database_url = (
        os.getenv("TEST_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not database_url or "unconfigured" in database_url:
        pytest.fail(
            "TEST_DATABASE_URL must point to a reachable PostgreSQL test database"
        )
    if make_url(database_url).get_backend_name() != "postgresql":
        pytest.fail("TEST_DATABASE_URL must use the PostgreSQL dialect")
    return database_url


@pytest.fixture
def store_factory(postgres_database_url: str):
    stores: list[tuple[GatewayStore, str]] = []

    def create() -> GatewayStore:
        schema = f"test_{uuid4().hex}"
        admin_engine = create_engine(
            postgres_database_url,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
        try:
            with admin_engine.begin() as connection:
                connection.execute(CreateSchema(schema))
        finally:
            admin_engine.dispose()

        test_url = make_url(postgres_database_url).update_query_dict(
            {"options": f"-csearch_path={schema}"}
        )
        store = GatewayStore(
            test_url.render_as_string(hide_password=False)
        )
        store.initialize()
        stores.append((store, schema))
        return store

    yield create

    for store, schema in reversed(stores):
        store.dispose()
        admin_engine = create_engine(
            postgres_database_url,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
        try:
            with admin_engine.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True))
        finally:
            admin_engine.dispose()


@pytest.fixture
def test_store(store_factory) -> GatewayStore:
    return store_factory()


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
