from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import inspect, text

from app.db import GatewayStore


def fetch_all(
    store: GatewayStore,
    statement: str,
    parameters: Mapping[str, object] | None = None,
):
    with store.engine.connect() as connection:
        return connection.execute(
            text(statement),
            dict(parameters or {}),
        ).fetchall()


def fetch_one(
    store: GatewayStore,
    statement: str,
    parameters: Mapping[str, object] | None = None,
):
    rows = fetch_all(store, statement, parameters)
    return rows[0] if rows else None


def fetch_scalar(
    store: GatewayStore,
    statement: str,
    parameters: Mapping[str, object] | None = None,
):
    row = fetch_one(store, statement, parameters)
    return None if row is None else row[0]


def table_columns(store: GatewayStore, table: str) -> set[str]:
    return {
        column["name"]
        for column in inspect(store.engine).get_columns(table)
    }


def table_names(store: GatewayStore) -> set[str]:
    return set(inspect(store.engine).get_table_names())


def index_names(store: GatewayStore, table: str) -> set[str]:
    return {
        index["name"]
        for index in inspect(store.engine).get_indexes(table)
    }
