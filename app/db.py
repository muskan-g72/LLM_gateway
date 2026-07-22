from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ReservationResult = Literal["reserved", "unknown", "over_budget"]

SEEDED_KEYS = {
    "vk_open": 50,
    "vk_tiny": 2,
    "vk_edge": 1,
}


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
                """
            )
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
    ) -> None:
        """Atomically record every completion reported for one admitted request."""
        if not events:
            return

        token_input_total = sum(tokens_in for _, tokens_in, _ in events)
        token_output_total = sum(tokens_out for _, _, tokens_out in events)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
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
            connection.commit()
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
