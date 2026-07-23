from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


class DatabaseMigrationError(RuntimeError):
    """Database verification or schema migration failed without exposing its URL."""


def upgrade_database(database_url: str) -> None:
    """Upgrade one PostgreSQL database or isolated test schema to Alembic head."""
    project_root = Path(__file__).resolve().parent.parent
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    try:
        command.upgrade(config, "head")
    except Exception:
        raise DatabaseMigrationError(
            "PostgreSQL schema migration failed"
        ) from None
