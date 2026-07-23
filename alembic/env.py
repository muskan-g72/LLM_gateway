from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url


config = context.config
configured_url = config.get_main_option("sqlalchemy.url").strip()
database_url = configured_url or os.getenv("DATABASE_URL", "").strip()
if not database_url:
    raise RuntimeError("DATABASE_URL is required for Alembic")
if make_url(database_url).get_backend_name() != "postgresql":
    raise RuntimeError("Alembic requires a PostgreSQL DATABASE_URL")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
