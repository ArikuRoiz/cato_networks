"""Alembic environment — targets the DATABASE_URL environment variable.

Supports both offline (SQL generation) and online (direct Postgres) migration modes.
Metadata is imported from the ORM models so autogenerate reflects all declared tables.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Alembic Config object — exposes the .ini file contents
# ---------------------------------------------------------------------------

config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Resolve the database URL
# ---------------------------------------------------------------------------
# Priority: DATABASE_URL env var → alembic.ini sqlalchemy.url fallback.
# The env var is the production-safe path; the .ini default is a dev convenience.

from firm.persistence.db_url import to_sqlalchemy_url  # noqa: E402

_db_url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
if _db_url:
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(_db_url))

# ---------------------------------------------------------------------------
# Import ORM metadata so autogenerate picks up all table definitions
# ---------------------------------------------------------------------------

from firm.persistence.models import Base  # noqa: E402 — after sys.path setup

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Run migrations offline (emit SQL to stdout / file)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection.

    Emits SQL to the script output so it can be reviewed or applied manually.
    Useful for generating a migration script to hand off to a DBA.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Run migrations online (connect and execute against Postgres)
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations with a live Postgres connection.

    Uses NullPool so Alembic does not leave idle connections after the run.
    """
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


# ---------------------------------------------------------------------------
# Entry point — Alembic calls this module directly
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
