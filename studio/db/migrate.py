"""Migration runner for Cat's Eye.

Applies additive migrations in order, tracks applied migrations in the schema_migrations
table, and supports both SQLite and PostgreSQL.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import sqlalchemy as sa

logger = logging.getLogger("studio.migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def ensure_migrations_table(conn: sa.Connection) -> None:
    """Ensure the schema_migrations tracking table exists."""
    insp = sa.inspect(conn)
    if not insp.has_table("schema_migrations"):
        conn.execute(
            sa.text(
                """
                CREATE TABLE schema_migrations (
                    name VARCHAR(255) PRIMARY KEY,
                    applied_at DOUBLE PRECISION NOT NULL
                )
                """
            )
        )


def get_applied_migrations(conn: sa.Connection) -> List[str]:
    """Return a list of migration names that have already been applied."""
    ensure_migrations_table(conn)
    result = conn.execute(sa.text("SELECT name FROM schema_migrations ORDER BY name ASC"))
    return [row[0] for row in result.fetchall()]


def discover_migrations() -> List[Tuple[str, Path]]:
    """Discover migration files in studio/db/migrations sorted by filename."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    migrations = []
    for f in MIGRATIONS_DIR.iterdir():
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__"):
            migrations.append((f.stem, f))
    migrations.sort(key=lambda x: x[0])
    return migrations


def load_migration_module(name: str, path: Path):
    """Dynamically load a migration module from file."""
    spec = importlib.util.spec_from_file_location(f"studio.db.migrations.{name}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load migration module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_all(engine: sa.Engine) -> List[str]:
    """Run all pending migrations in ascending order."""
    applied_in_this_run = []
    with engine.begin() as conn:
        ensure_migrations_table(conn)
        already_applied = set(get_applied_migrations(conn))
        all_migrations = discover_migrations()

        for name, path in all_migrations:
            if name in already_applied:
                continue
            logger.info("Applying migration %s...", name)
            mod = load_migration_module(name, path)
            if hasattr(mod, "up"):
                mod.up(conn)
            now = time.time()
            conn.execute(
                sa.text("INSERT INTO schema_migrations (name, applied_at) VALUES (:name, :t)"),
                {"name": name, "t": now},
            )
            applied_in_this_run.append(name)
            logger.info("Applied migration %s successfully", name)
    return applied_in_this_run


def rollback_latest(engine: sa.Engine) -> Optional[str]:
    """Roll back the most recently applied migration."""
    with engine.begin() as conn:
        ensure_migrations_table(conn)
        applied = get_applied_migrations(conn)
        if not applied:
            return None
        latest = applied[-1]
        all_migrations = dict(discover_migrations())
        if latest not in all_migrations:
            raise RuntimeError(f"Migration file for {latest} not found on disk")
        path = all_migrations[latest]
        mod = load_migration_module(latest, path)
        if hasattr(mod, "down"):
            mod.down(conn)
        conn.execute(sa.text("DELETE FROM schema_migrations WHERE name = :name"), {"name": latest})
        logger.info("Rolled back migration %s", latest)
        return latest
