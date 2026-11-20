"""Migration 0001_initial: Initial database schema.

Captures initial Cat's Eye schema (users, sessions, usage, email_ok, email_tokens,
feedback, notices, admin_log) idempotently.
"""
import sqlalchemy as sa
from studio.db import (
    admin_log,
    email_ok,
    email_tokens,
    feedback,
    notices,
    sessions,
    usage,
    users,
)

INITIAL_TABLES = [
    admin_log,
    notices,
    feedback,
    email_tokens,
    email_ok,
    usage,
    sessions,
    users,
]


def up(conn: sa.Connection) -> None:
    # Create tables in dependency order (users first, then children) idempotently
    for table in reversed(INITIAL_TABLES):
        table.create(conn, checkfirst=True)


def down(conn: sa.Connection) -> None:
    # Drop tables in reverse dependency order
    for table in INITIAL_TABLES:
        table.drop(conn, checkfirst=True)
