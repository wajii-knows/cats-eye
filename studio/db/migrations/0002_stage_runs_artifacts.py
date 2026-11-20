"""Migration 0002_stage_runs_artifacts: Stage execution and artifact tables.

Adds stage_runs and artifacts tables for resumable pipeline stages and content-addressed
artifact storage.
"""
import sqlalchemy as sa
from studio.db import artifacts, stage_runs


def up(conn: sa.Connection) -> None:
    stage_runs.create(conn, checkfirst=True)
    insp = sa.inspect(conn)
    cols = {c["name"] for c in insp.get_columns("stage_runs")}
    if "local_done" not in cols:
        conn.execute(sa.text("ALTER TABLE stage_runs ADD COLUMN local_done BOOLEAN NOT NULL DEFAULT FALSE"))
    if "durable_done" not in cols:
        conn.execute(sa.text("ALTER TABLE stage_runs ADD COLUMN durable_done BOOLEAN NOT NULL DEFAULT FALSE"))
    artifacts.create(conn, checkfirst=True)


def down(conn: sa.Connection) -> None:
    artifacts.drop(conn, checkfirst=True)
    stage_runs.drop(conn, checkfirst=True)
