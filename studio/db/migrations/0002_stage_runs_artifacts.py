"""Migration 0002_stage_runs_artifacts: Stage execution and artifact tables.

Adds stage_runs and artifacts tables for resumable pipeline stages and content-addressed
artifact storage.
"""
import sqlalchemy as sa
from studio.db import artifacts, stage_runs


def up(conn: sa.Connection) -> None:
    stage_runs.create(conn, checkfirst=True)
    artifacts.create(conn, checkfirst=True)


def down(conn: sa.Connection) -> None:
    artifacts.drop(conn, checkfirst=True)
    stage_runs.drop(conn, checkfirst=True)
