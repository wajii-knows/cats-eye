"""SQLite online backup and generation checkpoint manager for Google Colab Cat's Eye.

Features:
- Non-blocking online SQLite backup API (never copies active WAL directly).
- PRAGMA integrity_check verification prior to publishing.
- Retains the last 5 verified database generations.
- Automatic fallback to earlier generation if the latest is corrupt.
- Debounced and periodic background snapshot timers.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .durable import DurableStore, IntegrityError

logger = logging.getLogger("studio.storage.db_checkpoint")


class DatabaseCheckpointManager:
    """Manages verified multi-generation SQLite snapshots to durable storage."""

    def __init__(
        self,
        work_db_path: Union[str, Path],
        durable_store: DurableStore,
        max_generations: int = 5,
        checkpoint_interval: float = 30.0,
    ):
        self.work_db_path = Path(work_db_path).resolve()
        self.durable_store = durable_store
        self.max_generations = max_generations
        self.checkpoint_interval = checkpoint_interval
        self._lock = threading.Lock()
        self._last_checkpoint_time = 0.0
        self._is_dirty = False
        self._stop_timer = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None

    def mark_dirty(self) -> None:
        """Mark that database state has changed and needs checkpointing."""
        self._is_dirty = True

    def get_generations(self) -> List[int]:
        """Return sorted list of generation numbers currently in durable store."""
        files = self.durable_store.list("db/gen-")
        gens = []
        for f in files:
            m = re.search(r"gen-(\d+)\.sqlite$", f)
            if m:
                gens.append(int(m.group(1)))
        return sorted(gens)

    def create_checkpoint(self, reason: str = "periodic") -> Optional[Dict[str, Any]]:
        """Perform verified SQLite online backup, check integrity, and commit to durable store."""
        with self._lock:
            if not self.work_db_path.is_file():
                return None

            t0 = time.time()
            existing_gens = self.get_generations()
            next_gen = (existing_gens[-1] + 1) if existing_gens else 1

            with tempfile.TemporaryDirectory(prefix="ce_db_snap_") as tmp_dir:
                tmp_backup = Path(tmp_dir) / f"backup_gen_{next_gen}.sqlite"

                # 1. Non-blocking SQLite online backup
                try:
                    src_conn = sqlite3.connect(f"file:{self.work_db_path.as_posix()}?mode=ro", uri=True)
                    dest_conn = sqlite3.connect(tmp_backup)
                    src_conn.backup(dest_conn)
                    src_conn.close()
                except Exception as e:
                    # Fallback to direct connect if URI read-only fails
                    src_conn = sqlite3.connect(str(self.work_db_path))
                    dest_conn = sqlite3.connect(tmp_backup)
                    src_conn.backup(dest_conn)
                    src_conn.close()

                # 2. PRAGMA integrity_check
                try:
                    cur = dest_conn.cursor()
                    rows = cur.execute("PRAGMA integrity_check").fetchall()
                    if not rows or rows[0][0] != "ok":
                        dest_conn.close()
                        raise IntegrityError(f"Database integrity check failed: {rows}")
                    # Sanitize sessions for cloud snapshot security
                    try:
                        cur.execute("DELETE FROM sessions")
                        dest_conn.commit()
                    except Exception:
                        pass
                    dest_conn.close()
                except Exception as e:
                    logger.error("DB backup integrity check failed: %s", e)
                    raise

                # 3. Publish to durable store via verified write protocol
                rel_path = f"db/gen-{next_gen:04d}.sqlite"
                meta = self.durable_store.put_file(tmp_backup, rel_path)
                meta.update(generation=next_gen, reason=reason, at=t0)

                # 4. Update manifest generations metadata
                manifest = self.durable_store.get_manifest()
                manifest.setdefault("db_checkpoints", []).append(meta)
                # Keep last 10 in manifest history
                manifest["db_checkpoints"] = manifest["db_checkpoints"][-10:]
                self.durable_store.commit(manifest)

                # 5. Prune older generations beyond max_generations
                all_gens = self.get_generations()
                if len(all_gens) > self.max_generations:
                    to_remove = all_gens[: len(all_gens) - self.max_generations]
                    for g in to_remove:
                        self.durable_store.delete(f"db/gen-{g:04d}.sqlite")
                        logger.info("Pruned old DB generation %d", g)

                self._last_checkpoint_time = time.time()
                self._is_dirty = False
                logger.info("Successfully created DB checkpoint gen-%04d (%s)", next_gen, reason)
                return meta

    def maybe_checkpoint(self, debounce_sec: float = 30.0, reason: str = "debounced") -> Optional[Dict[str, Any]]:
        """Create a checkpoint if dirty and at least debounce_sec have passed."""
        if self._is_dirty and (time.time() - self._last_checkpoint_time >= debounce_sec):
            return self.create_checkpoint(reason=reason)
        return None

    def start_checkpoint_timer(self) -> None:
        """Start background daemon thread periodically checking dirty state."""
        if self._timer_thread and self._timer_thread.is_alive():
            return
        self._stop_timer.clear()

        def _timer_loop():
            while not self._stop_timer.wait(self.checkpoint_interval):
                try:
                    self.maybe_checkpoint(debounce_sec=self.checkpoint_interval, reason="timer")
                except Exception as e:
                    logger.warning("Periodic DB checkpoint error: %s", e)

        self._timer_thread = threading.Thread(target=_timer_loop, daemon=True, name="db_checkpoint_timer")
        self._timer_thread.start()

    def stop_checkpoint_timer(self) -> None:
        self._stop_timer.set()
        if self._timer_thread:
            self._timer_thread.join(timeout=2.0)
            self._timer_thread = None

    def restore_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Restore the latest verified database generation to the local work directory."""
        with self._lock:
            gens = self.get_generations()
            if not gens:
                logger.info("No existing DB checkpoints on Drive. Starting with fresh DB.")
                return None

            # Try newest generation first, falling back to older generations if corrupt
            for gen in reversed(gens):
                rel_path = f"db/gen-{gen:04d}.sqlite"
                with tempfile.TemporaryDirectory(prefix="ce_restore_") as tmp_dir:
                    candidate = Path(tmp_dir) / f"candidate_gen_{gen}.sqlite"
                    try:
                        self.durable_store.get_file(rel_path, candidate)
                        # Verify integrity
                        conn = sqlite3.connect(str(candidate))
                        cur = conn.cursor()
                        rows = cur.execute("PRAGMA integrity_check").fetchall()
                        conn.close()
                        if not rows or rows[0][0] != "ok":
                            logger.warning("DB checkpoint gen-%04d failed integrity check! Trying earlier generation...", gen)
                            continue

                        # Atomically install verified generation
                        self.work_db_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_target = self.work_db_path.parent / f"{self.work_db_path.name}.restoring"
                        shutil.copyfile(str(candidate), str(tmp_target))
                        os.replace(str(tmp_target), str(self.work_db_path))

                        meta = self.durable_store.stat(rel_path) or {}
                        meta.update(generation=gen)
                        logger.info("Restored verified DB checkpoint gen-%04d to %s", gen, self.work_db_path)
                        return meta
                    except Exception as e:
                        logger.warning("Failed restoring DB gen-%04d: %s. Falling back...", gen, e)

            logger.error("All existing DB checkpoints failed verification! Starting fresh.")
            return None
