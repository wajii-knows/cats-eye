"""Colab mode manager for Cat's Eye: coordinates session lease, durable store, DB snapshots, and uploader."""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from studio import config
from studio.storage.durable import DriveMountStore, DurableStore, LocalFolderStore
from studio.storage.lease import ActiveLeaseError, SessionLease
from studio.storage.db_checkpoint import DatabaseCheckpointManager
from studio.storage.uploader import DurableUploader

logger = logging.getLogger("studio.storage.colab")

_manager: Optional[ColabManager] = None
_lock = threading.Lock()


class ColabManager:
    """Coordinates durable operations for Google Colab runtime."""

    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        durable_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        durable_store: Optional[DurableStore] = None,
    ):
        self.work_dir = Path(work_dir or config.WORK_DIR).resolve()
        self.durable_dir = Path(durable_dir or config.DURABLE_DIR).resolve()
        self.session_id = session_id or f"colab-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.boot_time = time.time()

        self.durable_store: DurableStore = (
            durable_store if durable_store is not None else DriveMountStore(self.durable_dir, session_id=self.session_id)
        )
        self.lease = SessionLease(self.durable_dir, session_id=self.session_id)

        work_db = self.work_dir / "peakcut.db"
        self.db_checkpoint_mgr = DatabaseCheckpointManager(
            work_db_path=work_db,
            durable_store=self.durable_store,
            max_generations=5,
            checkpoint_interval=config.DB_CHECKPOINT_INTERVAL_SEC,
        )
        self.uploader = DurableUploader(self.durable_store, max_workers=2)

        self.recovered_from_checkpoint = False
        self.recovery_checkpoint_info: Optional[Dict[str, Any]] = None
        self.resumed_jobs_count = 0
        self._started = False

    def start(self, force_lease: bool = False) -> None:
        if self._started:
            return

        if not config.OWNER_PASSWORD:
            raise RuntimeError("PEAKCUT_OWNER_PASSWORD is required when PEAKCUT_COLAB is enabled.")

        # Ensure directories exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.durable_dir.mkdir(parents=True, exist_ok=True)

        # 1. Acquire session lease
        self.lease.acquire(force=force_lease)
        self.lease.start_heartbeat()

        # 2. Checkpoint Restore if DB doesn't exist locally
        work_db = self.work_dir / "peakcut.db"
        if not work_db.exists() or work_db.stat().st_size == 0:
            restored = self.db_checkpoint_mgr.restore_latest_checkpoint()
            if restored:
                self.recovered_from_checkpoint = True
                self.recovery_checkpoint_info = restored
                logger.info(
                    "Restored database from durable checkpoint gen-%s (%s)",
                    restored.get("generation"),
                    str(restored.get("sha256", ""))[:8],
                )

        # 3. Restore durable user metadata (job.json, creators) if missing locally
        user_files = self.durable_store.list("users/")
        for uf in user_files:
            if uf.endswith(".json"):
                local_dest = self.work_dir / uf
                if not local_dest.exists():
                    try:
                        self.durable_store.get_file(uf, local_dest)
                    except Exception as e:
                        logger.warning("Could not restore user metadata %s: %s", uf, e)

        # 4. Start background services
        self.uploader.start()
        self.db_checkpoint_mgr.start_checkpoint_timer()
        self._started = True

    def stop(self) -> None:
        self.db_checkpoint_mgr.stop_checkpoint_timer()
        self.uploader.stop()
        self.lease.release()
        self._started = False

    def checkpoint_now(self, reason: str = "manual_ui") -> Optional[Dict[str, Any]]:
        return self.db_checkpoint_mgr.create_checkpoint(reason=reason)

    def set_resumed_jobs_count(self, count: int) -> None:
        self.resumed_jobs_count = count

    def get_status(self) -> Dict[str, Any]:
        uploader_status = self.uploader.get_status()

        # Drive disk free space
        drive_free_gb = 0.0
        try:
            drive_free_gb = round(shutil.disk_usage(self.durable_dir).free / 1e9, 2)
        except OSError:
            pass

        # Last 5 DB checkpoints
        gens = self.db_checkpoint_mgr.get_generations()
        recent_gens = sorted(gens, reverse=True)[:5]
        checkpoints_info = []
        for g in recent_gens:
            st = self.durable_store.stat(f"db/gen-{g}.sqlite")
            if st:
                checkpoints_info.append({
                    "generation": g,
                    "sha256": st.get("sha256", ""),
                    "size": st.get("size", 0),
                    "created_at": st.get("updated_at", 0),
                    "verified": True,
                })

        lease_data = self.lease.read_lease() or {}

        return {
            "sync_state": uploader_status["state"],
            "sync_label": uploader_status["label"],
            "pending_count": uploader_status["pending_count"],
            "pending_bytes": uploader_status["pending_bytes"],
            "last_sync_time": uploader_status["last_sync_time"],
            "last_error": uploader_status["last_error"],
            "safe_to_disconnect": uploader_status["safe_to_disconnect"],
            "drive_free_gb": drive_free_gb,
            "drive_min_free_gb": config.DRIVE_MIN_FREE_GB,
            "drive_low_space": drive_free_gb < config.DRIVE_MIN_FREE_GB,
            "last_checkpoints": checkpoints_info,
            "lease": {
                "session_id": self.session_id,
                "current_holder": lease_data.get("session_id"),
                "holder_host": lease_data.get("hostname"),
                "last_heartbeat": lease_data.get("last_heartbeat"),
            },
            "recovery": {
                "recovered": self.recovered_from_checkpoint,
                "checkpoint": self.recovery_checkpoint_info,
                "resumed_jobs": self.resumed_jobs_count,
            },
            "uptime_sec": round(time.time() - self.boot_time, 1),
            "colab_mode": True,
        }


def get_colab_manager(create: bool = True) -> Optional[ColabManager]:
    global _manager
    with _lock:
        if _manager is None and create and config.COLAB_MODE:
            _manager = ColabManager()
        return _manager


def set_colab_manager(mgr: Optional[ColabManager]) -> None:
    global _manager
    with _lock:
        _manager = mgr
