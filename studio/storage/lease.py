"""Colab session lease manager to prevent multiple notebook runtimes from corrupting Drive."""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger("studio.storage.lease")

LEASE_FILENAME = "lease.json"
LEASE_TIMEOUT_SEC = 180.0  # 3 minutes


class LeaseError(Exception):
    pass


class ActiveLeaseError(LeaseError):
    def __init__(self, message: str, lease_data: Dict[str, Any]):
        super().__init__(message)
        self.lease_data = lease_data


class SessionLease:
    """Manages the lease file on durable storage with automated heartbeat."""

    def __init__(
        self,
        durable_dir: Union[str, Path],
        session_id: str,
        heartbeat_interval: float = 30.0,
    ):
        self.durable_dir = Path(durable_dir).resolve()
        self.lease_file = self.durable_dir / LEASE_FILENAME
        self.session_id = session_id
        self.heartbeat_interval = heartbeat_interval
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def read_lease(self) -> Optional[Dict[str, Any]]:
        if not self.lease_file.is_file():
            return None
        try:
            return json.loads(self.lease_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def acquire(self, force: bool = False) -> Dict[str, Any]:
        """Acquire or take over the session lease.

        Raises ActiveLeaseError if held by another active session (< 3 minutes since heartbeat).
        """
        self.durable_dir.mkdir(parents=True, exist_ok=True)
        existing = self.read_lease()
        now = time.time()

        if existing and not force:
            last_hb = float(existing.get("heartbeat_at", 0.0))
            elapsed = now - last_hb
            other_id = existing.get("session_id", "unknown")
            if other_id != self.session_id and elapsed < LEASE_TIMEOUT_SEC:
                raise ActiveLeaseError(
                    f"Colab lease held by active session {other_id} on {existing.get('hostname')} "
                    f"(last heartbeat {int(elapsed)}s ago). Wait 3m or use 'Take over'.",
                    lease_data=existing,
                )

        lease_payload = {
            "session_id": self.session_id,
            "started_at": existing.get("started_at", now) if (existing and existing.get("session_id") == self.session_id) else now,
            "heartbeat_at": now,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }

        tmp = self.durable_dir / f"{LEASE_FILENAME}.tmp.{self.session_id}"
        tmp.write_text(json.dumps(lease_payload, indent=2), encoding="utf-8")
        with tmp.open("rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(str(tmp), str(self.lease_file))

        self.start_heartbeat()
        return lease_payload

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._stop_heartbeat.clear()

        def _beat_loop():
            while not self._stop_heartbeat.wait(self.heartbeat_interval):
                try:
                    now = time.time()
                    existing = self.read_lease() or {}
                    existing.update(heartbeat_at=now, session_id=self.session_id, hostname=socket.gethostname())
                    tmp = self.durable_dir / f"{LEASE_FILENAME}.tmp.{self.session_id}"
                    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    with tmp.open("rb") as f:
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                    os.replace(str(tmp), str(self.lease_file))
                except Exception as e:
                    logger.warning("Error writing lease heartbeat: %s", e)

        self._heartbeat_thread = threading.Thread(target=_beat_loop, daemon=True, name="lease_heartbeat")
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def release(self) -> bool:
        self.stop_heartbeat()
        existing = self.read_lease()
        if existing and existing.get("session_id") == self.session_id:
            try:
                self.lease_file.unlink(missing_ok=True)
                return True
            except OSError:
                pass
        return False
