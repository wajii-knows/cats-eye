"""Background durable uploader queue with bounded parallelism and metrics for UI status."""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .durable import DurableStore

logger = logging.getLogger("studio.storage.uploader")


class UploadTask:
    def __init__(
        self,
        local_path: Union[str, Path],
        durable_rel_path: str,
        is_directory: bool = False,
        archive_name: Optional[str] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.local_path = Path(local_path)
        self.durable_rel_path = durable_rel_path
        self.is_directory = is_directory
        self.archive_name = archive_name
        self.on_complete = on_complete
        self.on_error = on_error
        self.size_bytes = self._estimate_size()
        self.retries = 0

    def _estimate_size(self) -> int:
        try:
            if self.is_directory and self.local_path.is_dir():
                return sum(f.stat().st_size for f in self.local_path.rglob("*") if f.is_file())
            elif self.local_path.is_file():
                return self.local_path.stat().st_size
        except OSError:
            pass
        return 0


class DurableUploader:
    """Processes uploads to durable Drive store asynchronously with retry and queue observability."""

    def __init__(self, durable_store: DurableStore, max_workers: int = 2):
        self.store = durable_store
        self.max_workers = max_workers
        self.queue: queue.Queue[UploadTask] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending_rels: set[str] = set()
        self._tombstones: dict[str, float] = {}
        self.pending_bytes = 0
        self.last_sync_time = time.time()
        self.last_error: Optional[str] = None

    @staticmethod
    def _norm(rel: str) -> str:
        return str(rel or "").replace("\\", "/").strip("/")

    def _is_tombstoned_locked(self, norm_rel: str) -> bool:
        if not norm_rel:
            return False
        for pref in self._tombstones:
            if norm_rel == pref or norm_rel.startswith(pref + "/"):
                return True
        return False

    def cancel_prefix(self, prefix: str) -> None:
        """Stop pending/retried uploads under `prefix` and prevent resurfaced writes for deleted resources."""
        norm = self._norm(prefix)
        if not norm:
            return
        with self._lock:
            self._tombstones[norm] = time.time()
            self._pending_rels = {
                r for r in self._pending_rels if not (r == norm or r.startswith(norm + "/"))
            }
            if self.last_error and (norm in self.last_error or "Local file does not exist" in self.last_error):
                self.last_error = None

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"durable_uploader_{i+1}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def enqueue_file(
        self,
        local_path: Union[str, Path],
        durable_rel_path: str,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> UploadTask:
        norm_rel = self._norm(durable_rel_path)
        task = UploadTask(
            local_path=local_path,
            durable_rel_path=durable_rel_path,
            is_directory=False,
            on_complete=on_complete,
            on_error=on_error,
        )
        with self._lock:
            if self._is_tombstoned_locked(norm_rel) or norm_rel in self._pending_rels:
                return task
            self._pending_rels.add(norm_rel)
            self.pending_bytes += task.size_bytes
        self.queue.put(task)
        return task

    def enqueue_directory(
        self,
        local_dir: Union[str, Path],
        durable_rel_path: str,
        archive_name: Optional[str] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> UploadTask:
        norm_rel = self._norm(durable_rel_path)
        task = UploadTask(
            local_path=local_dir,
            durable_rel_path=durable_rel_path,
            is_directory=True,
            archive_name=archive_name,
            on_complete=on_complete,
            on_error=on_error,
        )
        with self._lock:
            if self._is_tombstoned_locked(norm_rel) or norm_rel in self._pending_rels:
                return task
            self._pending_rels.add(norm_rel)
            self.pending_bytes += task.size_bytes
        self.queue.put(task)
        return task

    def is_idle(self) -> bool:
        with self._lock:
            return self.queue.empty() and self.pending_bytes == 0

    def get_status(self) -> Dict[str, Any]:
        """Return metrics for the UI top bar cloud sync chip."""
        with self._lock:
            pending_count = self.queue.qsize()
            bytes_left = self.pending_bytes
            err = self.last_error
            last_sync = self.last_sync_time

        if err:
            state = "error"
            label = "Sync problem"
        elif pending_count > 0:
            state = "syncing"
            mb = max(1, round(bytes_left / (1024 * 1024)))
            label = f"Syncing {mb} MB"
        else:
            state = "synced"
            label = "Synced"

        return {
            "state": state,
            "label": label,
            "pending_count": pending_count,
            "pending_bytes": bytes_left,
            "last_sync_time": last_sync,
            "last_error": err,
            "safe_to_disconnect": (pending_count == 0 and err is None),
        }

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            norm_rel = self._norm(task.durable_rel_path)
            with self._lock:
                self._pending_rels.discard(norm_rel)
                tombstoned = self._is_tombstoned_locked(norm_rel)

            success = False
            dropped = False
            try:
                exists_now = task.local_path.is_dir() if task.is_directory else task.local_path.is_file()
                if tombstoned or not exists_now:
                    dropped = True
                    continue

                if task.is_directory:
                    meta = self.store.put_dir_packed(task.local_path, task.durable_rel_path, archive_name=task.archive_name)
                else:
                    meta = self.store.put_file(task.local_path, task.durable_rel_path)
                success = True
                self.last_sync_time = time.time()
                self.last_error = None
                if task.on_complete:
                    try:
                        task.on_complete(meta)
                    except Exception as e:
                        logger.warning("Error in upload on_complete: %s", e)
            except FileNotFoundError:
                dropped = True
            except Exception as e:
                exists_after = task.local_path.is_dir() if task.is_directory else task.local_path.is_file()
                with self._lock:
                    tombstoned_after = self._is_tombstoned_locked(norm_rel)
                if tombstoned_after or not exists_after:
                    dropped = True
                    continue
                logger.warning("Durable upload failed for %s: %s", task.durable_rel_path, e)
                task.retries += 1
                if task.retries <= 3:
                    time.sleep(1.0 * task.retries)
                    with self._lock:
                        still_ok = (not self._is_tombstoned_locked(norm_rel)) and (
                            task.local_path.is_dir() if task.is_directory else task.local_path.is_file()
                        )
                        if still_ok:
                            self._pending_rels.add(norm_rel)
                            self.queue.put(task)
                        else:
                            dropped = True
                else:
                    self.last_error = str(e)
                    if task.on_error:
                        try:
                            task.on_error(e)
                        except Exception:
                            pass
            finally:
                if success or dropped or task.retries > 3:
                    with self._lock:
                        self.pending_bytes = max(0, self.pending_bytes - task.size_bytes)
                self.queue.task_done()
