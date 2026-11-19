"""Stage execution contract and fingerprinting for Cat's Eye.

A Stage declares inputs, parameters schema, outputs, cacheability, and a run(ctx) method.
Fingerprints are deterministic SHA-256 digests over source hash, canonical parameters,
stage metadata, and upstream fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .artifacts import ArtifactStore

logger = logging.getLogger("studio.pipeline.stage")

# (size, mtime) -> sha256 cache for streaming file hashes
_FILE_HASH_CACHE: Dict[str, Tuple[int, float, str]] = {}
_CACHE_LOCK = threading.Lock()


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Stream a large source file in 1MB chunks and return its SHA-256 digest.

    Caches the hash by (size, mtime) so unchanged files are not rehashed.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    st = p.stat()
    size, mtime = st.st_size, st.st_mtime
    key = str(p)

    with _CACHE_LOCK:
        cached = _FILE_HASH_CACHE.get(key)
        if cached and cached[0] == size and cached[1] == mtime:
            return cached[2]

    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    digest = h.hexdigest()

    with _CACHE_LOCK:
        _FILE_HASH_CACHE[key] = (size, mtime, digest)

    return digest


def canonical_json(data: Any) -> str:
    """Produce deterministic canonical JSON (sorted keys, compact separators)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_stage_fingerprint(
    stage_name: str,
    stage_version: int,
    source_identifier: str,
    params: Dict[str, Any],
    upstream_fingerprints: Optional[Dict[str, str]] = None,
) -> str:
    """Generate deterministic SHA-256 fingerprint for a stage run.

    Combines:
    - source identifier (content hash of source file or URL + media id)
    - canonical-JSON of params
    - stage name
    - stage version
    - canonical-JSON of upstream stage fingerprints
    """
    payload = {
        "source": str(source_identifier),
        "stage": str(stage_name),
        "version": int(stage_version),
        "params": params or {},
        "upstreams": upstream_fingerprints or {},
    }
    raw = canonical_json(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class StageError(Exception):
    """Execution error within a pipeline stage with retryable flag and error details."""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        detail: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }


class StageRunRecord(BaseModel):
    """Model tracking a single stage execution lifecycle."""

    id: str
    job_id: str
    stage: str
    status: str  # pending | running | done | failed | skipped_cached
    started_at: float
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None
    input_fingerprint: str
    output_artifacts: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None


class StageContext:
    """Execution context passed into Stage.run()."""

    def __init__(
        self,
        job: Dict[str, Any],
        user_dir: Path,
        user_id: str,
        store: ArtifactStore,
        inputs: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        upstream_fingerprints: Optional[Dict[str, str]] = None,
        cancel_event: Optional[threading.Event] = None,
        active_processes: Optional[List[subprocess.Popen]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.job = job
        self.user_dir = Path(user_dir)
        self.user_id = user_id
        self.store = store
        self.inputs = inputs or {}
        self.params = params or {}
        self.upstream_fingerprints = upstream_fingerprints or {}
        self.cancel_event = cancel_event or threading.Event()
        self.active_processes = active_processes if active_processes is not None else []
        self.logger = logger_instance or logger

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def register_process(self, proc: subprocess.Popen) -> None:
        self.active_processes.append(proc)

    def unregister_process(self, proc: subprocess.Popen) -> None:
        if proc in self.active_processes:
            self.active_processes.remove(proc)

    def kill_active_processes(self) -> None:
        for p in list(self.active_processes):
            try:
                p.kill()
            except Exception:
                pass
        self.active_processes.clear()


class Stage:
    """Base class for all pipeline processing stages."""

    name: str = ""
    version: int = 1
    inputs: Tuple[str, ...] = ()
    params_schema: Any = None
    outputs: Tuple[str, ...] = ()
    cacheable: bool = True

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        """Extract params relevant to this stage from the job options."""
        return ctx.params

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        """Execute stage computation.

        Returns a dictionary of stage outputs:
        - file paths to be saved as artifacts
        - or in-memory metadata / JSON objects
        """
        raise NotImplementedError
