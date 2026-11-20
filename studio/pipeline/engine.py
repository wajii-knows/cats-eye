"""DAG pipeline execution engine for Cat's Eye.

Executes pipeline stages as a directed acyclic graph (DAG):
    fetch -> probe -> extract_audio -> transcribe -> select_clips -> render_clip -> post
With:
- Content-addressed artifact caching (skipped_cached)
- Crash recovery and automatic resume from the last completed stage
- Exponential backoff retry for retryable stage errors (up to 3 times)
- Child process cleanup on job cancellation
- Structured JSON logging and observability metrics (timing, cache badges, memory deltas)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sqlalchemy as sa
from studio import config, ctx, db, load
from studio.pipeline.artifacts import ArtifactStore
from studio.storage.durable import DurableStore
from studio.storage.uploader import DurableUploader
from studio.storage.db_checkpoint import DatabaseCheckpointManager
from studio.pipeline.stage import (
    Stage,
    StageContext,
    StageError,
    StageRunRecord,
    compute_file_sha256,
    compute_stage_fingerprint,
)
from studio.pipeline.stages import (
    AnalyzeSnapshotsStage,
    BuildMomentsStage,
    CaptionSnapshotsStage,
    DetectScenesStage,
    ExtractAudioStage,
    ExtractSnapshotsStage,
    FetchStage,
    PostStage,
    ProbeStage,
    RenderClipStage,
    SelectClipsStage,
    TranscribeStage,
)

logger = logging.getLogger("studio.pipeline.engine")


def get_process_memory_mb() -> float:
    """Return the current process memory working set in megabytes."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if fn(h, ctypes.byref(pmc), pmc.cb):
                return round(pmc.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
    else:
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        except Exception:
            pass
    return 0.0


class PipelineEngine:
    """Orchestrates stage runs, artifact caching, crash resume, and failure recovery."""

    def __init__(
        self,
        store: Optional[ArtifactStore] = None,
        durable_store: Optional[DurableStore] = None,
        durable_uploader: Optional[DurableUploader] = None,
        db_checkpoint_mgr: Optional[DatabaseCheckpointManager] = None,
    ):
        self.store = store or ArtifactStore()
        self.durable_store = durable_store
        self.durable_uploader = durable_uploader
        self.db_checkpoint_mgr = db_checkpoint_mgr
        self.stages: Dict[str, Stage] = {
            "fetch": FetchStage(),
            "probe": ProbeStage(),
            "extract_audio": ExtractAudioStage(),
            "transcribe": TranscribeStage(),
            "select_clips": SelectClipsStage(),
            "render_clip": RenderClipStage(),
            "post": PostStage(),
            "detect_scenes": DetectScenesStage(),
            "extract_snapshots": ExtractSnapshotsStage(),
            "analyze_snapshots": AnalyzeSnapshotsStage(),
            "caption_snapshots": CaptionSnapshotsStage(),
            "build_moments": BuildMomentsStage(),
        }

    def _log_event(self, event_name: str, **kwargs) -> None:
        payload = {"event": event_name, "t": round(time.time(), 3), **kwargs}
        logger.info(json.dumps(payload))

    def _record_stage_run(
        self,
        run_id: str,
        job_id: str,
        stage_name: str,
        status: str,
        started_at: float,
        finished_at: Optional[float] = None,
        duration_ms: Optional[float] = None,
        fingerprint: str = "",
        output_artifacts: Optional[List[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        local_done: bool = False,
        durable_done: bool = False,
    ) -> None:
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    db.stage_runs.delete().where(db.stage_runs.c.id == run_id)
                )
                vals = {
                    "id": run_id,
                    "job_id": job_id,
                    "stage": stage_name,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "input_fingerprint": fingerprint,
                    "output_artifacts": json.dumps(output_artifacts or []),
                    "error": json.dumps(error) if error else None,
                }
                cols = {c["name"] for c in sa.inspect(conn).get_columns("stage_runs")}
                if "local_done" in cols:
                    vals["local_done"] = local_done
                if "durable_done" in cols:
                    vals["durable_done"] = durable_done
                conn.execute(db.stage_runs.insert().values(**vals))
        except Exception as e:
            logger.warning("Could not record stage run %s: %s", run_id, e)

    def get_last_completed_stage(self, job_id: str) -> Optional[str]:
        """Find the last completed stage for a job from the database."""
        try:
            with db.engine.connect() as conn:
                cols = {c["name"] for c in sa.inspect(conn).get_columns("stage_runs")}
                if "durable_done" in cols and (config.COLAB_MODE or self.durable_store):
                    rows = conn.execute(
                        sa.select(db.stage_runs.c.stage, db.stage_runs.c.status, db.stage_runs.c.durable_done)
                        .where(db.stage_runs.c.job_id == job_id)
                        .order_by(db.stage_runs.c.started_at.asc())
                    ).fetchall()
                    last = None
                    for stg, status, dur_done in rows:
                        if status == "skipped_cached" or (status == "done" and bool(dur_done)):
                            last = stg
                    return last
                else:
                    rows = conn.execute(
                        sa.select(db.stage_runs.c.stage, db.stage_runs.c.status)
                        .where(db.stage_runs.c.job_id == job_id)
                        .order_by(db.stage_runs.c.started_at.asc())
                    ).fetchall()
                    last = None
                    for stg, status in rows:
                        if status in ("done", "skipped_cached"):
                            last = stg
                    return last
        except Exception:
            return None

    def execute_stage(
        self,
        stage: Stage,
        ctx: StageContext,
        source_id: str,
        stage_index: int = 1,
        total_stages: int = 7,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        job_id = ctx.job["id"]
        stage_name = stage.name
        run_id = f"{job_id}_{stage_name}_{int(time.time() * 1000)}"
        params = stage.get_params(ctx)

        # 1. Compute deterministic fingerprint (exclude self from upstreams so re-runs hit cache)
        upstreams = {k: v for k, v in (ctx.upstream_fingerprints or {}).items() if k != stage.name}
        fp = compute_stage_fingerprint(
            stage_name=stage.name,
            stage_version=stage.version,
            source_identifier=source_id,
            params=params,
            upstream_fingerprints=upstreams,
        )

        ctx.job.setdefault("stages", {})

        ctx.stage_name = stage.name
        if self.durable_store and not ctx.durable_store:
            ctx.durable_store = self.durable_store

        # 2. Check artifact store for cache hit (locally or on durable Drive store)
        if stage.cacheable:
            cached_manifest = ctx.store.get(fp, stage=stage.name)
            d_store = self.durable_store or ctx.durable_store
            pack_candidates = (
                f"users/{ctx.user_id}/artifacts/{stage.name}/{fp[:2]}/{fp}/{fp}.tar",
                f"artifacts/{stage.name}/{fp[:2]}/{fp}/{fp}.tar",
            )
            if d_store:
                durable_has_pack = any(d_store.exists(rp) for rp in pack_candidates)
                if cached_manifest and not durable_has_pack:
                    # A stage counts as complete for resume only once verified in DurableStore
                    cached_manifest = None
                elif not cached_manifest and durable_has_pack:
                    for rel_pack in pack_candidates:
                        if d_store.exists(rel_pack):
                            try:
                                dest_local = ctx.store.root / stage.name / fp[:2] / fp
                                d_store.get_dir_unpacked(rel_pack, dest_local)
                                cached_manifest = ctx.store.get(fp, stage=stage.name)
                                if cached_manifest:
                                    break
                            except Exception as e:
                                logger.warning("Error restoring artifact from durable store: %s", e)

            if cached_manifest:
                self._log_event(
                    "stage_skipped_cached",
                    stage=stage.name,
                    job_id=job_id,
                    fingerprint=fp,
                    duration_ms=0.0,
                )
                self._record_stage_run(
                    run_id=run_id,
                    job_id=job_id,
                    stage_name=stage.name,
                    status="skipped_cached",
                    started_at=time.time(),
                    finished_at=time.time(),
                    duration_ms=0.0,
                    fingerprint=fp,
                    output_artifacts=list(cached_manifest.get("sha256_map", {}).keys()),
                    local_done=True,
                    durable_done=True,
                )
                ctx.job["stages"][stage.name] = {
                    "status": "skipped_cached",
                    "cached": True,
                    "durable_done": True,
                    "duration_ms": 0.0,
                    "fingerprint": fp,
                }
                ctx.upstream_fingerprints[stage.name] = fp
                cached_outputs = dict(cached_manifest.get("metadata", {}).get("outputs", {}))
                art_dir = cached_manifest.get("_dir")
                if art_dir:
                    cached_outputs["_artifact_dir"] = art_dir
                    art_path = Path(art_dir)
                    for k, v in list(cached_outputs.items()):
                        if isinstance(v, str) and not k.startswith("_"):
                            cand = art_path / Path(v).name
                            if cand.is_file():
                                cached_outputs[k] = str(cand)
                    if "transcript_words" not in cached_outputs and (art_path / "words.json").is_file():
                        try:
                            cached_outputs["transcript_words"] = json.loads(
                                (art_path / "words.json").read_text(encoding="utf-8")
                            )
                        except Exception:
                            pass
                ctx.inputs.update(cached_outputs)
                if progress_cb:
                    progress_cb(stage_index / total_stages, f"{stage.name} (cached)")
                return cached_outputs

        # 3. Check cancellation before running
        if ctx.is_cancelled():
            self._record_stage_run(
                run_id=run_id,
                job_id=job_id,
                stage_name=stage.name,
                status="failed",
                started_at=time.time(),
                finished_at=time.time(),
                duration_ms=0.0,
                fingerprint=fp,
                error={"cancelled": True, "message": "Job cancelled"},
            )
            ctx.job["stages"][stage.name] = {
                "status": "failed",
                "cancelled": True,
                "fingerprint": fp,
            }
            raise StageError("CANCELLED", "Job was cancelled by user", retryable=False, detail={"cancelled": True})

        # 4. Execute stage with retry on retryable errors
        t0 = time.time()
        start_mem = get_process_memory_mb()
        self._record_stage_run(
            run_id=run_id,
            job_id=job_id,
            stage_name=stage.name,
            status="running",
            started_at=t0,
            fingerprint=fp,
        )
        ctx.job["stages"][stage.name] = {
            "status": "running",
            "started_at": t0,
            "fingerprint": fp,
        }
        self._log_event(
            "stage_started",
            stage=stage.name,
            job_id=job_id,
            fingerprint=fp,
            mem_mb=start_mem,
        )
        if progress_cb:
            progress_cb((stage_index - 0.5) / total_stages, f"Running {stage.name}")

        retries = 0
        max_retries = 3
        backoff = 1.0

        outputs = {}
        while True:
            try:
                outputs = stage.run(ctx)
                break
            except StageError as e:
                if e.retryable and retries < max_retries:
                    retries += 1
                    self._log_event(
                        "stage_retry",
                        stage=stage.name,
                        job_id=job_id,
                        retry=retries,
                        error=e.message,
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    duration_ms = (time.time() - t0) * 1000
                    self._record_stage_run(
                        run_id=run_id,
                        job_id=job_id,
                        stage_name=stage.name,
                        status="failed",
                        started_at=t0,
                        finished_at=time.time(),
                        duration_ms=duration_ms,
                        fingerprint=fp,
                        error=e.to_dict(),
                    )
                    ctx.job["stages"][stage.name] = {
                        "status": "failed",
                        "error": e.message,
                        "duration_ms": duration_ms,
                        "fingerprint": fp,
                    }
                    self._log_event(
                        "stage_failed",
                        stage=stage.name,
                        job_id=job_id,
                        fingerprint=fp,
                        duration_ms=duration_ms,
                        error=e.message,
                    )
                    raise
            except Exception as e:
                duration_ms = (time.time() - t0) * 1000
                err_dict = {"code": "UNHANDLED_ERROR", "message": str(e), "retryable": False}
                self._record_stage_run(
                    run_id=run_id,
                    job_id=job_id,
                    stage_name=stage.name,
                    status="failed",
                    started_at=t0,
                    finished_at=time.time(),
                    duration_ms=duration_ms,
                    fingerprint=fp,
                    error=err_dict,
                )
                ctx.job["stages"][stage.name] = {
                    "status": "failed",
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "fingerprint": fp,
                }
                self._log_event(
                    "stage_failed",
                    stage=stage.name,
                    job_id=job_id,
                    fingerprint=fp,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise

        t1 = time.time()
        duration_ms = (t1 - t0) * 1000
        end_mem = get_process_memory_mb()
        mem_delta = round(end_mem - start_mem, 2)

        # 5. Persist output artifacts if cacheable (skip raw media > 50MB and inline transcript words)
        files_to_store = {}
        for k, v in outputs.items():
            if k in ("source_video", "audio_wav", "audio_file", "proxy_video", "artifact_dirs"):
                continue
            if isinstance(v, (str, Path)) and Path(v).is_file():
                p = Path(v)
                try:
                    if p.stat().st_size <= 50 * 1024 * 1024:
                        files_to_store[p.name] = p
                except OSError:
                    pass

        artifact_dirs = outputs.get("artifact_dirs") if isinstance(outputs.get("artifact_dirs"), dict) else None
        meta_outputs = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in outputs.items()
            if k not in ("transcript_words", "artifact_dirs")
        }
        if artifact_dirs:
            meta_outputs["packed_subdirs"] = list(artifact_dirs.keys())

        stored_artifact_names = list(files_to_store.keys())
        if stage.cacheable:
            try:
                put_manifest = ctx.store.put(
                    stage=stage.name,
                    fingerprint=fp,
                    files=files_to_store,
                    metadata={"outputs": meta_outputs},
                    version=stage.version,
                    dirs=artifact_dirs,
                )
                stored_artifact_names = list(put_manifest.get("sha256_map", {}).keys()) or stored_artifact_names
                if put_manifest.get("_dir"):
                    outputs["_artifact_dir"] = put_manifest["_dir"]
            except Exception as e:
                logger.warning("Could not cache artifact for %s: %s", stage.name, e)

        d_store = self.durable_store or ctx.durable_store
        durable_synced = False
        if d_store and stage.cacheable:
            dest_local = ctx.store.root / stage.name / fp[:2] / fp
            if dest_local.is_dir():
                try:
                    d_store.put_dir_packed(
                        dest_local,
                        f"users/{ctx.user_id}/artifacts/{stage.name}/{fp[:2]}/{fp}",
                        archive_name=f"{fp}.tar",
                    )
                    durable_synced = True
                except Exception as e:
                    err = StageError(
                        "DURABLE_SYNC_FAILED",
                        f"Could not sync {stage.name} artifact pack to durable store: {e}",
                        retryable=True,
                    )
                    self._record_stage_run(
                        run_id=run_id,
                        job_id=job_id,
                        stage_name=stage.name,
                        status="failed",
                        started_at=t0,
                        finished_at=time.time(),
                        duration_ms=(time.time() - t0) * 1000,
                        fingerprint=fp,
                        output_artifacts=stored_artifact_names,
                        error=err.to_dict(),
                        local_done=True,
                        durable_done=False,
                    )
                    ctx.job["stages"][stage.name] = {
                        "status": "failed",
                        "error": err.message,
                        "duration_ms": round((time.time() - t0) * 1000, 1),
                        "fingerprint": fp,
                        "durable_done": False,
                    }
                    raise err from e
            if durable_synced and stage.name == "extract_snapshots":
                # Clean up temporary per-batch checkpoint packs once full artifact pack is verified on Drive
                try:
                    batch_prefix = f"users/{ctx.user_id}/checkpoints/{job_id}/extract_snapshots_batch_"
                    for rel_bp in d_store.list(batch_prefix):
                        d_store.delete(rel_bp)
                except Exception:
                    pass
                try:
                    ctx.store.evict_stage_cache(
                        "extract_snapshots",
                        int(config.SNAPSHOT_LOCAL_CACHE_MAX_MB * 1024 * 1024),
                        d_store,
                        protect_fingerprints={fp},
                    )
                except Exception as e:
                    logger.warning("Snapshot cache eviction check failed: %s", e)
        elif not d_store:
            durable_synced = True

        self._record_stage_run(
            run_id=run_id,
            job_id=job_id,
            stage_name=stage.name,
            status="done",
            started_at=t0,
            finished_at=t1,
            duration_ms=duration_ms,
            fingerprint=fp,
            output_artifacts=stored_artifact_names,
            local_done=True,
            durable_done=durable_synced,
        )

        if self.db_checkpoint_mgr:
            try:
                self.db_checkpoint_mgr.create_checkpoint(reason=f"stage_{stage.name}")
            except Exception as e:
                logger.warning("DB checkpoint failed after %s: %s", stage.name, e)
        ctx.job["stages"][stage.name] = {
            "status": "done",
            "cached": False,
            "durable_done": durable_synced,
            "duration_ms": round(duration_ms, 1),
            "fingerprint": fp,
        }
        self._log_event(
            "stage_finished",
            stage=stage.name,
            job_id=job_id,
            fingerprint=fp,
            duration_ms=round(duration_ms, 1),
            mem_delta_mb=mem_delta,
        )

        ctx.upstream_fingerprints[stage.name] = fp
        ctx.inputs.update(outputs)
        if progress_cb:
            progress_cb(stage_index / total_stages, f"{stage.name} done")
        return outputs

    def run_pipeline(
        self,
        job: Dict[str, Any],
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Execute the standard Cat's Eye pipeline DAG for a job."""
        user_id = job.get("user") or ctx.uid_or_none() or "default"
        if self.store:
            store = self.store
            user_dir = store.user_dir
        else:
            user_dir = Path(job.get("dir", config.DATA_DIR)).parent.parent
            store = ArtifactStore(user_dir=user_dir, user_id=user_id)
        job_ctx = StageContext(
            job=job,
            user_dir=user_dir,
            user_id=user_id,
            store=store,
            cancel_event=cancel_event,
        )

        src = job.get("source", {})
        if src.get("type") == "url":
            source_id = f"url_{src.get('url')}"
        else:
            p = Path(src.get("path", ""))
            source_id = compute_file_sha256(p) if p.is_file() else f"file_{p.name}"

        stage_order = [
            ("fetch", self.stages["fetch"]),
            ("probe", self.stages["probe"]),
            ("extract_audio", self.stages["extract_audio"]),
            ("transcribe", self.stages["transcribe"]),
            ("select_clips", self.stages["select_clips"]),
        ]

        total_steps = len(stage_order) + 2  # plus render and post
        step_idx = 1

        for name, stg in stage_order:
            self.execute_stage(
                stage=stg,
                ctx=job_ctx,
                source_id=source_id,
                stage_index=step_idx,
                total_stages=total_steps,
                progress_cb=progress_cb,
            )
            step_idx += 1

        # Render clips
        clips_spec = job_ctx.inputs.get("clips_spec", [])
        rendered_clips = []
        render_stg = self.stages["render_clip"]

        for i, clip in enumerate(clips_spec):
            job_ctx.inputs["current_clip"] = clip
            clip_res = self.execute_stage(
                stage=render_stg,
                ctx=job_ctx,
                source_id=f"{source_id}_clip_{i}",
                stage_index=step_idx,
                total_stages=total_steps,
                progress_cb=progress_cb,
            )
            rendered_clips.append(clip_res)

        step_idx += 1
        # Optional post stage
        post_stg = self.stages["post"]
        post_res = self.execute_stage(
            stage=post_stg,
            ctx=job_ctx,
            source_id=source_id,
            stage_index=step_idx,
            total_stages=total_steps,
            progress_cb=progress_cb,
        )

        return {
            "stages": job.get("stages", {}),
            "clips": rendered_clips,
            "post": post_res,
        }
