"""Visual understanding pipeline orchestrator, lazy Drive unpacking, cache eviction & cleanup."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlalchemy as sa

from studio import config, ctx, db, jobs, media, safe
from studio.pipeline.artifacts import ArtifactStore, _validate_segment
from studio.pipeline.engine import PipelineEngine
from studio.pipeline.stage import StageContext, StageError, compute_file_sha256
from studio.pipeline.stages.detect_scenes import probe_visual_media

logger = logging.getLogger("studio.pipeline.visuals")

VISUAL_STAGES = (
    ("detect_scenes", 0.25, "Detecting scene cuts"),
    ("extract_snapshots", 0.55, "Extracting keyframes"),
    ("analyze_snapshots", 0.75, "Analyzing frames & layouts"),
    ("caption_snapshots", 0.90, "Captioning contact sheets"),
    ("build_moments", 1.00, "Linking timeline moments"),
)

_ANALYZE_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _job_lock(jid: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if jid not in _ANALYZE_LOCKS:
            _ANALYZE_LOCKS[jid] = threading.Lock()
        return _ANALYZE_LOCKS[jid]


def _default_durable_store(durable_store: Optional[Any] = None) -> Optional[Any]:
    if durable_store is not None:
        return durable_store
    if config.COLAB_MODE:
        try:
            from studio.storage.colab import get_colab_manager
            mgr = get_colab_manager(create=False)
            if mgr:
                return mgr.durable_store
        except Exception:
            pass
    return None


def _user_store(job: Dict[str, Any]) -> Tuple[str, Path, ArtifactStore]:
    uid = str(job.get("user") or ctx.uid_or_none() or "default")
    try:
        u_dir = config.user_dir(uid)
    except Exception:
        u_dir = Path(job.get("dir", config.DATA_DIR)).parent.parent
    store = ArtifactStore(user_dir=u_dir, user_id=uid)
    return uid, u_dir, store


def _ensure_stage_artifact(
    job: Dict[str, Any],
    stage_name: str,
    fp: Optional[str],
    store: ArtifactStore,
    durable_store: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    if not fp:
        return None
    man = store.get(fp, stage=stage_name)
    if man:
        return man
    d_store = _default_durable_store(durable_store)
    if d_store:
        uid = store.user_id
        for rp in (
            f"users/{uid}/artifacts/{stage_name}/{fp[:2]}/{fp}/{fp}.tar",
            f"artifacts/{stage_name}/{fp[:2]}/{fp}/{fp}.tar",
        ):
            if d_store.exists(rp):
                dest = store.root / stage_name / fp[:2] / fp
                d_store.get_dir_unpacked(rp, dest)
                man = store.get(fp, stage=stage_name)
                if man:
                    if stage_name == "extract_snapshots":
                        try:
                            store.evict_stage_cache(
                                "extract_snapshots",
                                int(config.SNAPSHOT_LOCAL_CACHE_MAX_MB * 1024 * 1024),
                                d_store,
                                protect_fingerprints={fp},
                            )
                        except Exception:
                            pass
                    return man
    return None


def _resolve_source_video(
    job: Dict[str, Any],
    d_store: Optional[Any] = None,
    explicit: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    src_obj = job.get("source") or {}
    cands = [explicit, job.get("source_path"), src_obj.get("path")]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    job_dir = Path(job.get("dir", ""))
    if job_dir.is_dir():
        for ext in ("mp4", "mkv", "mov", "webm", "m4a", "mp3", "wav"):
            p = job_dir / f"source.{ext}"
            if p.is_file():
                return p
    # Check durable Drive backup
    if d_store:
        uid = job.get("user", "default")
        jid = job.get("id", "job")
        for rp in d_store.list(f"users/{uid}/jobs/{jid}/sources/"):
            local_p = job_dir / Path(rp).name
            d_store.get_file(rp, local_p)
            if local_p.is_file():
                job["source_path"] = str(local_p)
                return local_p
    # URL job fallback when full source video wasn't retained locally
    if src_obj.get("type") == "url" and src_obj.get("url") and job_dir.is_dir():
        q = int((job.get("options") or {}).get("quality", 720) or 720)
        dl = media.download(src_obj["url"], job_dir, q)
        if dl and dl.is_file():
            job["source_path"] = str(dl)
            return dl
    return None


def cleanup_job_artifacts(job: Dict[str, Any], durable_store: Optional[Any] = None) -> None:
    """Remove visual/pipeline artifacts, checkpoints, and DB rows owned only by this job."""
    uid, _u_dir, store = _resolve_store_for_job(job)
    d_store = _default_durable_store(durable_store)
    jid = str(job.get("id", ""))
    vis_arts = (job.get("visuals") or {}).get("artifacts") or {}

    # Preserve shared content-addressed artifacts if another job of the same user still uses them
    still_used: set[Tuple[str, str]] = set()
    try:
        with ctx.as_user(uid):
            for other in jobs.list_jobs():
                if other.get("id") == jid:
                    continue
                for stg, fp in ((other.get("visuals") or {}).get("artifacts") or {}).items():
                    if fp:
                        still_used.add((str(stg), str(fp)))
    except Exception:
        pass

    for stg, fp in vis_arts.items():
        if not fp or (str(stg), str(fp)) in still_used:
            continue
        try:
            store.delete(fp, stage=stg)
        except Exception as e:
            logger.warning("Failed deleting local artifact %s/%s: %s", stg, fp, e)
        if d_store:
            for rp in (
                f"users/{uid}/artifacts/{stg}/{fp[:2]}/{fp}/{fp}.tar",
                f"artifacts/{stg}/{fp[:2]}/{fp}/{fp}.tar",
            ):
                try:
                    d_store.delete(rp)
                except Exception:
                    pass

    # Clean per-job intra-stage checkpoints locally and on Drive
    chk_local = store.user_dir / "checkpoints" / jid
    if chk_local.is_dir():
        shutil.rmtree(str(chk_local), ignore_errors=True)
    if d_store:
        for pref in (f"users/{uid}/checkpoints/{jid}/", f"checkpoints/{jid}/"):
            for rp in d_store.list(pref):
                try:
                    d_store.delete(rp)
                except Exception:
                    pass

    try:
        with db.engine.begin() as conn:
            conn.execute(db.stage_runs.delete().where(db.stage_runs.c.job_id == jid))
    except Exception:
        pass


def _resolve_store_for_job(job: Dict[str, Any]) -> Tuple[str, Path, ArtifactStore]:
    job_dir = Path(job.get("dir", ""))
    # If job lives inside an isolated workspace (e.g. tests with custom user_dir), keep artifacts beside it
    if job_dir.parent.name == "jobs" and job_dir.parent.parent.exists():
        uid = str(job.get("user") or ctx.uid_or_none() or "default")
        u_dir = job_dir.parent.parent
        return uid, u_dir, ArtifactStore(user_dir=u_dir, user_id=uid)
    return _user_store(job)


def load_visual_data(
    job: Dict[str, Any],
    stage_name: str,
    filename: str,
    durable_store: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Load JSON payload from a visual stage artifact, lazily unpacking from Drive on local miss."""
    _uid, _u_dir, store = _resolve_store_for_job(job)
    d_store = _default_durable_store(durable_store)
    vis = job.get("visuals") or {}
    fp = (vis.get("artifacts") or {}).get(stage_name)
    if fp:
        man = _ensure_stage_artifact(job, stage_name, fp, store, d_store)
        if man and man.get("_dir"):
            f_path = (Path(man["_dir"]) / filename).resolve()
            if f_path.is_relative_to(Path(man["_dir"]).resolve()) and f_path.is_file():
                try:
                    return json.loads(f_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

    # Fallback: scan job staging folder if still running or just completed
    job_dir = Path(job.get("dir", ""))
    for cand in job_dir.glob(f"visual_work/**/{filename}"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def resolve_snapshot_image(
    job: Dict[str, Any],
    sid: str,
    size: str = "thumb",
    durable_store: Optional[Any] = None,
) -> Optional[Tuple[Path, str]]:
    """Resolve snapshot WebP/JPEG image path + MIME type, lazily fetching packs or extracting full frame."""
    safe_sid = _validate_segment(sid, "snapshot id")
    if size not in ("thumb", "analysis", "full"):
        raise ValueError("Invalid size")

    _uid, _u_dir, store = _resolve_store_for_job(job)
    d_store = _default_durable_store(durable_store)
    vis = job.get("visuals") or {}
    ext_fp = (vis.get("artifacts") or {}).get("extract_snapshots")

    if size in ("thumb", "analysis"):
        rel_inside = f"thumbs/{safe_sid}.webp" if size == "thumb" else f"analysis/{safe_sid}.jpg"
        mime = "image/webp" if size == "thumb" else "image/jpeg"
        if ext_fp:
            man = _ensure_stage_artifact(job, "extract_snapshots", ext_fp, store, d_store)
            if man and man.get("_dir"):
                art_root = Path(man["_dir"]).resolve()
                cand = (art_root / rel_inside).resolve()
                if cand.is_relative_to(art_root) and cand.is_file():
                    return cand, mime
        # Staging directory fallback
        staging_cand = (Path(job.get("dir", "")) / "visual_work" / "snapshots_stage" / rel_inside).resolve()
        if staging_cand.is_file():
            return staging_cand, mime
        return None

    # size == "full": look up timestamp, extract lazily at original resolution, and cache in job/visual_full
    snaps_data = load_visual_data(job, "analyze_snapshots", "analyzed_snapshots.json", d_store) or load_visual_data(
        job, "extract_snapshots", "snapshots.json", d_store
    )
    if not snaps_data:
        return None
    snap = next((s for s in snaps_data.get("snapshots", []) if s.get("id") == safe_sid), None)
    if not snap:
        return None

    full_dir = (Path(job["dir"]) / "visual_full").resolve()
    full_dir.mkdir(parents=True, exist_ok=True)
    target = (full_dir / f"{safe_sid}.jpg").resolve()
    if not target.is_relative_to(full_dir):
        raise ValueError("Invalid snapshot target path")
    if target.is_file():
        return target, "image/jpeg"

    src_video = _resolve_source_video(job, d_store=d_store)
    if not src_video or not src_video.is_file():
        return None
    ts = max(0.0, float(snap["timestamp"]))
    tmp_jpg = full_dir / f"{safe_sid}.tmp.jpg"
    try:
        media.run([
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", f"{ts:.3f}", "-i", str(src_video),
            "-frames:v", "1", "-q:v", "2", str(tmp_jpg),
        ], timeout=60)
        if tmp_jpg.is_file():
            tmp_jpg.replace(target)
            return target, "image/jpeg"
    finally:
        tmp_jpg.unlink(missing_ok=True)
    return None


def run_visual_analysis(
    job_id: str,
    force: bool = False,
    durable_store: Optional[Any] = None,
    engine: Optional[PipelineEngine] = None,
    source_video: Optional[Union[str, Path]] = None,
    params: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Execute the 5 visual pipeline stages for a job and update job['visuals']."""
    job = jobs.get(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    lock = _job_lock(job_id)
    with lock:
        if engine and engine.store:
            store = engine.store
            u_dir = store.user_dir
            uid = store.user_id
        else:
            uid, u_dir, store = _user_store(job)
        d_store = _default_durable_store(durable_store or (engine.durable_store if engine else None))
        eng = engine or PipelineEngine(store=store, durable_store=d_store)

        if force:
            cleanup_job_artifacts(job, durable_store=d_store)

        with ctx.as_user(uid):
            prev_vis = dict(job.get("visuals") or {})
            vis_state: Dict[str, Any] = {
                "status": "analyzing",
                "progress": 0.02,
                "stage": "Preparing visual pipeline",
                "error": None,
                "artifacts": prev_vis.get("artifacts", {}) if not force else {},
                "counts": prev_vis.get("counts", {}),
                "facecam_box": prev_vis.get("facecam_box"),
                "captions_status": prev_vis.get("captions_status", ""),
                "captions_message": prev_vis.get("captions_message", ""),
                "pending": False,
                "pending_from": None,
            }
            job["visuals"] = vis_state
            jobs.save(job)

            try:
                src_path = _resolve_source_video(job, d_store=d_store, explicit=source_video)
                job_dir = Path(job["dir"])
                is_live_buf = job.get("kind") == "live" and (job_dir / "buffer").is_dir() and any((job_dir / "buffer").glob("*.ts"))

                if not src_path and not is_live_buf:
                    raise StageError("SOURCE_MISSING", "Source media is no longer available on disk for visual analysis.")

                if src_path and src_path.is_file():
                    source_id = compute_file_sha256(src_path)
                    info = probe_visual_media(src_path)
                    job.setdefault("duration", info.get("duration") or 0.0)
                    job.setdefault("width", info.get("width") or 0)
                    job.setdefault("height", info.get("height") or 0)
                else:
                    source_id = f"live_{job_id}"

                inputs: Dict[str, Any] = {}
                if src_path:
                    inputs["source_video"] = str(src_path)
                    inputs["video_fingerprint"] = source_id
                if job.get("clips"):
                    inputs["candidate_clips"] = list(job["clips"])

                sctx = StageContext(
                    job=job,
                    user_dir=u_dir,
                    user_id=uid,
                    store=store,
                    inputs=inputs,
                    params=dict(params or {}),
                    cancel_event=cancel_event,
                    durable_store=d_store,
                )

                for idx, (stg_name, frac, label) in enumerate(VISUAL_STAGES, start=1):
                    vis_state.update({"stage": label, "progress": max(vis_state["progress"], round(frac - 0.12, 2))})
                    job["visuals"] = dict(vis_state)
                    jobs.save(job)

                    stg = eng.stages[stg_name]
                    eng.execute_stage(
                        stage=stg,
                        ctx=sctx,
                        source_id=source_id,
                        stage_index=idx,
                        total_stages=len(VISUAL_STAGES),
                    )
                    fp = sctx.upstream_fingerprints.get(stg_name)
                    if fp:
                        vis_state["artifacts"][stg_name] = fp

                scenes_json = sctx.inputs.get("analyzed_scenes_json") or sctx.inputs.get("scenes_json")
                snaps_json = sctx.inputs.get("analyzed_snapshots_json") or sctx.inputs.get("snapshots_json")
                moments_json = sctx.inputs.get("moments_json")

                scenes_count = len(json.loads(Path(scenes_json).read_text(encoding="utf-8")).get("scenes", [])) if scenes_json and Path(scenes_json).is_file() else 0
                snaps_count = len(json.loads(Path(snaps_json).read_text(encoding="utf-8")).get("snapshots", [])) if snaps_json and Path(snaps_json).is_file() else 0
                moments_count = len(json.loads(Path(moments_json).read_text(encoding="utf-8")).get("moments", [])) if moments_json and Path(moments_json).is_file() else 0

                vis_state.update({
                    "status": "done",
                    "progress": 1.0,
                    "stage": "Visual analysis complete",
                    "error": None,
                    "counts": {"scenes": scenes_count, "snapshots": snaps_count, "moments": moments_count},
                    "facecam_box": sctx.inputs.get("facecam_box"),
                    "captions_status": sctx.inputs.get("captions_status", "not_available"),
                    "captions_message": sctx.inputs.get("captions_message", ""),
                    "pending": bool(sctx.inputs.get("pending")),
                    "pending_from": sctx.inputs.get("pending_from"),
                    "has_video": bool(sctx.inputs.get("has_video", True)),
                })
                job["visuals"] = vis_state
                jobs.save(job)
                return vis_state
            except Exception as e:
                logger.warning("Visual analysis failed for job %s: %s", job_id, e)
                vis_state.update({
                    "status": "failed",
                    "stage": "Visual analysis failed",
                    "error": safe.friendly(e, 300),
                })
                job["visuals"] = vis_state
                jobs.save(job)
                raise
