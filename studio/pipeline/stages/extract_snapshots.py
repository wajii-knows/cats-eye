"""ExtractSnapshotsStage (v1): batched FFmpeg extraction, 64-bit dHash dedupe, WebP/JPEG output & budget thinning."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from studio import config
from studio.contracts import Snapshot, make_snapshot_id
from studio.pipeline.stage import Stage, StageContext, StageError, compute_file_sha256

logger = logging.getLogger("studio.pipeline.extract_snapshots")

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SENTENCE_END_RE = re.compile(r"[.?!]['\"\)]?$")


def compute_dhash(img: Image.Image) -> str:
    """Compute 64-bit difference hash (dHash) returned as a 16-char lowercase hex string."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two 16-char hex 64-bit hashes."""
    if not h1 or not h2:
        return 64
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def plan_snapshot_timestamps(
    scenes: List[Dict[str, Any]],
    duration: float,
    words: Optional[List[Dict[str, Any]]] = None,
    candidate_clips: Optional[List[Any]] = None,
    interval_sec: float = 10.0,
    clip_interval_sec: float = 3.0,
    dedupe_sec: float = 0.5,
) -> List[Tuple[float, bool]]:
    """Plan sorted, monotonically increasing (timestamp, is_scene_boundary) targets."""
    dur = float(duration or 0.0)
    if dur <= 0.0:
        return []
    max_t = max(0.0, round(dur - 0.08, 3))
    cands: List[Tuple[float, bool]] = []

    def add(t_val: float, is_boundary: bool = False) -> None:
        t_clamped = round(min(max_t, max(0.0, float(t_val))), 3)
        cands.append((t_clamped, bool(is_boundary)))

    # 1. Scene start + midpoint + (end - 0.2s)
    for sc in scenes or []:
        s = float(sc["start"])
        e = float(sc["end"])
        add(s, is_boundary=True)
        if e - s >= 0.3:
            add(0.5 * (s + e), is_boundary=False)
        if e - s >= 0.8:
            add(max(s, e - 0.2), is_boundary=True)

    # 2. Fixed intervals (3s inside candidate clips, 10s elsewhere)
    clip_spans: List[Tuple[float, float]] = []
    for cl in candidate_clips or []:
        if isinstance(cl, dict) and "start" in cl and "end" in cl:
            clip_spans.append((float(cl["start"]), float(cl["end"])))
        elif isinstance(cl, (list, tuple)) and len(cl) >= 2:
            clip_spans.append((float(cl[0]), float(cl[1])))

    base_step = max(0.5, float(interval_sec))
    t = 0.0
    while t < max_t:
        add(t, is_boundary=False)
        t += base_step

    clip_step = max(0.5, float(clip_interval_sec))
    for cs, ce in clip_spans:
        ct = max(0.0, cs)
        c_hi = min(max_t, ce)
        while ct <= c_hi:
            add(ct, is_boundary=False)
            ct += clip_step

    # 3. Transcript sentence boundaries
    if words:
        prev_e: Optional[float] = None
        prev_txt = ""
        for i, w in enumerate(words):
            try:
                ws = float(w.get("s", w.get("start", 0.0)))
                we = float(w.get("e", w.get("end", ws)))
                wt = str(w.get("w", w.get("word", ""))).strip()
            except (TypeError, ValueError):
                continue
            if i == 0 or (prev_e is not None and (ws - prev_e >= 1.5 or _SENTENCE_END_RE.search(prev_txt))):
                add(ws, is_boundary=False)
            prev_e = we
            prev_txt = wt

    if not cands:
        add(0.0, is_boundary=True)

    # Sort with scene boundaries winning same-timestamp ties
    cands.sort(key=lambda x: (x[0], 0 if x[1] else 1))

    kept: List[Tuple[float, bool]] = []
    for ts, is_b in cands:
        if not kept:
            kept.append((ts, is_b))
            continue
        prev_t, prev_b = kept[-1]
        if (ts - prev_t) < dedupe_sec:
            if is_b and not prev_b:
                # Replace preceding non-boundary point only if strictly ahead of element before it
                if len(kept) < 2 or ts > kept[-2][0]:
                    kept[-1] = (ts, True)
                else:
                    kept[-1] = (prev_t, True)
            elif is_b and prev_b:
                # Never duplicate identical timestamp; keep only if >= dedupe_sec apart
                continue
            continue
        if ts > prev_t:
            kept.append((ts, is_b))
    return kept


def _extract_batch_multi_seek(
    video_input: Path,
    timestamps: List[float],
    out_dir: Path,
    ctx: StageContext,
    is_concat: bool = False,
) -> Dict[int, Path]:
    """Extract N <= 64 frames in one single FFmpeg pass via inline -filter_complex."""
    n = len(timestamps)
    if n == 0:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("f_*.jpg"):
        f.unlink(missing_ok=True)

    scale_expr = "scale=w=640:h=640:force_original_aspect_ratio=decrease:flags=fast_bilinear,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    if n == 1:
        fc = f"[0:v]trim=end_frame=1,setpts=PTS-STARTPTS,{scale_expr}[out]"
    else:
        fc = (
            "".join(f"[{i}:v]trim=end_frame=1,setpts=PTS-STARTPTS[v{i}];" for i in range(n))
            + "".join(f"[v{i}]" for i in range(n))
            + f"concat=n={n}:v=1:a=0,setpts=N/TB,{scale_expr}[out]"
        )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error"]
    for t in timestamps:
        if is_concat:
            cmd.extend(["-f", "concat", "-safe", "0", "-ss", f"{t:.3f}", "-i", str(video_input)])
        else:
            cmd.extend(["-ss", f"{t:.3f}", "-i", str(video_input)])
    cmd.extend([
        "-filter_complex", fc,
        "-map", "[out]", "-fps_mode", "passthrough", "-q:v", "4",
        str(out_dir / "f_%04d.jpg"),
    ])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
    )
    ctx.register_process(proc)
    stderr = ""
    try:
        _, stderr = proc.communicate(timeout=max(60, n * 5))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        logger.warning("Timed out extracting %d-frame batch", n)
        return {}
    finally:
        ctx.unregister_process(proc)

    if ctx.is_cancelled():
        raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

    files = sorted(out_dir.glob("f_*.jpg"))
    if proc.returncode == 0 and len(files) == n:
        return {i: files[i] for i in range(n)}
    if proc.returncode != 0:
        logger.warning("FFmpeg multi-seek batch (n=%d) failed (rc=%d): %s", n, proc.returncode, (stderr or "")[-250:])
    return {}


def _extract_batch_resilient(
    video_input: Path,
    timestamps: List[float],
    out_dir: Path,
    ctx: StageContext,
    is_concat: bool = False,
) -> Dict[int, Path]:
    """Extract batch in a single FFmpeg pass, recursively splitting only if corrupt frames fail a pass."""
    mapped = _extract_batch_multi_seek(video_input, timestamps, out_dir, ctx, is_concat=is_concat)
    if len(mapped) == len(timestamps) or not timestamps:
        return mapped
    if len(timestamps) == 1:
        return {}

    mid = len(timestamps) // 2
    left_dir = out_dir / "sub_l"
    right_dir = out_dir / "sub_r"
    left = _extract_batch_resilient(video_input, timestamps[:mid], left_dir, ctx, is_concat=is_concat)
    right = _extract_batch_resilient(video_input, timestamps[mid:], right_dir, ctx, is_concat=is_concat)
    combined: Dict[int, Path] = {}
    for k, p in left.items():
        combined[k] = p
    for k, p in right.items():
        combined[mid + k] = p
    return combined


class ExtractSnapshotsStage(Stage):
    """Streamed 64-frame-batch snapshot extraction, dHash dedupe, and packed WebP/JPEG storage."""

    name = "extract_snapshots"
    version = 1
    inputs = ("source_video", "scenes_json")
    outputs = ("snapshots_json", "artifact_dirs", "snapshot_count", "video_fingerprint", "pending", "pending_from")
    cacheable = True
    progress_unit = "batch_64"

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        opts = ctx.job.get("options", {})
        return {
            "interval_sec": float(ctx.params.get("snapshot_interval_sec", opts.get("snapshot_interval_sec", config.SNAPSHOT_INTERVAL_SEC))),
            "clip_interval_sec": float(ctx.params.get("snapshot_clip_interval_sec", config.SNAPSHOT_CLIP_INTERVAL_SEC)),
            "max_snapshots": int(ctx.params.get("snapshot_max_count", opts.get("snapshot_max_count", config.SNAPSHOT_MAX_COUNT))),
            "dhash_threshold": int(ctx.params.get("snapshot_dhash_threshold", config.SNAPSHOT_DHASH_THRESHOLD)),
            "disk_budget_mb": float(ctx.params.get("snapshot_disk_budget_mb", config.SNAPSHOT_DISK_BUDGET_MB)),
            "batch_size": min(64, max(1, int(ctx.params.get("snapshot_batch_size", config.SNAPSHOT_BATCH_SIZE)))),
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.is_cancelled():
            raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

        params = self.get_params(ctx)
        batch_size = int(params["batch_size"])
        dhash_thresh = int(params["dhash_threshold"])
        max_snapshots = max(1, int(params["max_snapshots"]))
        budget_bytes = max(1024, int(float(params["disk_budget_mb"]) * 1024 * 1024))

        d = Path(ctx.job["dir"])
        stage_work = d / "visual_work" / "snapshots_stage"
        thumbs_dir = stage_work / "thumbs"
        analysis_dir = stage_work / "analysis"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        scenes_data: Dict[str, Any] = {}
        scenes_json = ctx.inputs.get("scenes_json")
        if scenes_json and Path(scenes_json).is_file():
            scenes_data = json.loads(Path(scenes_json).read_text(encoding="utf-8"))
        scenes = scenes_data.get("scenes", [])
        has_video = bool(scenes_data.get("has_video", True))
        duration = float(scenes_data.get("duration") or ctx.inputs.get("duration") or ctx.job.get("duration") or 0.0)
        video_fp = str(scenes_data.get("video_fingerprint") or ctx.inputs.get("video_fingerprint") or "")
        pending = bool(scenes_data.get("pending", False))
        pending_from = scenes_data.get("pending_from")

        # Audio-only or empty edge case
        if not has_video or duration <= 0.0:
            snaps_path = stage_work / "snapshots.json"
            snaps_path.write_text(
                json.dumps({
                    "video_fingerprint": video_fp,
                    "duration": duration,
                    "has_video": False,
                    "pending": pending,
                    "pending_from": pending_from,
                    "snapshots": [],
                }, indent=2),
                encoding="utf-8",
            )
            return {
                "snapshots_json": str(snaps_path),
                "artifact_dirs": {"thumbs": str(thumbs_dir), "analysis": str(analysis_dir)},
                "snapshot_count": 0,
                "video_fingerprint": video_fp,
                "pending": pending,
                "pending_from": pending_from,
            }

        concat_str = ctx.inputs.get("source_concat")
        is_concat = bool(concat_str and Path(concat_str).is_file())
        media_input = Path(concat_str) if is_concat else Path(
            ctx.inputs.get("source_video") or ctx.job.get("source_path") or (ctx.job.get("source") or {}).get("path", "")
        )
        if not media_input.is_file():
            rel_src = f"users/{ctx.user_id}/jobs/{ctx.job.get('id', 'job')}/sources/{media_input.name}" if media_input.name else ""
            if rel_src and ctx.durable_store and ctx.durable_store.exists(rel_src):
                media_input = d / media_input.name
                ctx.durable_store.get_file(rel_src, media_input)
        if not media_input.is_file():
            raise StageError("SOURCE_MISSING", f"Video source missing for snapshot extraction: {media_input}")
        if not video_fp and not is_concat:
            video_fp = compute_file_sha256(media_input)

        words = ctx.inputs.get("transcript_words")
        if not isinstance(words, list) and (d / "words.json").is_file():
            try:
                words = json.loads((d / "words.json").read_text(encoding="utf-8"))
            except Exception:
                words = None
        candidate_clips = ctx.inputs.get("candidate_clips") or ctx.job.get("clips") or []

        planned = plan_snapshot_timestamps(
            scenes=scenes,
            duration=duration,
            words=words if isinstance(words, list) else None,
            candidate_clips=candidate_clips,
            interval_sec=float(params["interval_sec"]),
            clip_interval_sec=float(params["clip_interval_sec"]),
            dedupe_sec=0.5,
        )

        # Assign scene_id per target timestamp via bisect
        from studio.pipeline.visual_index import scene_at
        scene_starts = [float(s["start"]) for s in scenes]

        all_kept: List[Dict[str, Any]] = []
        prev_dhash: Optional[str] = None
        num_batches = max(1, (len(planned) + batch_size - 1) // batch_size) if planned else 0
        job_id = str(ctx.job.get("id", "job"))

        for b in range(num_batches):
            if ctx.is_cancelled():
                raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

            batch_items = planned[b * batch_size: (b + 1) * batch_size]
            chk_key = f"batch_{b:04d}"
            saved = ctx.resume_state(chk_key)

            restored_ok = False
            if saved and isinstance(saved.get("kept"), list):
                batch_kept = saved["kept"]
                missing_local = [
                    sn for sn in batch_kept
                    if not (thumbs_dir / f"{sn['id']}.webp").is_file()
                    or not (analysis_dir / f"{sn['id']}.jpg").is_file()
                ]
                if missing_local and ctx.durable_store and saved.get("batch_pack") and ctx.durable_store.exists(saved["batch_pack"]):
                    try:
                        ctx.durable_store.get_dir_unpacked(saved["batch_pack"], stage_work)
                        missing_local = [
                            sn for sn in batch_kept
                            if not (thumbs_dir / f"{sn['id']}.webp").is_file()
                            or not (analysis_dir / f"{sn['id']}.jpg").is_file()
                        ]
                    except Exception as e:
                        logger.warning("Could not unpack checkpoint batch %d: %s", b, e)
                if not missing_local:
                    all_kept.extend(batch_kept)
                    prev_dhash = saved.get("last_kept_dhash", prev_dhash)
                    restored_ok = True

            if restored_ok:
                continue

            batch_kept = []
            with tempfile.TemporaryDirectory(prefix="ce_snaps_b_") as tmp_b:
                extracted = _extract_batch_resilient(
                    media_input,
                    [t for t, _ in batch_items],
                    Path(tmp_b),
                    ctx,
                    is_concat=is_concat,
                )
                for idx_in_b, (ts, is_b) in enumerate(batch_items):
                    frame_file = extracted.get(idx_in_b)
                    if not frame_file or not frame_file.is_file():
                        continue  # skip corrupt frame safely
                    try:
                        with Image.open(frame_file) as img:
                            img.load()
                            w_an, h_an = img.size
                            if w_an <= 0 or h_an <= 0:
                                continue
                            dhash = compute_dhash(img)
                            if not is_b and prev_dhash is not None and hamming_distance(dhash, prev_dhash) <= dhash_thresh:
                                continue

                            sid = make_snapshot_id(video_fp, ts)
                            thumb_file = thumbs_dir / f"{sid}.webp"
                            analysis_file = analysis_dir / f"{sid}.jpg"

                            tw = min(320, w_an)
                            th = max(2, int(round(h_an * tw / max(1, w_an))))
                            with img.convert("RGB").resize((tw, th), Image.Resampling.BILINEAR) as th_img:
                                th_img.save(thumb_file, format="WEBP", quality=78, method=4)

                        shutil.copy2(str(frame_file), str(analysis_file))
                        f_bytes = thumb_file.stat().st_size + analysis_file.stat().st_size
                        sc_match = scene_at(scenes, ts, _starts=scene_starts)
                        snap_obj = Snapshot(
                            id=sid,
                            timestamp=ts,
                            path=str(analysis_file),
                            thumb_path=f"thumbs/{sid}.webp",
                            analysis_path=f"analysis/{sid}.jpg",
                            width=w_an,
                            height=h_an,
                            dhash=dhash,
                            is_scene_boundary=is_b,
                            scene_id=sc_match["id"] if sc_match else None,
                            face_count=None,
                            faces=None,
                        ).validate(check_files=True)
                        rec = snap_obj.model_dump()
                        rec["_bytes"] = int(f_bytes)
                        batch_kept.append(rec)
                        prev_dhash = dhash
                    except Exception as e:
                        logger.warning("Skipping unreadable frame at %.2fs: %s", ts, e)

            # Pack and checkpoint batch before appending so crash & resume stay atomic
            rel_batch_pack = f"users/{ctx.user_id}/checkpoints/{job_id}/extract_snapshots_batch_{b:04d}.tar"
            if ctx.durable_store and batch_kept:
                with tempfile.TemporaryDirectory(prefix="ce_bpack_") as bpack_tmp:
                    bp_root = Path(bpack_tmp)
                    (bp_root / "thumbs").mkdir()
                    (bp_root / "analysis").mkdir()
                    for sn in batch_kept:
                        shutil.copy2(thumbs_dir / f"{sn['id']}.webp", bp_root / "thumbs" / f"{sn['id']}.webp")
                        shutil.copy2(analysis_dir / f"{sn['id']}.jpg", bp_root / "analysis" / f"{sn['id']}.jpg")
                    ctx.durable_store.put_dir_packed(bp_root, rel_batch_pack)

            ctx.checkpoint(chk_key, {
                "batch_idx": b,
                "kept": batch_kept,
                "last_kept_dhash": prev_dhash,
                "batch_pack": rel_batch_pack if batch_kept else None,
            })
            all_kept.extend(batch_kept)

        # Cap total snapshots & disk budget while preserving scene boundaries
        def _total_bytes(items: List[Dict[str, Any]]) -> int:
            return sum(int(x.get("_bytes", 0)) for x in items)

        if len(all_kept) > max_snapshots or _total_bytes(all_kept) > budget_bytes:
            boundaries = [x for x in all_kept if x.get("is_scene_boundary")]
            non_bounds = [x for x in all_kept if not x.get("is_scene_boundary")]

            # Drop excess non-boundary snapshots evenly first
            if len(boundaries) <= max_snapshots:
                cap_nb = max(0, max_snapshots - len(boundaries))
                if len(non_bounds) > cap_nb:
                    if cap_nb == 0:
                        non_bounds = []
                    else:
                        pick = set(int(i) for i in np.linspace(0, len(non_bounds) - 1, cap_nb))
                        non_bounds = [non_bounds[i] for i in range(len(non_bounds)) if i in pick]
                kept_set = boundaries + non_bounds
            else:
                pick = set(int(i) for i in np.linspace(0, len(boundaries) - 1, max_snapshots))
                kept_set = [boundaries[i] for i in range(len(boundaries)) if i in pick]

            kept_set.sort(key=lambda x: float(x["timestamp"]))
            # If disk budget still exceeded, progressively thin non-boundaries then oldest boundaries
            while _total_bytes(kept_set) > budget_bytes and len(kept_set) > len(boundaries):
                nb_indices = [i for i, x in enumerate(kept_set) if not x.get("is_scene_boundary")]
                if not nb_indices:
                    break
                drop_step = max(2, len(nb_indices) // max(1, (len(nb_indices) + 1) // 2))
                to_drop = {nb_indices[i] for i in range(0, len(nb_indices), drop_step)}
                kept_set = [x for i, x in enumerate(kept_set) if i not in to_drop]
            while _total_bytes(kept_set) > budget_bytes and len(kept_set) > 2:
                kept_set = [kept_set[i] for i in range(0, len(kept_set), 2)] + ([kept_set[-1]] if len(kept_set) % 2 == 0 else [])

            kept_ids = {x["id"] for x in kept_set}
            for x in all_kept:
                if x["id"] not in kept_ids:
                    (thumbs_dir / f"{x['id']}.webp").unlink(missing_ok=True)
                    (analysis_dir / f"{x['id']}.jpg").unlink(missing_ok=True)
            all_kept = kept_set

        # Clean internal byte counters and store portable paths in manifest
        clean_snaps: List[Dict[str, Any]] = []
        for x in all_kept:
            clean = {k: v for k, v in x.items() if k != "_bytes"}
            clean_snaps.append(clean)

        snaps_path = stage_work / "snapshots.json"
        snaps_path.write_text(
            json.dumps({
                "video_fingerprint": video_fp,
                "duration": round(duration, 3),
                "has_video": True,
                "pending": pending,
                "pending_from": pending_from,
                "snapshot_count": len(clean_snaps),
                "snapshots": clean_snaps,
            }, indent=2),
            encoding="utf-8",
        )

        return {
            "snapshots_json": str(snaps_path),
            "artifact_dirs": {"thumbs": str(thumbs_dir), "analysis": str(analysis_dir)},
            "snapshot_count": len(clean_snaps),
            "video_fingerprint": video_fp,
            "pending": pending,
            "pending_from": pending_from,
        }
