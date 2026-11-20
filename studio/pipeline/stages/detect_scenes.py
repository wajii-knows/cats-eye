"""DetectScenesStage (v1): chunked scene-cut detection + silence-gap & window soft scenes."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from studio import config
from studio.contracts import Scene, make_scene_id
from studio.pipeline.stage import Stage, StageContext, StageError, compute_file_sha256

logger = logging.getLogger("studio.pipeline.detect_scenes")

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9]+(?:\.[0-9]+)?)")


def _parse_rate(val: str) -> float:
    s = str(val or "").strip()
    if not s or s == "0/0":
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            den = float(b)
            return float(a) / den if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def probe_visual_media(path: Path) -> Dict[str, Any]:
    """Probe media file via ffprobe, supporting video-only, audio-only, VFR, and rotated video."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {p}")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(p),
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=NO_WINDOW, timeout=60,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr[-400:]}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    v = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    dur = float(fmt.get("duration") or 0.0)
    if dur <= 0 and v:
        dur = float(v.get("duration") or 0.0)

    if not v:
        return {
            "has_video": False,
            "has_audio": has_audio,
            "width": 0,
            "height": 0,
            "rotation": 0,
            "duration": max(0.0, dur),
            "is_vfr": False,
            "fps": 0.0,
        }

    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    rot_raw = None
    for sd in (v.get("side_data_list") or []):
        if "rotation" in sd:
            rot_raw = sd["rotation"]
            break
    if rot_raw is None:
        rot_raw = (v.get("tags") or {}).get("rotate", "0")
    try:
        rot = int(round(float(rot_raw))) % 360
    except (TypeError, ValueError):
        rot = 0
    if rot in (90, 270) and w > 0 and h > 0:
        w, h = h, w

    r_fps = _parse_rate(v.get("r_frame_rate", ""))
    avg_fps = _parse_rate(v.get("avg_frame_rate", ""))
    is_vfr = bool(r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.05)
    return {
        "has_video": w >= 2 and h >= 2,
        "has_audio": has_audio,
        "width": w,
        "height": h,
        "rotation": rot,
        "duration": max(0.0, dur),
        "is_vfr": is_vfr,
        "fps": round(avg_fps or r_fps, 3),
    }


def _collect_live_completed_segments(job: Dict[str, Any]) -> Tuple[List[Path], float, bool]:
    """For active live recordings, collect only finalized .ts buffer segments and return (segs, dur, is_pending)."""
    job_dir = Path(job.get("dir", ""))
    buf_dir = job_dir / "buffer"
    if not buf_dir.is_dir():
        return [], 0.0, False
    segs = sorted(buf_dir.glob("*.ts"))
    if not segs:
        return [], 0.0, False
    is_live_active = job.get("status") == "live"
    try:
        from studio import live
        sess = live.SESSIONS.get(job.get("id"))
        if sess and not sess.finished:
            is_live_active = True
    except Exception:
        pass
    now = time.time()
    completed = []
    for i, seg in enumerate(segs):
        try:
            age = now - seg.stat().st_mtime
            if is_live_active and (i == len(segs) - 1 or age < 2.0):
                continue
            if seg.stat().st_size > 188:
                completed.append(seg)
        except OSError:
            pass
    if not completed:
        return [], 0.0, is_live_active
    total_dur = 0.0
    for seg in completed:
        try:
            info = probe_visual_media(seg)
            total_dur += info.get("duration") or 0.0
        except Exception:
            total_dur += 6.0
    return completed, round(total_dur, 3), is_live_active


def _load_transcript_words(ctx: StageContext) -> List[Dict[str, Any]]:
    words = ctx.inputs.get("transcript_words")
    if isinstance(words, list):
        return words
    for cand in (ctx.inputs.get("words_json"), str(Path(ctx.job["dir"]) / "words.json")):
        if cand and Path(cand).is_file():
            try:
                data = json.loads(Path(cand).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []


def _extract_silence_gaps(words: List[Dict[str, Any]], duration: float, min_gap: float = 2.0) -> List[Tuple[float, float]]:
    """Find silence intervals >= min_gap seconds from word timestamps."""
    if not words:
        return []
    clean = []
    for w in words:
        try:
            s = float(w.get("s", w.get("start", 0.0)))
            e = float(w.get("e", w.get("end", s)))
            if e > s >= 0.0:
                clean.append((s, min(duration, e) if duration > 0 else e))
        except (TypeError, ValueError):
            pass
    if not clean:
        return []
    clean.sort(key=lambda x: (x[0], x[1]))
    gaps: List[Tuple[float, float]] = []
    prev_e = 0.0
    for s, e in clean:
        if s - prev_e >= min_gap:
            gaps.append((round(prev_e, 3), round(s, 3)))
        prev_e = max(prev_e, e)
    if duration > 0 and duration - prev_e >= min_gap:
        gaps.append((round(prev_e, 3), round(duration, 3)))
    return gaps


def _run_ffmpeg_scene_chunk(
    video_input: Path,
    chunk_start: float,
    chunk_len: float,
    threshold: float,
    ctx: StageContext,
    is_concat: bool = False,
) -> List[Dict[str, Any]]:
    """Run FFmpeg scene-score filter on a downscaled 320px/4fps proxy chunk."""
    vf = (
        "fps=4,"
        "scale='if(gt(iw\\,ih)\\,320\\,-2)':'if(gt(iw\\,ih)\\,-2\\,320)':flags=fast_bilinear,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
        f"select='gt(scene\\,{threshold:.3f})',"
        "metadata=print:direct=1"
    )
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info"]
    if is_concat:
        cmd.extend(["-f", "concat", "-safe", "0", "-ss", f"{chunk_start:.3f}", "-t", f"{chunk_len:.3f}", "-i", str(video_input)])
    else:
        cmd.extend(["-ss", f"{chunk_start:.3f}", "-t", f"{chunk_len:.3f}", "-i", str(video_input)])
    cmd.extend(["-an", "-sn", "-dn", "-vf", vf, "-f", "null", "-"])

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
    try:
        _, stderr = proc.communicate(timeout=max(60, int(chunk_len * 2)))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise StageError("SCENE_TIMEOUT", f"Scene detection timed out on chunk at {chunk_start:.1f}s", retryable=True)
    finally:
        ctx.unregister_process(proc)

    if ctx.is_cancelled():
        raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})
    if proc.returncode != 0:
        raise StageError("SCENE_FFMPEG_FAILED", f"ffmpeg scene pass failed: {stderr[-300:]}", retryable=False)

    cuts: List[Dict[str, Any]] = []
    cur_pts: Optional[float] = None
    for line in (stderr or "").splitlines():
        mp = _PTS_RE.search(line)
        if mp:
            cur_pts = float(mp.group(1))
            continue
        ms = _SCORE_RE.search(line)
        if ms and cur_pts is not None:
            score = min(1.0, max(0.0, float(ms.group(1))))
            if chunk_start > 0 and cur_pts < 0.25:
                cur_pts = None
                continue
            abs_t = round(chunk_start + cur_pts, 3)
            cuts.append({"t": abs_t, "score": round(score, 4)})
            cur_pts = None
    return cuts


def merge_and_build_scenes(
    duration: float,
    raw_cuts: List[Dict[str, Any]],
    silence_gaps: List[Tuple[float, float]],
    video_fp: str,
    min_scene_sec: float = 0.6,
    max_scene_sec: float = 60.0,
    merge_gap_sec: float = 0.5,
    pending: bool = False,
) -> List[Dict[str, Any]]:
    """Dedupe boundary cuts, merge micro-scenes < min_scene_sec, split scenes > max_scene_sec."""
    dur = max(0.05, float(duration))
    # 1. Dedupe nearby cuts within merge_gap_sec, keeping higher confidence score
    sorted_cuts = sorted(
        (c for c in raw_cuts if 0.0 < float(c["t"]) < dur),
        key=lambda x: float(x["t"]),
    )
    merged_cuts: List[Dict[str, Any]] = []
    for c in sorted_cuts:
        t = float(c["t"])
        sc = min(1.0, max(0.0, float(c.get("score", 0.5))))
        if merged_cuts and (t - merged_cuts[-1]["t"]) <= merge_gap_sec:
            if sc > merged_cuts[-1]["score"]:
                merged_cuts[-1] = {"t": round(t, 3), "score": sc}
        else:
            merged_cuts.append({"t": round(t, 3), "score": sc})

    # 2. Construct initial hard-cut segments [0, dur]
    segments: List[Dict[str, Any]] = []
    prev_t = 0.0
    prev_conf = 1.0
    for c in merged_cuts:
        t = c["t"]
        if t <= prev_t:
            continue
        segments.append({"start": prev_t, "end": t, "method": "cut", "confidence": prev_conf})
        prev_t = t
        prev_conf = max(0.5, min(1.0, round(c["score"], 4)))
    if dur - prev_t > 1e-4 or not segments:
        segments.append({"start": prev_t, "end": round(dur, 3), "method": "cut", "confidence": prev_conf})

    # 3. Merge scenes shorter than min_scene_sec into neighbors
    while len(segments) > 1:
        short_idx = next((i for i, s in enumerate(segments) if (s["end"] - s["start"]) < min_scene_sec), None)
        if short_idx is None:
            break
        cur = segments[short_idx]
        if short_idx == 0:
            nxt = segments[1]
            segments[1] = {**nxt, "start": cur["start"]}
            segments.pop(0)
        elif short_idx == len(segments) - 1:
            prv = segments[-2]
            segments[-2] = {**prv, "end": cur["end"]}
            segments.pop()
        else:
            prv = segments[short_idx - 1]
            nxt = segments[short_idx + 1]
            # Merge into longer neighbor so true cut boundaries stay stable
            if (prv["end"] - prv["start"]) >= (nxt["end"] - nxt["start"]):
                segments[short_idx - 1] = {**prv, "end": cur["end"]}
            else:
                segments[short_idx + 1] = {**nxt, "start": cur["start"]}
            segments.pop(short_idx)

    # 4. Split scenes > max_scene_sec via silence gaps (>=2s) first, then uniform time windows
    final_segs: List[Dict[str, Any]] = []
    for seg in segments:
        s0, e0 = seg["start"], seg["end"]
        if (e0 - s0) <= max_scene_sec:
            final_segs.append(seg)
            continue

        cur = s0
        first_piece = True
        while (e0 - cur) > max_scene_sec + 1e-6:
            win_lo = cur + min_scene_sec
            win_hi = min(e0 - min_scene_sec, cur + max_scene_sec)
            cand_gaps = [
                (g0, g1, 0.5 * (g0 + g1), g1 - g0)
                for g0, g1 in silence_gaps
                if win_lo <= 0.5 * (g0 + g1) <= win_hi
            ]
            # Prefer a gap past 35% of max_scene_sec so we don't emit tiny slivers unnecessarily
            late_gaps = [g for g in cand_gaps if g[2] >= cur + max_scene_sec * 0.35]
            picked = max(late_gaps or cand_gaps, key=lambda g: g[3]) if cand_gaps else None
            if picked is not None:
                split_t = round(picked[2], 3)
                method = seg["method"] if first_piece else "soft"
                conf = seg["confidence"] if first_piece else 0.65
                final_segs.append({"start": round(cur, 3), "end": split_t, "method": method, "confidence": conf})
                cur = split_t
                first_piece = False
            else:
                rem = e0 - cur
                n_parts = max(2, int(math.ceil(rem / max_scene_sec)))
                step = rem / n_parts
                for p in range(n_parts):
                    ps = round(cur + p * step, 3)
                    pe = round(e0 if p == n_parts - 1 else cur + (p + 1) * step, 3)
                    if pe > ps:
                        final_segs.append({
                            "start": ps,
                            "end": pe,
                            "method": seg["method"] if (first_piece and p == 0) else "soft",
                            "confidence": seg["confidence"] if (first_piece and p == 0) else 0.45,
                        })
                cur = e0
                break
        if (e0 - cur) > 1e-4:
            final_segs.append({
                "start": round(cur, 3),
                "end": round(e0, 3),
                "method": seg["method"] if first_piece else "soft",
                "confidence": seg["confidence"] if first_piece else 0.6,
            })

    # Enforce strict start < end and deterministic IDs validated against contracts.Scene
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(final_segs):
        st = round(float(s["start"]), 3)
        en = round(float(s["end"]), 3)
        if en <= st:
            continue
        is_last = (i == len(final_segs) - 1)
        model = Scene(
            id=make_scene_id(video_fp, st),
            start=st,
            end=en,
            method="soft" if s.get("method") == "soft" else "cut",
            confidence=round(min(1.0, max(0.0, float(s.get("confidence", 0.5)))), 4),
            pending=bool(pending and is_last),
        ).validate(check_files=False)
        out.append(model.model_dump())
    return out


class DetectScenesStage(Stage):
    """Chunked 320px/4fps FFmpeg scene-cut detection with soft scene subdivision."""

    name = "detect_scenes"
    version = 1
    inputs = ("source_video",)
    outputs = ("scenes_json", "video_fingerprint", "duration", "has_video", "pending", "pending_from")
    cacheable = True
    progress_unit = "10m_chunk"

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        opts = ctx.job.get("options", {})
        words = _load_transcript_words(ctx)
        gaps = _extract_silence_gaps(words, 0.0, min_gap=2.0)
        gaps_sig = hashlib.sha256(json.dumps(gaps).encode("utf-8")).hexdigest()[:12]
        return {
            "threshold": float(ctx.params.get("scene_threshold", opts.get("scene_threshold", config.SCENE_THRESHOLD))),
            "chunk_sec": float(ctx.params.get("scene_chunk_sec", config.SCENE_CHUNK_SEC)),
            "overlap_sec": float(ctx.params.get("scene_overlap_sec", config.SCENE_CHUNK_OVERLAP_SEC)),
            "min_scene_sec": float(ctx.params.get("scene_min_duration_sec", config.SCENE_MIN_DURATION_SEC)),
            "max_scene_sec": float(ctx.params.get("scene_max_duration_sec", config.SCENE_MAX_DURATION_SEC)),
            "gaps_sig": gaps_sig,
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.is_cancelled():
            raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

        params = self.get_params(ctx)
        threshold = float(params["threshold"])
        chunk_sec = max(10.0, float(params["chunk_sec"]))
        overlap_sec = max(0.0, float(params["overlap_sec"]))
        min_scene_sec = max(0.1, float(params["min_scene_sec"]))
        max_scene_sec = max(5.0, float(params["max_scene_sec"]))

        d = Path(ctx.job["dir"])
        d.mkdir(parents=True, exist_ok=True)
        stage_work = d / "visual_work"
        stage_work.mkdir(parents=True, exist_ok=True)

        # Handle growing live recording edge case
        live_segs, live_dur, is_live_pending = _collect_live_completed_segments(ctx.job)
        concat_file: Optional[Path] = None
        if live_segs:
            concat_file = stage_work / "live_completed.txt"
            concat_file.write_text(
                "\n".join(f"file '{str(s.resolve()).replace(chr(39), chr(39)*2)}'" for s in live_segs),
                encoding="utf-8",
            )
            media_path = live_segs[0]
            info = probe_visual_media(media_path)
            duration = live_dur
            video_fp = compute_file_sha256(live_segs[-1])
        else:
            src_str = ctx.inputs.get("source_video") or ctx.job.get("source_path") or (ctx.job.get("source") or {}).get("path", "")
            media_path = Path(src_str)
            if not src_str or not media_path.is_file():
                # Fallback restore from durable store if uploaded source missing locally
                rel_src = f"users/{ctx.user_id}/jobs/{ctx.job.get('id', 'job')}/sources/{media_path.name}" if media_path.name else ""
                if rel_src and ctx.durable_store and ctx.durable_store.exists(rel_src):
                    media_path = d / media_path.name
                    ctx.durable_store.get_file(rel_src, media_path)
            if not media_path.is_file():
                raise StageError("SOURCE_MISSING", f"Source media file not found: {src_str}")
            info = probe_visual_media(media_path)
            duration = float(info.get("duration") or ctx.job.get("duration") or 0.0)
            video_fp = ctx.inputs.get("video_fingerprint") or compute_file_sha256(media_path)

        has_video = bool(info.get("has_video"))
        if duration <= 0.0:
            raise StageError("EMPTY_MEDIA", "Media file has zero duration")

        words = _load_transcript_words(ctx)
        silence_gaps = _extract_silence_gaps(words, duration, min_gap=2.0)

        raw_cuts: List[Dict[str, Any]] = []
        if has_video:
            n_chunks = max(1, int(math.ceil(duration / chunk_sec)))
            for k in range(n_chunks):
                if ctx.is_cancelled():
                    raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})
                base_s = k * chunk_sec
                base_e = min(duration, (k + 1) * chunk_sec)
                c_start = max(0.0, base_s - (overlap_sec if k > 0 else 0.0))
                c_end = min(duration, base_e + (overlap_sec if k < n_chunks - 1 else 0.0))
                c_len = max(0.1, c_end - c_start)

                chk_key = f"chunk_{k:04d}"
                saved = ctx.resume_state(chk_key)
                if saved and "cuts" in saved:
                    chunk_cuts = saved["cuts"]
                else:
                    chunk_cuts = _run_ffmpeg_scene_chunk(
                        video_input=(concat_file or media_path),
                        chunk_start=c_start,
                        chunk_len=c_len,
                        threshold=threshold,
                        ctx=ctx,
                        is_concat=bool(concat_file),
                    )
                    ctx.checkpoint(chk_key, {"chunk_idx": k, "start": c_start, "end": c_end, "cuts": chunk_cuts})
                raw_cuts.extend(chunk_cuts)

        scenes = merge_and_build_scenes(
            duration=duration,
            raw_cuts=raw_cuts if has_video else [],
            silence_gaps=silence_gaps,
            video_fp=video_fp,
            min_scene_sec=min_scene_sec,
            max_scene_sec=max_scene_sec,
            merge_gap_sec=0.5,
            pending=is_live_pending,
        )

        scenes_path = stage_work / "scenes.json"
        scenes_payload = {
            "video_fingerprint": video_fp,
            "duration": round(duration, 3),
            "has_video": has_video,
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "rotation": int(info.get("rotation") or 0),
            "is_vfr": bool(info.get("is_vfr")),
            "pending": is_live_pending,
            "pending_from": round(duration, 3) if is_live_pending else None,
            "threshold": threshold,
            "scenes": scenes,
        }
        scenes_path.write_text(json.dumps(scenes_payload, indent=2), encoding="utf-8")

        return {
            "scenes_json": str(scenes_path),
            "video_fingerprint": video_fp,
            "duration": round(duration, 3),
            "has_video": has_video,
            "pending": is_live_pending,
            "pending_from": round(duration, 3) if is_live_pending else None,
            "scene_count": len(scenes),
            "media_info": info,
            "source_concat": str(concat_file) if concat_file else None,
        }
