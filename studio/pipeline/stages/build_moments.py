"""BuildMomentsStage (v1): join transcript segments, snapshots, and scenes into Moment skeletons."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from studio.contracts import Moment, TranscriptSegment, Word, make_moment_id, make_segment_id
from studio.pipeline.stage import Stage, StageContext, StageError
from studio.pipeline.visual_index import VisualTimelineIndex

logger = logging.getLogger("studio.pipeline.build_moments")

_SENTENCE_END_RE = re.compile(r"[.?!]['\"\)]?$")


def build_transcript_segments(
    words: List[Dict[str, Any]],
    video_fp: str,
    min_sec: float = 3.0,
    max_sec: float = 10.0,
) -> List[Dict[str, Any]]:
    """Group word timestamps into half-open non-overlapping [start, end) segments of ~3-10s."""
    if not words:
        return []
    clean_words: List[Dict[str, Any]] = []
    for w in words:
        try:
            ws = max(0.0, float(w.get("s", w.get("start", 0.0))))
            we = float(w.get("e", w.get("end", ws + 0.05)))
            if we <= ws:
                we = round(ws + 0.05, 3)
            wt = str(w.get("w", w.get("word", ""))).strip()
            conf = float(w.get("p", w.get("confidence", 1.0)))
            conf = min(1.0, max(0.0, conf))
            if wt:
                clean_words.append({"word": wt, "start": round(ws, 3), "end": round(we, 3), "confidence": round(conf, 3)})
        except (TypeError, ValueError):
            continue
    clean_words.sort(key=lambda x: (x["start"], x["end"]))
    if not clean_words:
        return []

    raw_groups: List[List[Dict[str, Any]]] = []
    cur_group: List[Dict[str, Any]] = []
    for w in clean_words:
        if not cur_group:
            cur_group.append(w)
            continue
        g_start = cur_group[0]["start"]
        prev_end = cur_group[-1]["end"]
        cur_span = prev_end - g_start
        gap = w["start"] - prev_end
        should_split = (
            gap >= 1.5
            or (w["end"] - g_start) > max_sec
            or (cur_span >= min_sec and _SENTENCE_END_RE.search(cur_group[-1]["word"]) is not None)
        )
        if should_split:
            raw_groups.append(cur_group)
            cur_group = [w]
        else:
            cur_group.append(w)
    if cur_group:
        raw_groups.append(cur_group)

    segments: List[Dict[str, Any]] = []
    prev_seg_end = 0.0
    for grp in raw_groups:
        s0 = max(prev_seg_end, grp[0]["start"])
        e0 = max(s0 + 0.05, grp[-1]["end"])
        word_models: List[Dict[str, Any]] = []
        for w in grp:
            ws = max(s0, w["start"])
            we = max(ws + 0.02, w["end"])
            word_models.append(Word(word=w["word"], start=round(ws, 3), end=round(we, 3), confidence=w["confidence"]).model_dump())
            e0 = max(e0, we)
        s0 = round(s0, 3)
        e0 = round(e0, 3)
        seg = TranscriptSegment(
            id=make_segment_id(video_fp, s0, e0),
            start=s0,
            end=e0,
            text=" ".join(x["word"] for x in grp),
            words=[Word(**wm) for wm in word_models],
        ).validate(check_files=False)
        segments.append(seg.model_dump())
        prev_seg_end = e0
    return segments


class BuildMomentsStage(Stage):
    """Cheap recomputed join between transcript segments, analyzed snapshots, and scenes."""

    name = "build_moments"
    version = 1
    inputs = ("scenes_json", "snapshots_json", "analyzed_scenes_json", "analyzed_snapshots_json", "captions_json")
    outputs = ("moments_json", "moment_count")
    cacheable = True
    progress_unit = "stage"

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        words = ctx.inputs.get("transcript_words")
        if not isinstance(words, list):
            wf = Path(ctx.job.get("dir", "")) / "words.json"
            if wf.is_file():
                try:
                    words = json.loads(wf.read_text(encoding="utf-8"))
                except Exception:
                    words = []
        sig_src = json.dumps(
            [(round(float(w.get("s", w.get("start", 0.0))), 2), w.get("w", w.get("word", ""))) for w in (words or [])[:2000]],
            separators=(",", ":"),
        )
        return {"words_count": len(words or []), "words_sig": hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:12]}

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.is_cancelled():
            raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

        d = Path(ctx.job["dir"])
        stage_work = d / "visual_work" / "moments_stage"
        stage_work.mkdir(parents=True, exist_ok=True)

        sc_file = ctx.inputs.get("analyzed_scenes_json") or ctx.inputs.get("scenes_json")
        sn_file = ctx.inputs.get("analyzed_snapshots_json") or ctx.inputs.get("snapshots_json")
        cap_file = ctx.inputs.get("captions_json")

        sc_payload = json.loads(Path(sc_file).read_text(encoding="utf-8")) if sc_file and Path(sc_file).is_file() else {}
        sn_payload = json.loads(Path(sn_file).read_text(encoding="utf-8")) if sn_file and Path(sn_file).is_file() else {}
        cap_payload = json.loads(Path(cap_file).read_text(encoding="utf-8")) if cap_file and Path(cap_file).is_file() else {}

        video_fp = str(sc_payload.get("video_fingerprint") or sn_payload.get("video_fingerprint") or "")
        duration = float(sc_payload.get("duration") or sn_payload.get("duration") or ctx.job.get("duration") or 0.0)
        pending = bool(sc_payload.get("pending") or sn_payload.get("pending"))
        pending_from = sc_payload.get("pending_from") or sn_payload.get("pending_from")

        scenes = list(sc_payload.get("scenes", []))
        snapshots = list(sn_payload.get("snapshots", []))
        captions_map = cap_payload.get("captions", {}) if isinstance(cap_payload.get("captions"), dict) else {}

        # Merge AI captions onto snapshot copies cleanly
        merged_snaps: List[Dict[str, Any]] = []
        for sn in snapshots:
            s_copy = dict(sn)
            if s_copy.get("id") in captions_map:
                s_copy["caption"] = captions_map[s_copy["id"]]
            merged_snaps.append(s_copy)

        words = ctx.inputs.get("transcript_words")
        if not isinstance(words, list) and (d / "words.json").is_file():
            try:
                words = json.loads((d / "words.json").read_text(encoding="utf-8"))
            except Exception:
                words = []

        segments = build_transcript_segments(words if isinstance(words, list) else [], video_fp)
        tl = VisualTimelineIndex(scenes=scenes, snapshots=merged_snaps, segments=segments)

        # Build half-open non-overlapping [start, end) spans covering segments, or scenes if silent
        spans: List[Dict[str, Any]] = []
        if segments:
            for i, seg in enumerate(segments):
                st = float(seg["start"])
                if i + 1 < len(segments):
                    en = max(st + 0.05, float(segments[i + 1]["start"]))
                else:
                    en = max(float(seg["end"]), duration) if duration > float(seg["end"]) else float(seg["end"])
                spans.append({
                    "start": round(st, 3),
                    "end": round(en, 3),
                    "seg_ids": [seg["id"]],
                    "text": seg["text"],
                })
        else:
            for sc in tl.scenes:
                s0, e0 = float(sc["start"]), float(sc["end"])
                n_parts = max(1, int(math.ceil((e0 - s0) / 10.0)))
                step = (e0 - s0) / n_parts
                for p in range(n_parts):
                    st = round(s0 + p * step, 3)
                    en = round(e0 if p == n_parts - 1 else s0 + (p + 1) * step, 3)
                    if en > st:
                        spans.append({"start": st, "end": en, "seg_ids": [], "text": ""})

        # Ensure initial leading gap [0, first_seg) if snapshots exist before first speech word
        if spans and spans[0]["start"] > 0.0 and tl.snapshots_in(0.0, spans[0]["start"]):
            spans.insert(0, {"start": 0.0, "end": spans[0]["start"], "seg_ids": [], "text": ""})

        moments: List[Dict[str, Any]] = []
        seen_snaps = set()
        for i, sp in enumerate(spans):
            st, en = sp["start"], sp["end"]
            sn_list = tl.snapshots_in(st, en)
            snap_ids = []
            for s in sn_list:
                sid = s["id"]
                if sid not in seen_snaps:
                    seen_snaps.add(sid)
                    snap_ids.append(sid)
            ov_scenes = tl.scenes_overlapping(st, en)
            scene_ids = [sc["id"] for sc in ov_scenes]
            mom = Moment(
                id=make_moment_id(video_fp, st, en),
                start=st,
                end=en,
                score=0.0,
                reason="",
                transcript_segment_ids=list(sp["seg_ids"]),
                snapshot_ids=snap_ids,
                scene_ids=scene_ids,
                text=sp["text"],
                pending=bool(pending and i == len(spans) - 1),
            ).validate(check_files=False)
            moments.append(mom.model_dump())

        moments_path = stage_work / "moments.json"
        moments_payload = {
            "video_fingerprint": video_fp,
            "duration": duration,
            "pending": pending,
            "pending_from": pending_from,
            "moments": moments,
            "segments": segments,
        }
        moments_path.write_text(json.dumps(moments_payload, indent=2), encoding="utf-8")

        return {
            "moments_json": str(moments_path),
            "moment_count": len(moments),
        }
