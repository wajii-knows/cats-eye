"""Select clips stage: selects viral highlight moments via AI analysis or loudness peaks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from studio import brain, flags
from studio.contracts import Moment
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.select_clips")


class SelectClipsStage(Stage):
    name = "select_clips"
    version = 1
    inputs = ("transcript_words", "loudness_db")
    outputs = ("moments", "clips_spec")
    cacheable = True

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        opts = ctx.job.get("options", {})
        return {
            "title": ctx.job.get("title", ""),
            "creator": ctx.job.get("creator", ""),
            "clip_style": opts.get("style", "bounce"),
            "layout": opts.get("layout", "crop"),
            "range_start": opts.get("range_start", 0),
            "range_end": opts.get("range_end"),
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        words = ctx.inputs.get("transcript_words", [])
        loudness = ctx.inputs.get("loudness_db", [])
        visual_snaps = ctx.inputs.get("analyzed_snapshots")
        if not isinstance(visual_snaps, list):
            visual_snaps = None
        duration = float(ctx.job.get("duration") or 0.0)
        opts = ctx.job.get("options", {})

        title = ctx.job.get("title") or "Video"
        uploader = ctx.job.get("uploader") or ctx.job.get("creator") or ""
        max_clips = int(opts.get("max_clips", 6) or 6)
        min_len = int(opts.get("min_len", 15) or 15)
        max_len = int(opts.get("max_len", 35) or 35)

        import numpy as np
        db_arr = np.array(loudness, dtype=np.float32) if isinstance(loudness, list) else loudness

        try:
            found, ai_resp = brain.find_clips(
                words=words,
                db=db_arr,
                duration=duration,
                title=title,
                uploader=uploader,
                max_clips=max_clips,
                min_len=min_len,
                max_len=max_len,
                visual_metrics=visual_snaps,
            )
        except Exception as e:
            # Fallback to pure loudness if AI fails
            logger.warning("AI clip selection failed, falling back to loudness: %s", e)
            found = brain.fallback_by_loudness(
                db_arr, words, max_clips, max_len, duration, min_len=min_len, visual_metrics=visual_snaps,
            )
            ai_resp = {}

        if not found and duration > 5.0:
            z = brain.loud_zscore(db_arr)
            order = np.argsort(-z)
            target = min(float(max_len), max(float(min_len), 22.0))
            lead = min(3.0, target * 0.2)
            for sec in order:
                s = round(max(0.0, min(float(sec) - lead, max(0.0, duration - target))), 2)
                e = round(min(duration, s + target), 2)
                if any(not (e <= p["start"] or s >= p["end"]) for p in found):
                    continue
                sc = brain.score_clip(
                    s, e, words, z, min_len=min_len, max_len=max_len, visual_snaps=visual_snaps,
                )
                seg_z = z[int(s):int(e) + 1] if len(z) > int(s) else None
                found.append({
                    "start": s,
                    "end": e,
                    "score": sc["score"],
                    "scores": sc["scores"],
                    "tier": sc["tier"],
                    "scroll_stopper": sc["scroll_stopper"],
                    "hook": "HIGH ENERGY MOMENT",
                    "title": "Peak energy moment",
                    "reason": f"Picked by peak audio energy (Hook {sc['scores']['hook']}, Payoff {sc['scores']['payoff']}).",
                    "shots": brain.plan_multishots(s, e, words=words, db_slice=seg_z) if flags.is_on("PEAKCUT_MULTISHOT") else [],
                })
                if len(found) >= min(max_clips, 2):
                    break

        moments = []
        for i, c in enumerate(found):
            m = Moment(
                id=f"moment_{i+1}",
                start=float(c.get("start", 0.0)),
                end=float(c.get("end", 0.0)),
                score=float(c.get("score", 70.0)),
                reason=c.get("reason", "Highlighted audio/visual energy"),
                hook_text=c.get("hook", ""),
                suggested_title=c.get("title", ""),
            )
            moments.append(m)

        ctx.job["ai"] = ai_resp
        return {
            "moments": [m.model_dump() for m in moments],
            "clips_spec": found,
        }
