"""Select clips stage: selects viral highlight moments via AI analysis or loudness peaks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from studio import brain
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
        duration = float(ctx.job.get("duration") or 0.0)
        opts = ctx.job.get("options", {})

        title = ctx.job.get("title") or "Video"
        uploader = ctx.job.get("uploader") or ctx.job.get("creator") or ""
        max_clips = int(opts.get("max_clips", 6) or 6)
        min_len = int(opts.get("min_len", 15) or 15)
        max_len = int(opts.get("max_len", 60) or 60)

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
            )
        except Exception as e:
            # Fallback to pure loudness if AI fails
            logger.warning("AI clip selection failed, falling back to loudness: %s", e)
            found = brain.fallback_by_loudness(db_arr, words, max_clips, max_len, duration)
            ai_resp = {}

        if not found and duration > 5.0:
            z = brain.loud_zscore(db_arr)
            order = np.argsort(-z)
            for sec in order:
                s = max(0.0, float(sec) - max_len * 0.5)
                e = min(duration, s + min(max_len, 20.0))
                if any(not (e <= p["start"] or s >= p["end"]) for p in found):
                    continue
                found.append({
                    "start": s,
                    "end": e,
                    "score": 50.0,
                    "hook": "HIGHLIGHT",
                    "title": "Highlight Moment",
                    "reason": "Loudest audio peak",
                    "shots": brain.plan_multishots(s, e),
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
