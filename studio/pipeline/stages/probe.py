"""Probe stage: extracts technical video streams metadata via ffprobe."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from studio import media
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.probe")


class ProbeStage(Stage):
    name = "probe"
    version = 1
    inputs = ("source_video",)
    outputs = ("probe_info",)
    cacheable = True

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        source = ctx.inputs.get("source_video") or ctx.job.get("source_path")
        if not source or not Path(source).is_file():
            raise StageError("SOURCE_MISSING", f"Probe stage missing source video: {source}")

        try:
            info = media.probe(Path(source))
        except Exception as e:
            raise StageError("PROBE_FAILED", f"ffprobe failed: {e}") from e

        ctx.job.update(
            width=info.get("width"),
            height=info.get("height"),
            duration=info.get("duration", ctx.job.get("duration")),
        )
        return {
            "probe_info": info,
            "width": info.get("width"),
            "height": info.get("height"),
            "duration": info.get("duration"),
            "fps": info.get("fps"),
        }
