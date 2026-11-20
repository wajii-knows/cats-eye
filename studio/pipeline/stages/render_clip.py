"""Render clip stage: renders a vertical 9:16 clip with styled captions and cover image."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from studio import render
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.render_clip")


class RenderClipStage(Stage):
    name = "render_clip"
    version = 1
    inputs = ("source_video", "clip_spec", "transcript_words")
    outputs = ("rendered_video", "rendered_jpg", "rendered_ass")
    cacheable = True

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        clip = ctx.inputs.get("current_clip", {})
        return {
            "start": clip.get("start"),
            "end": clip.get("end"),
            "hook": clip.get("hook", ""),
            "style": clip.get("style", "bounce"),
            "layout": clip.get("layout", "blur"),
            "shots": clip.get("shots", []),
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        source = ctx.inputs.get("source_video") or ctx.job.get("source_path")
        if not source or not Path(source).is_file():
            raise StageError("SOURCE_VIDEO_MISSING", f"Source video not found for rendering: {source}")

        clip = ctx.inputs.get("current_clip") or {}
        if not clip:
            raise StageError("CLIP_SPEC_MISSING", "No clip spec provided to RenderClipStage")

        words = ctx.inputs.get("transcript_words", [])
        w = int(ctx.inputs.get("src_w") or ctx.job.get("width") or 1920)
        h = int(ctx.inputs.get("src_h") or ctx.job.get("height") or 1080)

        clip_name = clip.get("name") or clip.get("id") or f"clip_{int(clip.get('start', 0))}"
        out_dir = Path(ctx.job["dir"]) / "clips" / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        expected_mp4 = out_dir / f"{clip_name}.mp4"
        expected_jpg = out_dir / f"{clip_name}.jpg"
        expected_ass = out_dir / f"{clip_name}.ass"
        job_id = ctx.job.get("id", "job")
        durable_prefix = f"users/{ctx.user_id}/jobs/{job_id}/clips/{clip_name}"

        # Check durable store for existing finished render
        if ctx.durable_store and not expected_mp4.is_file():
            rel_mp4 = f"{durable_prefix}/{clip_name}.mp4"
            if ctx.durable_store.exists(rel_mp4):
                try:
                    ctx.durable_store.get_file(rel_mp4, expected_mp4)
                    if ctx.durable_store.exists(f"{durable_prefix}/{clip_name}.jpg"):
                        ctx.durable_store.get_file(f"{durable_prefix}/{clip_name}.jpg", expected_jpg)
                    if ctx.durable_store.exists(f"{durable_prefix}/{clip_name}.ass"):
                        ctx.durable_store.get_file(f"{durable_prefix}/{clip_name}.ass", expected_ass)
                    res = {"file": expected_mp4.name, "cover": expected_jpg.name, "layout": clip.get("layout", "crop")}
                    return {
                        "rendered_video": str(expected_mp4),
                        "rendered_jpg": str(expected_jpg),
                        "rendered_ass": str(expected_ass),
                        "clip_result": res,
                    }
                except Exception as e:
                    logger.warning("Could not restore render from Drive: %s", e)

        try:
            res = render.render_clip(
                src=Path(source),
                out_dir=out_dir,
                name=clip_name,
                clip=clip,
                words=words,
                src_w=w,
                src_h=h,
            )
        except Exception as e:
            raise StageError("RENDER_FAILED", f"ffmpeg rendering failed: {e}") from e

        mp4_path = out_dir / res["file"]
        jpg_path = out_dir / res["cover"]
        ass_path = out_dir / f"{clip_name}.ass"

        if ctx.durable_store:
            try:
                ctx.durable_store.put_file(mp4_path, f"{durable_prefix}/{mp4_path.name}")
                ctx.durable_store.put_file(jpg_path, f"{durable_prefix}/{jpg_path.name}")
                if ass_path.is_file():
                    ctx.durable_store.put_file(ass_path, f"{durable_prefix}/{ass_path.name}")
            except Exception as e:
                logger.warning("Could not persist rendered clip to Drive: %s", e)

        return {
            "rendered_video": str(mp4_path),
            "rendered_jpg": str(jpg_path),
            "rendered_ass": str(ass_path),
            "clip_result": res,
        }
