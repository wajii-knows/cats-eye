"""Fetch stage: downloads media from URL or validates uploaded video."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from studio import config, media
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.fetch")


class FetchStage(Stage):
    name = "fetch"
    version = 1
    inputs = ()
    outputs = ("source_video", "metadata", "thumb")
    cacheable = True

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        src = ctx.job.get("source", {})
        opts = ctx.job.get("options", {})
        return {
            "source_type": src.get("type", "upload"),
            "url": src.get("url", "") if src.get("type") == "url" else "",
            "quality": int(opts.get("quality", 1080) or 1080),
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        src = ctx.job.get("source", {})
        opts = ctx.job.get("options", {})
        d = Path(ctx.job["dir"])
        d.mkdir(parents=True, exist_ok=True)

        if src.get("type") == "url":
            url = src["url"]
            try:
                meta = media.info(url)
            except Exception as e:
                # Network or remote service error can be retryable
                raise StageError("FETCH_INFO_FAILED", f"Could not inspect URL: {e}", retryable=True) from e

            if meta.get("kind") != "video":
                raise StageError("NOT_A_VIDEO", "That link is a channel or playlist, not a single video.")
            if meta.get("is_live"):
                raise StageError("LIVE_STREAM", "This link is a live stream. Use live recording instead.")

            ctx.job.update(title=meta.get("title", ""), uploader=meta.get("uploader", ""), duration=meta.get("duration", 0))
            if media.save_thumb(d, url=url):
                ctx.job["thumb"] = True

            try:
                audio_file = media.download_audio(url, d)
            except Exception as e:
                raise StageError("DOWNLOAD_AUDIO_FAILED", f"Failed downloading audio: {e}", retryable=True) from e

            video_file = media.download(url, d, int(opts.get("quality", 1080) or 1080))
            return {
                "source_video": str(video_file),
                "audio_file": str(audio_file),
                "metadata": meta,
                "title": meta.get("title", ""),
                "duration": meta.get("duration", 0),
            }
        else:
            # Uploaded or local video
            path = Path(src.get("path", ""))
            if not path.is_file():
                raise StageError("FILE_NOT_FOUND", f"Uploaded video file not found: {path}")
            info = media.probe(path)
            ctx.job.update(source_path=str(path), width=info.get("width"), height=info.get("height"), duration=info.get("duration"))
            if media.save_thumb(d, video=path):
                ctx.job["thumb"] = True
            return {
                "source_video": str(path),
                "metadata": info,
                "title": ctx.job.get("title") or path.stem,
                "duration": info.get("duration", 0),
            }
