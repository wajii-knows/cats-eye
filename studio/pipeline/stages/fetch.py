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
            rel_src = f"users/{ctx.user_id}/jobs/{ctx.job.get('id', 'job')}/sources/{video_file.name}"
            if ctx.durable_store:
                try:
                    ctx.durable_store.put_file(video_file, rel_src)
                    ctx.job["source_durable"] = True
                except Exception as e:
                    logger.warning("Could not persist source video to Drive: %s", e)

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
            rel_src = f"users/{ctx.user_id}/jobs/{ctx.job.get('id', 'job')}/sources/{path.name}"
            if not path.is_file():
                # If missing locally in Colab mode, check durable store
                if ctx.durable_store and ctx.durable_store.exists(rel_src):
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        ctx.durable_store.get_file(rel_src, path)
                    except Exception as e:
                        logger.warning("Could not restore source from Drive: %s", e)

            if not path.is_file():
                raise StageError("FILE_NOT_FOUND", f"Uploaded video file not found: {path}")

            if ctx.durable_store and not ctx.durable_store.exists(rel_src):
                try:
                    ctx.durable_store.put_file(path, rel_src)
                    ctx.job["source_durable"] = True
                except Exception as e:
                    logger.warning("Could not persist uploaded video to Drive: %s", e)

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
