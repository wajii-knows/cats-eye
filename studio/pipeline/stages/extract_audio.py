"""Extract audio stage: extracts 16kHz mono audio and computes loudness profile."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from studio import media
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.extract_audio")


class ExtractAudioStage(Stage):
    name = "extract_audio"
    version = 1
    inputs = ("source_video",)
    outputs = ("audio_wav", "loudness_db")
    cacheable = True

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        opts = ctx.job.get("options", {})
        return {
            "range_start": opts.get("range_start", 0),
            "range_end": opts.get("range_end"),
        }

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        source = ctx.inputs.get("source_video") or ctx.job.get("source_path")
        audio_in = ctx.inputs.get("audio_file") or source
        if not audio_in or not Path(audio_in).is_file():
            raise StageError("AUDIO_SOURCE_MISSING", f"No input file for audio extraction: {audio_in}")

        d = Path(ctx.job["dir"])
        wav_file = d / "audio.wav"

        try:
            media.extract_audio(Path(audio_in), wav_file)
            audio = media.load_audio(wav_file)
        except Exception as e:
            raise StageError("EXTRACT_AUDIO_FAILED", f"Failed to extract audio: {e}") from e

        opts = ctx.job.get("options", {})
        lo = int((opts.get("range_start") or 0) * 16000)
        hi = min(len(audio), int(opts["range_end"] * 16000)) if opts.get("range_end") else len(audio)
        loudness = media.loudness_per_second(audio[lo:hi])
        loudness_list = loudness.tolist() if hasattr(loudness, "tolist") else list(loudness)

        return {
            "audio_wav": str(wav_file),
            "loudness_db": loudness_list,
            "duration": len(audio) / 16000,
        }
