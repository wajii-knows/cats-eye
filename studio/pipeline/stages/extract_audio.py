"""Extract audio stage: extracts 16kHz mono audio and computes loudness profile."""
from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any, Dict

import numpy as np

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
            opts = ctx.job.get("options", {})
            with wave.open(str(wav_file), "rb") as wf:
                rate = wf.getframerate() or 16000
                total_frames = wf.getnframes()
                lo_f = max(0, int((opts.get("range_start") or 0) * rate))
                hi_f = min(total_frames, int(opts["range_end"] * rate)) if opts.get("range_end") else total_frames
                wf.setpos(lo_f)
                rem = max(0, hi_f - lo_f)
                chunk_secs = 60
                loud_parts = []
                while rem > 0:
                    take = min(rem, chunk_secs * rate)
                    raw = wf.readframes(take)
                    if not raw:
                        break
                    arr = np.frombuffer(raw, dtype=np.int16)
                    if len(arr) >= rate:
                        loud_parts.append(media.loudness_per_second(arr))
                    rem -= len(arr)
        except Exception as e:
            raise StageError("EXTRACT_AUDIO_FAILED", f"Failed to extract audio: {e}") from e

        loudness = np.concatenate(loud_parts) if loud_parts else np.zeros(0, dtype=np.float32)
        loudness_list = loudness.tolist() if hasattr(loudness, "tolist") else list(loudness)

        return {
            "audio_wav": str(wav_file),
            "loudness_db": loudness_list,
            "duration": total_frames / float(rate),
        }
