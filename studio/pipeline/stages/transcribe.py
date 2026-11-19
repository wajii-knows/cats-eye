"""Transcribe stage: transcribes audio to word-level timestamps using sherpa-onnx / Parakeet."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from studio import config, media, transcribe
from studio.contracts import Word, TranscriptSegment
from studio.pipeline.stage import Stage, StageContext, StageError

logger = logging.getLogger("studio.pipeline.transcribe")


class TranscribeStage(Stage):
    name = "transcribe"
    version = 1
    inputs = ("audio_wav",)
    outputs = ("transcript_words", "words_json")
    cacheable = True

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        d = Path(ctx.job["dir"])
        words_file = d / "words.json"

        # If already transcribed in job directory, reuse
        if words_file.is_file():
            try:
                words = json.loads(words_file.read_text(encoding="utf-8"))
                return {
                    "transcript_words": words,
                    "words_json": str(words_file),
                }
            except Exception:
                pass

        wav = ctx.inputs.get("audio_wav") or str(d / "audio.wav")
        if not Path(wav).is_file():
            raise StageError("WAV_MISSING", f"Audio WAV file missing: {wav}")

        # Check model ready
        if not transcribe.model_ready() and not transcribe.REMOTE:
            from studio import fetch_model
            fetch_model.ensure()

        try:
            audio = media.load_audio(Path(wav))
            words = transcribe.transcribe(audio)
        except Exception as e:
            raise StageError("TRANSCRIBE_FAILED", f"Speech recognition failed: {e}") from e

        # Save words.json to job directory
        words_file.write_text(json.dumps(words, indent=2), encoding="utf-8")

        return {
            "transcript_words": words,
            "words_json": str(words_file),
        }
