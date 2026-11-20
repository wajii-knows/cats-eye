"""Transcribe stage: transcribes audio to word-level timestamps using sherpa-onnx / Parakeet."""
from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
    progress_unit = "5m_chunk"

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
            CHUNK_SAMPLES = 5 * 60 * 16000  # 5 minutes of 16kHz mono audio
            with wave.open(str(wav), "rb") as wf:
                rate = wf.getframerate() or 16000
                total_samples = wf.getnframes()
                words = []
                num_chunks = max(1, (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
                for idx in range(num_chunks):
                    c_start = idx * CHUNK_SAMPLES
                    take = min(CHUNK_SAMPLES, max(0, total_samples - c_start))
                    chk_key = f"chunk_{idx}"
                    saved = ctx.resume_state(chk_key)
                    if saved and "words" in saved:
                        chunk_words = saved["words"]
                    else:
                        wf.setpos(c_start)
                        sub_audio = np.frombuffer(wf.readframes(take), dtype=np.int16) if take > 0 else np.zeros(0, dtype=np.int16)
                        shift_sec = c_start / float(rate)
                        raw_words = transcribe.transcribe(sub_audio)
                        chunk_words = [
                            dict(w, s=round(w.get("s", 0) + shift_sec, 2), e=round(w.get("e", 0) + shift_sec, 2))
                            for w in raw_words
                        ]
                        ctx.checkpoint(chk_key, {"words": chunk_words, "chunk_index": idx, "shift_sec": shift_sec})
                    words.extend(chunk_words)
        except Exception as e:
            raise StageError("TRANSCRIBE_FAILED", f"Speech recognition failed: {e}") from e

        # Save words.json to job directory
        words_file.write_text(json.dumps(words, indent=2), encoding="utf-8")

        return {
            "transcript_words": words,
            "words_json": str(words_file),
        }
