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
            audio = media.load_audio(Path(wav))
            total_samples = len(audio)
            CHUNK_SAMPLES = 5 * 60 * 16000  # 5 minutes of 16kHz mono audio

            if total_samples <= CHUNK_SAMPLES:
                saved = ctx.resume_state("chunk_0")
                if saved and "words" in saved:
                    words = saved["words"]
                else:
                    words = transcribe.transcribe(audio)
                    ctx.checkpoint("chunk_0", {"words": words, "chunk_index": 0})
            else:
                words = []
                chunks = [(i, i + CHUNK_SAMPLES) for i in range(0, total_samples, CHUNK_SAMPLES)]
                for idx, (c_start, c_end) in enumerate(chunks):
                    slice_end = min(total_samples, c_end)
                    chk_key = f"chunk_{idx}"
                    saved = ctx.resume_state(chk_key)
                    if saved and "words" in saved:
                        chunk_words = saved["words"]
                    else:
                        sub_audio = audio[c_start:slice_end]
                        shift_sec = c_start / 16000.0
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
