"""Tests for the new snapshot extraction and multishot features."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import ai, brain, config, jobs, media, render


def test_snapshots_and_multishots():
    work = Path(tempfile.mkdtemp(prefix="pc-test-ms-"))
    try:
        print("\n=== Snapshot Extraction & Multishots ===")
        # 1. Create a dummy test video with audio
        src = work / "source.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=15:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=15",
            "-c:v", "libx264", "-c:a", "aac", str(src)
        ]
        subprocess.run(cmd, check=True)
        assert src.is_file(), "Source video was not created"

        # 2. Test media.extract_snapshots
        snap_dir = work / "snaps"
        snaps = media.extract_snapshots(src, snap_dir, start=0.0, end=15.0, interval_sec=3.0, max_snapshots=10)
        assert len(snaps) >= 3, f"Expected at least 3 snapshots, got {len(snaps)}"
        for s in snaps:
            assert "time" in s and "data" in s and "mime" in s
            assert len(s["data"]) > 100
        print("PASS media.extract_snapshots extracts periodic thumbnail snapshots with base64 data")

        # 3. Test brain.plan_multishots
        # Short clip (<=4s) returns 1 shot
        shots_short = brain.plan_multishots(0.0, 3.5)
        assert len(shots_short) == 1 and shots_short[0]["type"] == "wide"
        assert shots_short[0]["start"] == 0.0 and shots_short[0]["end"] == 3.5
        print("PASS brain.plan_multishots short clips (<=4s) use a single establishing shot")

        # Longer clip (20s) returns a dynamic sequence of shots
        shots_long = brain.plan_multishots(0.0, 20.0)
        assert len(shots_long) >= 3, f"Expected >= 3 shots for 20s clip, got {len(shots_long)}"
        # Verify contiguous coverage
        assert shots_long[0]["start"] == 0.0
        assert shots_long[-1]["end"] == 20.0
        for i in range(len(shots_long) - 1):
            assert abs(shots_long[i]["end"] - shots_long[i + 1]["start"]) <= 0.05
        # Verify multiple different shot types are used
        shot_types = {s["type"] for s in shots_long}
        assert len(shot_types) > 1, f"Expected multiple shot types, got {shot_types}"
        print(f"PASS brain.plan_multishots generates dynamic multishot sequence: {[s['type'] for s in shots_long]}")

        # Custom shot input normalization
        custom_input = [
            {"start": 1.0, "end": 6.0, "type": "close"},
            {"start": 6.0, "end": 12.0, "type": "crop"}
        ]
        shots_norm = brain.plan_multishots(0.0, 15.0, shots_input=custom_input)
        assert shots_norm[0]["start"] == 0.0  # gap filled
        assert shots_norm[-1]["end"] == 15.0  # trailing gap filled
        print("PASS brain.plan_multishots sanitizes and tiles user/AI shot input")

        # 4. Test multimodal AI call
        config.AI_PROVIDER = "gemini"
        config.AI_BASE_URL = ""
        seen_calls = []

        def mock_post(url, json=None, headers=None, **_kw):
            seen_calls.append((url, json))
            return type("Resp", (), {
                "status_code": 200,
                "text": "{}",
                "json": lambda self: {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": json_dump([{"start": 1, "end": 14, "score": 85,
                                                           "hook": "VIRAL MOMENT", "title": "Check this out",
                                                           "description": "Amazing clip", "hashtags": ["viral"],
                                                           "highlight_words": ["amazing"], "reason": "Great visual",
                                                           "shots": [{"start": 1, "end": 7, "type": "wide"},
                                                                     {"start": 7, "end": 14, "type": "close"}]}])}]
                        }
                    }],
                    "usageMetadata": {"promptTokenCount": 150, "totalTokenCount": 180}
                }
            })()

        def json_dump(obj):
            import json as _j
            return _j.dumps(obj)

        old_post = ai.httpx.post
        ai.httpx.post = mock_post
        try:
            out, sent, wrote = ai.call("test-model", "Find clips", brain.SCHEMA, images=snaps[:2])
            assert len(out) == 1
            assert "shots" in out[0]
            # Check inlineData was included in parts for gemini
            req_parts = seen_calls[0][1]["contents"][0]["parts"]
            assert len(req_parts) == 3  # 1 text + 2 images
            assert req_parts[1]["inlineData"]["mimeType"] == "image/jpeg"
            print("PASS ai.call transmits multimodal images to AI provider correctly")
        finally:
            ai.httpx.post = old_post

        # 5. Test rendering multishot clip with ffmpeg
        clip_data = {
            "start": 0.0,
            "end": 10.0,
            "hook": "TEST HOOK",
            "style": "bounce",
            "layout": "multishot",
            "shots": [
                {"start": 0.0, "end": 3.0, "type": "wide"},
                {"start": 3.0, "end": 6.5, "type": "close"},
                {"start": 6.5, "end": 10.0, "type": "zoom"}
            ]
        }
        words = [
            {"w": "Hello", "s": 0.5, "e": 1.2},
            {"w": "multishots", "s": 1.4, "e": 2.5},
            {"w": "world", "s": 3.2, "e": 4.0},
            {"w": "action", "s": 7.0, "e": 8.5}
        ]
        out_clip_dir = work / "rendered_clips"
        res = render.render_clip(src, out_clip_dir, "test_clip", clip_data, words, 1280, 720)
        assert (out_clip_dir / "test_clip.mp4").is_file()
        assert (out_clip_dir / "test_clip.jpg").is_file()
        assert (out_clip_dir / "test_clip.ass").is_file()
        assert (out_clip_dir / "test_clip.mp4").stat().st_size > 10000
        print("PASS render.render_clip renders multishot video with synced captions and cover image")

        print("\nALL SNAPSHOT & MULTISHOT TESTS PASSED!\n")
        return True
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    ok = test_snapshots_and_multishots()
    sys.exit(0 if ok else 1)
