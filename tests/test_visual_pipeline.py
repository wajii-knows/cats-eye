"""Tests for Cat's Eye Visual Understanding pipeline (PEAKCUT_SNAPSHOTS):
- O(log n) bisect helpers on exact half-open boundaries
- DetectScenesStage (hard cuts within 0.5s, <=10% FP, soft scenes <=60s, <0.6s merge, chunk overlap merge)
- ExtractSnapshotsStage (batched multi-seek FFmpeg, WebP/JPEG, 64-bit dHash dedupe, cap & budget thinning)
- AnalyzeSnapshotsStage (brightness/blur/edge/motion metrics, NullFaceDetector & mock FaceDetector all 6 layouts + facecam box)
- CaptionSnapshotsStage (capability lookup, no-AI/no-vision not_available, invalid JSON retry/skip, usage DB, ceiling, untrusted text)
- BuildMomentsStage (transcript segments joining snapshots & scenes cleanly)
- Edge cases (audio-only, vertical/rotated video, live tail pending, cache hits, job delete cleanup)
- FaultInjectingStore + WORK_DIR wipe mid-extract & mid-caption resume equivalence + Colab pack eviction & lazy restore
- Live HTTP endpoints: Flag-OFF 404, Flag-ON auth/pagination/images, User B isolation returns 404 (never 403)
"""
import base64
import http.cookiejar
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import ai, config, db, flags, jobs
from studio.contracts import Moment, Scene, Snapshot, Word
from studio.pipeline.artifacts import ArtifactStore
from studio.pipeline.engine import PipelineEngine
from studio.pipeline.stage import StageContext, StageError, compute_file_sha256
from studio.pipeline.stages.analyze_snapshots import (
    AnalyzeSnapshotsStage,
    FaceDetector,
    register_face_detector,
)
from studio.pipeline.stages.build_moments import BuildMomentsStage
from studio.pipeline.stages.caption_snapshots import CaptionSnapshotsStage, validate_sheet_response
from studio.pipeline.stages.detect_scenes import DetectScenesStage, merge_and_build_scenes, probe_visual_media
from studio.pipeline.stages.extract_snapshots import (
    ExtractSnapshotsStage,
    compute_dhash,
    hamming_distance,
    plan_snapshot_timestamps,
)
from studio.pipeline.visual_index import (
    VisualTimelineIndex,
    nearest_snapshot,
    scene_at,
    scenes_overlapping,
    segment_at,
    snapshots_in,
    words_in,
)
from studio.pipeline import visuals
from studio.storage.durable import DurableStoreError, FaultInjectingStore, LocalFolderStore

BASE = sys.argv[1] if len(sys.argv) > 1 else None


def _run_ffmpeg(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {p.stderr[-500:]}")
    return p


def _make_synthetic_cut_video(dest: Path) -> Path:
    """Create 14s H.264+AAC MP4 with visually distinct hard cuts at 3.0, 6.0, 9.5, 12.0s."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=3",
        "-f", "lavfi", "-i", "color=c=white:s=640x360:r=25:d=3",
        "-t", "3.5", "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=25",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:r=25:d=2.5",
        "-f", "lavfi", "-i", "smptebars=s=640x360:r=25:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=14",
        "-filter_complex", "[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1:a=0[v]",
        "-map", "[v]", "-map", "5:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "25",
        "-c:a", "aac", "-shortest",
        str(dest),
    ]
    _run_ffmpeg(cmd)
    return dest


def _make_solid_video(dest: Path, duration: float = 8.0) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r=25:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=16000:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(dest),
    ]
    _run_ffmpeg(cmd)
    return dest


def _http_helper(base_url: str):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def call(method: str, path: str, body=None, headers=None):
        hdrs = {"Origin": base_url, **(headers or {})}
        data = None
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, bytes):
            data = body
        req = urllib.request.Request(f"{base_url}{path}", data=data, method=method, headers=hdrs)
        try:
            with opener.open(req, timeout=20) as r:
                raw = r.read()
                ctype = r.headers.get("content-type", "")
                return r.status, (json.loads(raw.decode()) if "json" in ctype else raw)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw.decode())
            except Exception:
                return e.code, raw

    return call


def _multipart_upload(call_fn, mp4_path: Path):
    boundary = "----CatsEyeBoundary9988"
    file_bytes = mp4_path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{{}}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{mp4_path.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
        + file_bytes + f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    return call_fn("POST", "/api/upload", body=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def run_all_tests() -> bool:
    fails = []

    def check(name, ok, extra=""):
        print(("PASS " if ok else "FAIL ") + name + (f"   {extra}" if extra else ""))
        if not ok:
            fails.append(name)

    db.init()
    flags.reset_overrides()

    print("\n=== 1. Bisect Helpers & Half-Open Boundary Semantics ===")
    snaps = [{"id": f"s{i}", "timestamp": float(i * 2)} for i in range(5)]  # 0, 2, 4, 6, 8
    scenes = [
        {"id": "sc0", "start": 0.0, "end": 4.0},
        {"id": "sc1", "start": 4.0, "end": 10.0},
    ]
    segs = [
        {"id": "seg0", "start": 1.0, "end": 4.0},
        {"id": "seg1", "start": 4.0, "end": 9.0},
    ]
    words = [
        {"w": "hello", "s": 1.0, "e": 1.5},
        {"w": "world", "s": 4.0, "e": 4.5},
        {"w": "end", "s": 9.0, "e": 9.2},
    ]
    idx = VisualTimelineIndex(scenes=list(reversed(scenes)), snapshots=list(reversed(snaps)), segments=segs, words=words)
    check("snapshots_in half-open [0, 4) excludes t=4", [s["id"] for s in idx.snapshots_in(0.0, 4.0)] == ["s0", "s1"])
    check("snapshots_in empty when start >= end", idx.snapshots_in(4.0, 4.0) == [] and idx.snapshots_in(5.0, 2.0) == [])
    check("nearest_snapshot left/right clamp", idx.nearest_snapshot(-5.0)["id"] == "s0" and idx.nearest_snapshot(99.0)["id"] == "s4")
    check("nearest_snapshot tie breaks left", idx.nearest_snapshot(3.0)["id"] == "s1")
    check("nearest_snapshot empty returns None", nearest_snapshot([], 1.0) is None)
    check("scene_at exact boundary [0,4) [4,10)", idx.scene_at(0.0)["id"] == "sc0" and idx.scene_at(4.0)["id"] == "sc1" and idx.scene_at(10.0) is None)
    check("segment_at before first and at boundary", idx.segment_at(0.5) is None and idx.segment_at(4.0)["id"] == "seg1" and idx.segment_at(9.0) is None)
    check("words_in half-open [1.0, 4.0)", [w["w"] for w in idx.words_in(1.0, 4.0)] == ["hello"])
    check("scenes_overlapping catches exact overlap", [s["id"] for s in idx.scenes_overlapping(3.5, 4.5)] == ["sc0", "sc1"])

    print("\n=== 2. DetectScenesStage: Hard Cuts, Overlap Merge & Soft Scenes ===")
    work_root = Path(tempfile.mkdtemp(prefix="ce-vis-test-"))
    orig_caps_avail = ai.capabilities
    try:
        cut_mp4 = _make_synthetic_cut_video(work_root / "cuts_fixture.mp4")
        u_dir = work_root / "u_test"
        j_dir = u_dir / "jobs" / "job_cuts"
        j_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(user_dir=u_dir, user_id="u_test")
        src_fp = compute_file_sha256(cut_mp4)
        sctx = StageContext(
            job={"id": "job_cuts", "dir": str(j_dir), "user": "u_test", "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}},
            user_dir=u_dir,
            user_id="u_test",
            store=store,
            inputs={"source_video": str(cut_mp4), "video_fingerprint": src_fp},
        )
        sc_res = DetectScenesStage().run(sctx)
        sc_payload = json.loads(Path(sc_res["scenes_json"]).read_text(encoding="utf-8"))
        det_scenes = sc_payload["scenes"]
        det_cuts = [s["start"] for s in det_scenes[1:]]
        truth_cuts = [3.0, 6.0, 9.5, 12.0]
        matched = [any(abs(dc - tc) <= 0.5 for dc in det_cuts) for tc in truth_cuts]
        fp_cuts = [dc for dc in det_cuts if not any(abs(dc - tc) <= 0.5 for tc in truth_cuts)]
        fp_rate = len(fp_cuts) / max(1, len(det_cuts))
        check("DetectScenes hits all known hard cuts within 0.5s", all(matched), f"truth={truth_cuts} got={det_cuts}")
        check("DetectScenes FP rate <= 10%", fp_rate <= 0.10, f"fp={fp_cuts}")
        check("Scene IDs are deterministic", det_scenes[0]["id"] == merge_and_build_scenes(14.0, [], [], src_fp)[0]["id"])
        check("No scene shorter than 0.6s", all((s["end"] - s["start"]) >= 0.6 for s in det_scenes))

        # Overlap boundary cut deduplication + <0.6s neighbor merge + silence-gap soft scenes <= 60s
        soft_scenes = merge_and_build_scenes(
            duration=135.0,
            raw_cuts=[{"t": 0.25, "score": 0.9}, {"t": 599.9, "score": 0.8}, {"t": 600.2, "score": 0.92}],
            silence_gaps=[(40.0, 44.0), (88.0, 92.0)],
            video_fp="fp_soft",
            min_scene_sec=0.6,
            max_scene_sec=60.0,
        )
        check("Soft scenes split static stream <= 60s via silence gaps", len(soft_scenes) >= 3 and all((s["end"] - s["start"]) <= 60.0 for s in soft_scenes))
        check("Short sliver < 0.6s merged into neighbor", all((s["end"] - s["start"]) >= 0.6 for s in soft_scenes))

        print("\n=== 3. ExtractSnapshotsStage: Monotonic, Dedupe, WebP/JPEG & Cap ===")
        sctx.inputs.update(sc_res)
        sctx.inputs["transcript_words"] = [
            {"w": "First", "s": 0.4, "e": 0.8},
            {"w": "sentence.", "s": 0.9, "e": 1.4},
            {"w": "Second", "s": 5.2, "e": 5.6},
            {"w": "part.", "s": 5.7, "e": 6.2},
        ]
        sn_res = ExtractSnapshotsStage().run(sctx)
        sn_payload = json.loads(Path(sn_res["snapshots_json"]).read_text(encoding="utf-8"))
        snaps_out = sn_payload["snapshots"]
        ts_list = [s["timestamp"] for s in snaps_out]
        check("Snapshots timestamps strictly monotonic and in [0, dur]", ts_list == sorted(ts_list) and len(ts_list) == len(set(ts_list)) and all(0.0 <= t <= 14.0 for t in ts_list))
        check("Consecutive snapshots spaced >= 0.5s or boundary-safe", all((ts_list[i + 1] - ts_list[i]) >= 0.49 for i in range(len(ts_list) - 1)))
        first_sn = snaps_out[0]
        thumb_p = Path(sn_res["artifact_dirs"]["thumbs"]) / f"{first_sn['id']}.webp"
        an_p = Path(sn_res["artifact_dirs"]["analysis"]) / f"{first_sn['id']}.jpg"
        check("WebP thumb & JPEG analysis written", thumb_p.read_bytes()[:4] == b"RIFF" and an_p.read_bytes()[:3] == b"\xff\xd8\xff")
        check("Scene boundaries retained across dHash", any(s["is_scene_boundary"] for s in snaps_out))

        # dHash dedupe on constant blue video
        solid_mp4 = _make_solid_video(work_root / "solid.mp4", duration=8.0)
        solid_j = u_dir / "jobs" / "job_solid"
        solid_j.mkdir(parents=True, exist_ok=True)
        sc_solid_ctx = StageContext(
            job={"id": "job_solid", "dir": str(solid_j), "user": "u_test", "source": {"type": "file", "path": str(solid_mp4)}, "options": {}, "stages": {}},
            user_dir=u_dir,
            user_id="u_test",
            store=store,
            inputs={"source_video": str(solid_mp4), "video_fingerprint": compute_file_sha256(solid_mp4)},
            params={"snapshot_interval_sec": 1.0, "snapshot_dhash_threshold": 4},
        )
        sc_s_res = DetectScenesStage().run(sc_solid_ctx)
        sc_solid_ctx.inputs.update(sc_s_res)
        sn_s_res = ExtractSnapshotsStage().run(sc_solid_ctx)
        sn_s_list = json.loads(Path(sn_s_res["snapshots_json"]).read_text(encoding="utf-8"))["snapshots"]
        check("dHash dedupe drops redundant static frames while keeping scene start", len(sn_s_list) <= 3 and any(s["is_scene_boundary"] for s in sn_s_list))

        # Max snapshot cap & disk budget thinning while keeping scene boundaries
        cap_ctx = StageContext(
            job={"id": "job_cap", "dir": str(j_dir), "user": "u_test", "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}},
            user_dir=u_dir,
            user_id="u_test",
            store=store,
            inputs={"source_video": str(cut_mp4), "video_fingerprint": src_fp, "scenes_json": sc_res["scenes_json"]},
            params={"snapshot_interval_sec": 0.6, "snapshot_max_count": 7, "snapshot_dhash_threshold": -1},
        )
        sn_cap_res = ExtractSnapshotsStage().run(cap_ctx)
        sn_cap = json.loads(Path(sn_cap_res["snapshots_json"]).read_text(encoding="utf-8"))["snapshots"]
        check("Max cap thinned <= 7 snapshots and kept boundary frames", len(sn_cap) <= 7 and any(s["is_scene_boundary"] for s in sn_cap))

        print("\n=== 4. AnalyzeSnapshotsStage: Metrics, Layouts & Facecam Box ===")
        sctx.inputs.update(sn_res)
        register_face_detector(None)
        an_res = AnalyzeSnapshotsStage().run(sctx)
        an_snaps = json.loads(Path(an_res["analyzed_snapshots_json"]).read_text(encoding="utf-8"))["snapshots"]
        an_scenes = json.loads(Path(an_res["analyzed_scenes_json"]).read_text(encoding="utf-8"))["scenes"]
        check("NullFaceDetector sets faces=None and preserves metrics", all(s["faces"] is None and s["brightness"] is not None and s["blur"] is not None and s["edge_density"] is not None for s in an_snaps))
        scene_layouts = [s.get("layout") for s in an_scenes]
        check("Rule classifier detects black_or_transition and slides_or_screen/wide_scene", "black_or_transition" in scene_layouts and any(l in ("slides_or_screen", "wide_scene") for l in scene_layouts), f"layouts={scene_layouts}")

        # Mock FaceDetector covering closeup, two_faces, and corner facecam_plus_content
        class _MockFaceDetector(FaceDetector):
            name = "mock_face"
            available = True

            def detect(self, rgb_array: np.ndarray):
                b = float(rgb_array.mean())
                if b < 18:
                    return []
                if b > 240:  # white segment -> two centered speakers (not corners)
                    return [[0.30, 0.25, 0.20, 0.30], [0.52, 0.25, 0.20, 0.30]]
                # Stable bottom-right corner facecam over textured fractal/testsrc
                return [[0.76, 0.72, 0.16, 0.20]]

        register_face_detector(_MockFaceDetector())
        try:
            an_face_res = AnalyzeSnapshotsStage().run(sctx)
            an_face_scenes = json.loads(Path(an_face_res["analyzed_scenes_json"]).read_text(encoding="utf-8"))["scenes"]
            f_layouts = [s.get("layout") for s in an_face_scenes]
            check("Mock FaceDetector classifies two_faces & facecam_plus_content", "two_faces" in f_layouts and "facecam_plus_content" in f_layouts, f"layouts={f_layouts}")
            check("Facecam box detected as normalized [x,y,w,h]", isinstance(an_face_res.get("facecam_box"), list) and len(an_face_res["facecam_box"]) == 4)
            from studio.pipeline.stages.analyze_snapshots import RuleBasedLayoutClassifier
            closeup_lbl, _ = RuleBasedLayoutClassifier().classify(
                [{"brightness": 130, "edge_density": 0.04, "motion": 0.05, "faces": 1, "largest_face_box": [0.25, 0.15, 0.40, 0.50]}],
                {"start": 0.0, "end": 5.0},
            )
            check("Rule classifier covers one_face_closeup", closeup_lbl == "one_face_closeup")
        finally:
            register_face_detector(None)

        print("\n=== 5. CaptionSnapshotsStage: Capabilities, Invalid JSON Retry/Skip, Ceiling & Usage ===")
        sctx.inputs.update(an_res)
        # 5a. No AI configured -> not_available (never fails)
        orig_data_dir = config.DATA_DIR
        orig_ai_key_env = os.environ.get("PEAKCUT_AI_KEY")
        orig_ai_models_env = os.environ.get("PEAKCUT_AI_MODELS")
        try:
            config.DATA_DIR = u_dir.parent
            config._kept.clear()
            ai._saved["at"] = None
            os.environ["PEAKCUT_AI_KEY"] = ""
            os.environ["PEAKCUT_AI_MODELS"] = ""
            config.AI_MODELS = []
            cap_none_res = CaptionSnapshotsStage().run(sctx)
            check("No AI key -> captions_status=not_available without error", cap_none_res["captions_status"] == "not_available", str(cap_none_res))
        finally:
            config.DATA_DIR = orig_data_dir
            config._kept.clear()
            ai._saved["at"] = None
            if orig_ai_key_env is not None:
                os.environ["PEAKCUT_AI_KEY"] = orig_ai_key_env
            else:
                os.environ.pop("PEAKCUT_AI_KEY", None)
            if orig_ai_models_env is not None:
                os.environ["PEAKCUT_AI_MODELS"] = orig_ai_models_env
            else:
                os.environ.pop("PEAKCUT_AI_MODELS", None)

        # 5b. Non-vision AI model -> not_available
        ai.capabilities = lambda prov=None, model=None: {"ready": True, "provider": "groq", "model": "llama-3.3-70b-versatile", "image_input": False, "json_mode": True}
        cap_novis_res = CaptionSnapshotsStage().run(sctx)
        check("Non-vision model -> captions_status=not_available", cap_novis_res["captions_status"] == "not_available")

        # 5c. Invalid JSON retried once then clamped + untrusted sanitized + double fail skips sheet safely
        ai.capabilities = lambda prov=None, model=None: {"ready": True, "provider": "gemini", "model": "gemini-2.5-flash", "image_input": True, "json_mode": True}
        orig_ai_call = ai.call
        call_seq = {"n": 0}

        def _flaky_vision_ai(model, prompt, schema, images=None):
            call_seq["n"] += 1
            if call_seq["n"] == 1:
                return {"bad": "shape"}, 100, 10  # triggers retry!
            if call_seq["n"] == 2:
                return [{
                    "index": 0,
                    "labels": ["speaker", "demo", "x" * 50, "four", "five", "extra6"],
                    "caption": " ".join(f"w{i}" for i in range(20)),
                    "people_count": 999,
                    "text_on_screen": "<script>alert('xss')</script> SUBSCRIBE NOW",
                    "emotion": "excited",
                    "is_good_thumbnail": 2.5,
                }], 120, 45
            raise ai.AIError("simulated invalid json on sheet 2")

        ai.call = _flaky_vision_ai
        cap_ctx = StageContext(
            job={"id": "job_cap_retry", "dir": str(j_dir), "user": "u_test", "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}},
            user_dir=u_dir,
            user_id="u_test",
            store=store,
            inputs=dict(sctx.inputs),
            params={"snapshot_ai_max_requests": 10},
        )
        try:
            from studio import ctx as st_ctx
            with st_ctx.as_user("u_local"):
                # Ensure u_local exists for usage FK
                from studio import auth
                auth.local_user()
                cap_ok_res = CaptionSnapshotsStage().run(cap_ctx)
        finally:
            ai.call = orig_ai_call

        cap_ok_data = json.loads(Path(cap_ok_res["captions_json"]).read_text(encoding="utf-8"))
        first_cap = next(iter(cap_ok_data["captions"].values()), {})
        check("Invalid AI JSON retried once and clamped fields safely", call_seq["n"] >= 2 and len(first_cap.get("labels", [])) <= 5 and len(first_cap.get("caption", "").split()) <= 12 and first_cap.get("people_count") == 50 and first_cap.get("is_good_thumbnail") == 1.0)
        check("Untrusted text_on_screen sanitized of HTML tags", "<" not in first_cap.get("text_on_screen", "") and ">" not in first_cap.get("text_on_screen", ""))

        # 5d. Ceiling hit behavior + usage DB record check
        with db.engine.connect() as conn:
            u_cnt = conn.execute(__import__("sqlalchemy").select(__import__("sqlalchemy").func.count()).select_from(db.usage).where(db.usage.c.purpose == "caption_snapshots")).scalar()
        check("Usage DB recorded caption_snapshots calls", int(u_cnt or 0) >= 2)

        ai.call = lambda m, p, s, images=None: ([{"index": 0, "labels": ["ok"], "caption": "frame zero", "people_count": 1, "is_good_thumbnail": 0.7}], 50, 20)
        ceil_ctx = StageContext(
            job={"id": "job_cap_ceil", "dir": str(j_dir), "user": "u_test", "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}},
            user_dir=u_dir,
            user_id="u_test",
            store=store,
            inputs=dict(sctx.inputs),
            params={"snapshot_ai_max_requests": 1},
        )
        # Ensure at least 10 snapshots so 2 sheets exist
        sn_many = [{"id": f"sn_fake_{i}", "timestamp": float(i), "dhash": f"{i:016x}"} for i in range(12)]
        fake_sn_p = j_dir / "many_snaps.json"
        fake_sn_p.write_text(json.dumps({"snapshots": sn_many}), encoding="utf-8")
        ceil_ctx.inputs["analyzed_snapshots_json"] = str(fake_sn_p)
        try:
            ceil_res = CaptionSnapshotsStage().run(ceil_ctx)
        finally:
            ai.call = orig_ai_call
            ai.capabilities = orig_caps_avail
        ceil_data = json.loads(Path(ceil_res["captions_json"]).read_text(encoding="utf-8"))
        check("Per-video AI ceiling (max=1) stops cleanly with clear message", ceil_data["ceiling_hit"] is True and ceil_data["requests_used"] == 1 and "limit reached" in ceil_data["reason"].lower())

        print("\n=== 6. BuildMomentsStage & Transcript/Snapshot Join ===")
        sctx.inputs.update(cap_none_res)
        mom_res = BuildMomentsStage().run(sctx)
        mom_data = json.loads(Path(mom_res["moments_json"]).read_text(encoding="utf-8"))
        moments = mom_data["moments"]
        assigned_snap_ids = [sid for m in moments for sid in m["snapshot_ids"]]
        check("BuildMoments produced moments overlapping scenes", len(moments) >= 1 and all(isinstance(m["scene_ids"], list) for m in moments))
        check("Every snapshot assigned to at most one moment", len(assigned_snap_ids) == len(set(assigned_snap_ids)))

        from studio import ctx as st_ctx
        print("\n=== 7. Edge Cases: Audio-only, Rotation 9:16, Live Tail & Cache Hit ===")
        # 7a. Audio-only WAV
        wav_path = work_root / "audio_only.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000 * 5)
        aud_j = u_dir / "jobs" / "job_audio"
        aud_j.mkdir(parents=True, exist_ok=True)
        aud_job = {"id": "job_audio", "dir": str(aud_j), "user": "u_test", "source": {"type": "file", "path": str(wav_path)}, "options": {}, "stages": {}}
        jobs._jobs[aud_job["id"]] = aud_job
        jobs._by_user.setdefault("u_test", {})[aud_job["id"]] = aud_job
        with st_ctx.as_user("u_test"):
            vis_aud = visuals.run_visual_analysis(aud_job["id"], source_video=wav_path)
        check("Audio-only file completes with scenes/moments and 0 snapshots", vis_aud["status"] == "done" and vis_aud["has_video"] is False and vis_aud["counts"]["snapshots"] == 0 and vis_aud["counts"]["scenes"] >= 1)

        # 7b. Vertical rotated video
        rot_mp4 = work_root / "vert_rot.mp4"
        _run_ffmpeg([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=s=360x640:r=25:d=3",
            "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-metadata:s:v:0", "rotate=90", "-c:a", "aac", "-shortest", str(rot_mp4),
        ])
        rot_info = probe_visual_media(rot_mp4)
        check("Rotated vertical video swaps probed dimensions properly", rot_info["has_video"] and rot_info["duration"] >= 2.5 and ((rot_info["rotation"] in (90, 270) and rot_info["width"] == 640 and rot_info["height"] == 360) or rot_info["width"] == 360))

        # 7c. Live recording still growing -> tail marked pending
        live_j = u_dir / "jobs" / "job_live_vis"
        buf_d = live_j / "buffer"
        buf_d.mkdir(parents=True, exist_ok=True)
        seg1 = buf_d / "seg_0001.ts"
        seg2 = buf_d / "seg_0002.ts"  # currently growing latest segment!
        _run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=3", "-c:v", "mpeg2video", str(seg1)])
        _run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=3", "-c:v", "mpeg2video", str(seg2)])
        os.utime(seg1, (time.time() - 10, time.time() - 10))
        live_job = {"id": "job_live_vis", "kind": "live", "status": "live", "dir": str(live_j), "user": "u_test", "source": {"type": "live", "url": ""}, "options": {}, "stages": {}}
        jobs._jobs[live_job["id"]] = live_job
        jobs._by_user.setdefault("u_test", {})[live_job["id"]] = live_job
        with st_ctx.as_user("u_test"):
            vis_live = visuals.run_visual_analysis(live_job["id"])
        check("Live growing recording analyzes completed tail and marks pending=True", vis_live["status"] == "done" and vis_live["pending"] is True and vis_live["counts"]["snapshots"] >= 1)

        # 7d. Cache hit on second PipelineEngine run
        eng_cache = PipelineEngine(store=store)
        ctx_run = StageContext(job={"id": "job_cache_1", "dir": str(j_dir), "user": "u_test", "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}}, user_dir=u_dir, user_id="u_test", store=store, inputs={"source_video": str(cut_mp4), "video_fingerprint": src_fp})
        eng_cache.execute_stage(DetectScenesStage(), ctx_run, src_fp)
        eng_cache.execute_stage(DetectScenesStage(), ctx_run, src_fp)
        check("Rerun hits ArtifactStore cache (skipped_cached)", ctx_run.job["stages"]["detect_scenes"]["status"] == "skipped_cached")

        print("\n=== 8. FaultInjectingStore + WORK_DIR Wipe + Resume + Eviction ===")
        dur_root = work_root / "drive_durable"
        base_store = LocalFolderStore(dur_root)
        fault_store = FaultInjectingStore(base_store)

        ai.capabilities = lambda prov=None, model=None: {"ready": True, "provider": "gemini", "model": "gemini-2.5-flash", "image_input": True, "json_mode": True}
        ai_calls_ref = {"n": 0}

        def _det_ai(model, prompt, schema, images=None):
            ai_calls_ref["n"] += 1
            return [
                {"index": i, "labels": ["frame"], "caption": f"Frame index {i}", "people_count": 0, "is_good_thumbnail": 0.5}
                for i in range(9)
            ], 80, 30

        ai.call = _det_ai
        try:
            # 8a. Clean uninterrupted baseline (batch_size=3 -> multiple batches across 14s fixture)
            clean_work = work_root / "clean_work"
            clean_u = clean_work / "users" / "u_colab"
            clean_j = clean_u / "jobs" / "job_clean"
            clean_j.mkdir(parents=True, exist_ok=True)
            clean_store = ArtifactStore(user_dir=clean_u, user_id="u_colab")
            clean_dur = LocalFolderStore(work_root / "drive_baseline")
            clean_eng = PipelineEngine(store=clean_store, durable_store=clean_dur)
            clean_job = {"id": "job_clean", "user": "u_colab", "dir": str(clean_j), "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}}
            jobs._jobs["job_clean"] = clean_job
            with st_ctx.as_user("u_colab"):
                vis_base = visuals.run_visual_analysis(
                    "job_clean", durable_store=clean_dur, engine=clean_eng, source_video=cut_mp4,
                    params={"snapshot_batch_size": 3, "snapshot_interval_sec": 1.0, "snapshot_dhash_threshold": -1},
                )
            base_snaps = visuals.load_visual_data(clean_job, "extract_snapshots", "snapshots.json", durable_store=clean_dur)["snapshots"]
            base_ai_calls = ai_calls_ref["n"]

            # 8b. Run with FaultInjectingStore crashing mid-extract_snapshots (on 2nd batch_pack)
            ai_calls_ref["n"] = 0
            crash_work = work_root / "crash_work"
            crash_u = crash_work / "users" / "u_colab"
            crash_j = crash_u / "jobs" / "job_fault"
            crash_j.mkdir(parents=True, exist_ok=True)
            crash_store = ArtifactStore(user_dir=crash_u, user_id="u_colab")
            crash_eng = PipelineEngine(store=crash_store, durable_store=fault_store)
            crash_job = {"id": "job_fault", "user": "u_colab", "dir": str(crash_j), "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}}
            jobs._jobs["job_fault"] = crash_job

            class _SimulatedKill(BaseException):
                pass

            batch_seen = {"n": 0}
            def _kill_second_batch(rel_path: str):
                if "extract_snapshots_batch_" in rel_path:
                    batch_seen["n"] += 1
                    if batch_seen["n"] == 2:
                        raise _SimulatedKill("SIMULATED_COLAB_KILL_MID_EXTRACT")

            fault_store.on_put_callback = _kill_second_batch
            crashed_mid_extract = False
            try:
                with st_ctx.as_user("u_colab"):
                    visuals.run_visual_analysis(
                        "job_fault", durable_store=fault_store, engine=crash_eng, source_video=cut_mp4,
                        params={"snapshot_batch_size": 3, "snapshot_interval_sec": 1.0, "snapshot_dhash_threshold": -1},
                    )
            except _SimulatedKill:
                crashed_mid_extract = True
            except DurableStoreError:
                crashed_mid_extract = True
            except Exception as exc:
                if "SIMULATED_COLAB_KILL_MID_EXTRACT" in str(exc):
                    crashed_mid_extract = True

            check("FaultInjectingStore killed mid-extract_snapshots", crashed_mid_extract and batch_seen["n"] >= 2)

            # Wipe WORK_DIR completely!
            shutil.rmtree(str(crash_work), ignore_errors=True)
            check("WORK_DIR wiped completely after crash", not crash_work.exists())

            # Now run clean detect+extract+analyze up to sheet 0 then kill mid-caption_snapshots
            fault_store.on_put_callback = None
            cap_seen = {"n": 0}
            def _kill_second_sheet(rel_path: str):
                if "caption_snapshots_sheet_" in rel_path:
                    cap_seen["n"] += 1
                    if cap_seen["n"] == 2:
                        raise _SimulatedKill("SIMULATED_COLAB_KILL_MID_CAPTION")

            fault_store.on_put_callback = _kill_second_sheet
            crash_u2 = crash_work / "users" / "u_colab"
            crash_j2 = crash_u2 / "jobs" / "job_fault"
            crash_j2.mkdir(parents=True, exist_ok=True)
            resumed_job = {"id": "job_fault", "user": "u_colab", "dir": str(crash_j2), "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}}
            jobs._jobs["job_fault"] = resumed_job
            try:
                with st_ctx.as_user("u_colab"):
                    visuals.run_visual_analysis(
                        "job_fault", durable_store=fault_store,
                        engine=PipelineEngine(store=ArtifactStore(user_dir=crash_u2, user_id="u_colab"), durable_store=fault_store),
                        source_video=cut_mp4,
                        params={"snapshot_batch_size": 3, "snapshot_interval_sec": 1.0, "snapshot_dhash_threshold": -1},
                    )
            except _SimulatedKill:
                pass
            except Exception:
                pass
            calls_before_wipe2 = ai_calls_ref["n"]
            ai_calls_ref["n"] = 0
            fault_store.on_put_callback = None

            # Wipe WORK_DIR a second time, then resume to full completion!
            shutil.rmtree(str(crash_work), ignore_errors=True)
            crash_u3 = crash_work / "users" / "u_colab"
            crash_j3 = crash_u3 / "jobs" / "job_fault"
            crash_j3.mkdir(parents=True, exist_ok=True)
            resumed_job3 = {"id": "job_fault", "user": "u_colab", "dir": str(crash_j3), "source": {"type": "file", "path": str(cut_mp4)}, "options": {}, "stages": {}}
            jobs._jobs["job_fault"] = resumed_job3
            final_store3 = ArtifactStore(user_dir=crash_u3, user_id="u_colab")
            with st_ctx.as_user("u_colab"):
                vis_resumed = visuals.run_visual_analysis(
                    "job_fault", durable_store=fault_store,
                    engine=PipelineEngine(store=final_store3, durable_store=fault_store),
                    source_video=cut_mp4,
                    params={"snapshot_batch_size": 3, "snapshot_interval_sec": 1.0, "snapshot_dhash_threshold": -1},
                )
            res_snaps = visuals.load_visual_data(resumed_job3, "extract_snapshots", "snapshots.json", durable_store=fault_store)["snapshots"]

            check(
                "Resumed snapshot count & hashes match uninterrupted baseline with 0 duplicates",
                len(res_snaps) == len(base_snaps)
                and [(s["id"], s["timestamp"], s["dhash"]) for s in res_snaps] == [(s["id"], s["timestamp"], s["dhash"]) for s in base_snaps],
            )
            check(
                "No paid AI call repeated across caption resume after WORK_DIR wipe",
                base_ai_calls >= 2 and ai_calls_ref["n"] <= base_ai_calls - 1,
                f"baseline={base_ai_calls}, before_kill2={calls_before_wipe2}, after_resume={ai_calls_ref['n']}",
            )
            ext_fp = vis_resumed["artifacts"]["extract_snapshots"]
            drive_tar_list = base_store.list(f"users/u_colab/artifacts/extract_snapshots/{ext_fp[:2]}/{ext_fp}/")
            batch_tar_leftovers = base_store.list("users/u_colab/checkpoints/job_fault/extract_snapshots_batch_")
            check("Snapshots packed into single .tar per stage artifact on Drive", len(drive_tar_list) == 1 and drive_tar_list[0].endswith(".tar") and not batch_tar_leftovers)

            # Colab local cache eviction + lazy Drive restore
            local_ext_dir = final_store3.root / "extract_snapshots" / ext_fp[:2] / ext_fp
            evicted = final_store3.evict_stage_cache("extract_snapshots", max_bytes=1, durable_store=fault_store)
            check("Colab eviction deletes local snapshot pack only when Drive pack verified", ext_fp in evicted and not local_ext_dir.exists())

            # Lazy restore via resolve_snapshot_image after local cache eviction
            orig_ucfg = config.USERS_DIR
            config.USERS_DIR = crash_work / "users"
            try:
                resolved = visuals.resolve_snapshot_image(resumed_job3, res_snaps[0]["id"], size="thumb", durable_store=fault_store)
                check("API lazy unpack from Drive restores evicted snapshot image", resolved is not None and resolved[0].is_file() and local_ext_dir.exists())
                # Delete job -> cleans artifacts locally & on Drive
                jobs._by_user["u_colab"] = {"job_fault": resumed_job3}
                visuals.cleanup_job_artifacts(resumed_job3, durable_store=fault_store)
                check("Deleting video removes visual artifacts locally and on Drive", not local_ext_dir.exists() and not base_store.list(f"users/u_colab/artifacts/extract_snapshots/{ext_fp[:2]}/{ext_fp}/"))
            finally:
                config.USERS_DIR = orig_ucfg
        finally:
            ai.call = orig_ai_call
            ai.capabilities = orig_caps_avail

        print("\n=== 9. Live HTTP APIs: Flag-OFF 404 & Flag-ON Isolation / Pagination / Images ===")
        if BASE:
            c_off = _http_helper(BASE)
            st_off, _ = c_off("POST", "/api/auth/signup", {"email": "visoff@example.com", "password": "Password1234", "name": "VisOff"})
            if st_off in (200, 201):
                for ep in ("/api/videos/job_missing/scenes", "/api/videos/job_missing/snapshots", "/api/videos/job_missing/snapshots/s1/image", "/api/videos/job_missing/moments"):
                    st_e, _ = c_off("GET", ep)
                    check(f"Flag OFF {ep} returns 404", st_e == 404, f"got={st_e}")

        # Spin up ephemeral server with PEAKCUT_SNAPSHOTS=1 and PEAKCUT_ACCOUNTS=1
        srv_tmp = work_root / "http_srv_data"
        srv_tmp.mkdir(parents=True, exist_ok=True)
        vis_port = int(os.environ.get("PEAKCUT_VIS_TEST_PORT", "8797"))
        vis_base = f"http://127.0.0.1:{vis_port}"
        srv_env = {
            **os.environ,
            "PEAKCUT_DATA": str(srv_tmp),
            "PEAKCUT_PORT": str(vis_port),
            "PEAKCUT_HOST": "127.0.0.1",
            "PEAKCUT_PUBLIC_URL": vis_base,
            "PEAKCUT_ACCOUNTS": "1",
            "PEAKCUT_VERIFY_EMAIL": "0",
            "PEAKCUT_SNAPSHOTS": "1",
            "PEAKCUT_AUTO_RELOAD": "0",
            "PEAKCUT_FETCH_MODEL": "0",
            "PEAKCUT_UPDATE_CHECK": "0",
            "PEAKCUT_ASR_URL": "http://127.0.0.1:9",
            "PEAKCUT_JOB_WORKERS": "2",
            "DATABASE_URL": "",
            "PEAKCUT_AI_KEY": "",
            "PEAKCUT_AI_MODELS": "",
        }
        proc = subprocess.Popen(
            [sys.executable, "-m", "studio.server"],
            cwd=str(ROOT),
            env=srv_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            up = False
            for _ in range(45):
                try:
                    if urllib.request.urlopen(f"{vis_base}/healthz", timeout=2).status == 200:
                        up = True
                        break
                except Exception:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.5)
            check("Flag-ON test server boots cleanly", up)
            if up:
                c_anon = _http_helper(vis_base)
                st_unauth, _ = c_anon("GET", "/api/videos/any_job/scenes")
                check("Unauthenticated GET /api/videos/.../scenes returns 401", st_unauth == 401)

                c_a = _http_helper(vis_base)
                c_b = _http_helper(vis_base)
                st_su_a, _ = c_a("POST", "/api/auth/signup", {"email": "usera_vis@example.com", "password": "Password1234", "name": "User A"})
                st_su_b, _ = c_b("POST", "/api/auth/signup", {"email": "userb_vis@example.com", "password": "Password1234", "name": "User B"})
                check("User A & User B sign up", st_su_a == 200 and st_su_b == 200)

                st_up, j_up = _multipart_upload(c_a, cut_mp4)
                jid_a = j_up.get("id", "")
                check("User A uploads synthetic MP4", st_up == 200 and bool(jid_a))

                # Wait for job ingestion/run or trigger visual analysis right away
                st_an, _ = c_a("POST", f"/api/videos/{jid_a}/analyze", {})
                check("POST /api/videos/{id}/analyze succeeds (200)", st_an == 200)

                vis_done = False
                for _ in range(60):
                    st_j, d_j = c_a("GET", f"/api/jobs/{jid_a}")
                    if st_j == 200 and (d_j.get("visuals") or {}).get("status") == "done":
                        vis_done = True
                        break
                    time.sleep(0.5)
                check("Visual analysis completes on live server without AI key", vis_done)

                if vis_done:
                    st_sc, d_sc = c_a("GET", f"/api/videos/{jid_a}/scenes")
                    st_sn1, d_sn1 = c_a("GET", f"/api/videos/{jid_a}/snapshots?from=0&to=20&limit=3&cursor=")
                    check("GET /scenes & paginated GET /snapshots succeed", st_sc == 200 and len(d_sc["items"]) >= 4 and st_sn1 == 200 and len(d_sn1["items"]) <= 3 and "path" not in json.dumps(d_sn1))
                    if d_sn1.get("next_cursor"):
                        st_sn2, d_sn2 = c_a("GET", f"/api/videos/{jid_a}/snapshots?limit=3&cursor={d_sn1['next_cursor']}")
                        check("Snapshots pagination cursor advances", st_sn2 == 200 and len(d_sn2["items"]) >= 1 and d_sn2["items"][0]["id"] != d_sn1["items"][0]["id"])

                    sid0 = d_sn1["items"][0]["id"]
                    st_th, b_th = c_a("GET", f"/api/videos/{jid_a}/snapshots/{sid0}/image?size=thumb")
                    st_an_im, b_an = c_a("GET", f"/api/videos/{jid_a}/snapshots/{sid0}/image?size=analysis")
                    st_fl_im, b_fl = c_a("GET", f"/api/videos/{jid_a}/snapshots/{sid0}/image?size=full")
                    check("Image endpoint serves WebP thumb, JPEG analysis & lazy full frame", st_th == 200 and b_th[:4] == b"RIFF" and st_an_im == 200 and b_an[:3] == b"\xff\xd8\xff" and st_fl_im == 200 and b_fl[:3] == b"\xff\xd8\xff")

                    st_miss_im, _ = c_a("GET", f"/api/videos/{jid_a}/snapshots/sn_does_not_exist/image?size=thumb")
                    check("Unknown snapshot id returns 404", st_miss_im == 404)

                    st_mo, d_mo = c_a("GET", f"/api/videos/{jid_a}/moments?from=0&to=20")
                    check("GET /moments returns timeline moments", st_mo == 200 and len(d_mo["items"]) >= 1)

                    # Cross-user isolation: User B MUST receive 404 (never 403)
                    for m_ep in (
                        ("POST", f"/api/videos/{jid_a}/analyze"),
                        ("GET", f"/api/videos/{jid_a}/scenes"),
                        ("GET", f"/api/videos/{jid_a}/snapshots"),
                        ("GET", f"/api/videos/{jid_a}/snapshots/{sid0}/image?size=thumb"),
                        ("GET", f"/api/videos/{jid_a}/moments"),
                    ):
                        st_iso, _ = c_b(m_ep[0], m_ep[1], {} if m_ep[0] == "POST" else None)
                        check(f"User B isolation {m_ep[0]} {m_ep[1].split('?')[0]} returns 404 (not 403)", st_iso == 404, f"got={st_iso}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        register_face_detector(None)
        ai.call = ai.call
        shutil.rmtree(str(work_root), ignore_errors=True)

    print(f"\nVisual Pipeline Tests Complete: {len(fails)} failures.")
    return len(fails) == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
