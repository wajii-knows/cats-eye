"""Tests for Full-Video-First Global Viral Selection, Multimodal Scoring V2, Mid-Side Facecam, and Render V2."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-fullvid-")
os.environ["DATABASE_URL"] = ""
os.environ["PEAKCUT_AI_KEY"] = ""
os.environ["PEAKCUT_FETCH_MODEL"] = "0"
os.environ["PEAKCUT_MULTISHOT"] = "0"
os.environ["PEAKCUT_ASSETS"] = "0"
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from studio import brain, config, ctx, db, flags, jobs, media, render, transcribe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    if not ok:
        fails.append(name)


def make_words(text, start=0.0, step=0.42):
    out, t = [], start
    for token in text.split():
        dur = max(0.16, step * 0.78)
        out.append({"w": token, "s": round(t, 2), "e": round(t + dur, 2)})
        t += step
    return out


work = Path(tempfile.mkdtemp(prefix="pc-fv-work-"))
try:
    db.init()
    config.FONTS_DIR.mkdir(parents=True, exist_ok=True)
    render.install_fonts()

    # ---- 1. Real-video streaming frame-shot scanner (media.scan_frame_metrics)
    scan_src = work / "scan_test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=0x101820:size=320x180:duration=3:rate=24",
            "-f", "lavfi", "-i", "testsrc=size=320x180:duration=3:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "aac", str(scan_src),
        ],
        check=True,
    )
    f_metrics = media.scan_frame_metrics(scan_src, interval_sec=1.0, max_frames=20)
    static_m = [m["motion"] for m in f_metrics if 0.5 <= m["time"] <= 2.5]
    active_m = [m["motion"] for m in f_metrics if m["time"] >= 3.0]
    check(
        "media.scan_frame_metrics detects static vs active motion and scene boundary",
        len(f_metrics) >= 5
        and bool(static_m) and max(static_m) < 0.02
        and bool(active_m) and max(active_m) >= 0.14
        and any(m["is_scene_boundary"] for m in f_metrics if m["time"] >= 3.0),
        str(f_metrics),
    )
    canceled_m = media.scan_frame_metrics(scan_src, interval_sec=0.5, should_cancel=lambda: True)
    missing_m = media.scan_frame_metrics(work / "missing.mp4")
    check("media.scan_frame_metrics stops cleanly on cancel or missing file", canceled_m == [] and missing_m == [])

    # ---- 2. Static frozen screen penalty, Clip 1 calibration, non-verbal micro-clips, profanity & _auto_hook
    clip1_words = make_words(
        "Homies! Where are you? We gotta ask the chat what do we need to do right now in the sub goal!",
        start=0.1, step=0.45,
    )
    z_loud = np.zeros(26, dtype=np.float32)
    z_loud[0] = 2.8
    z_loud[15] = 2.6
    sc_clip1 = brain.score_clip(0.0, 22.0, words=clip1_words, db_z=z_loud)
    check(
        "Clip 1 live-chat filler cannot reach 1M+ Strangers or inflated 86 score",
        sc_clip1["score"] < 75 and sc_clip1["audience_reach"] != "1M+ Strangers",
        str(sc_clip1),
    )

    punchy_words = make_words(
        "WAIT look at these GTA 6 physics right there?! No way that car actually exploded like real life!",
        start=0.1, step=0.44,
    )
    frozen_snaps = [
        {"timestamp": float(t), "time": float(t), "motion": 0.002, "edge_density": 0.25, "blurriness": 0.05, "is_scene_boundary": False, "faces": 1}
        for t in (0, 3, 6, 9, 12, 15, 18)
    ]
    dynamic_snaps = [
        {"timestamp": float(t), "time": float(t), "motion": 0.28, "edge_density": 0.22, "blurriness": 0.05, "is_scene_boundary": (t in (0, 6, 15)), "faces": 1}
        for t in (0, 3, 6, 9, 12, 15, 18)
    ]
    sc_frozen = brain.score_clip(0.0, 19.0, words=punchy_words, db_z=z_loud, visual_snaps=frozen_snaps)
    sc_dyn = brain.score_clip(0.0, 19.0, words=punchy_words, db_z=z_loud, visual_snaps=dynamic_snaps)
    check(
        "Frozen static menu screen scores sharply below dynamic visual clip and blocks 1M+ Strangers",
        sc_dyn["score"] - sc_frozen["score"] >= 20
        and sc_frozen["score"] <= 55
        and sc_frozen["audience_reach"] != "1M+ Strangers"
        and sc_dyn["score"] >= 85,
        f"frozen={sc_frozen} | dyn={sc_dyn}",
    )

    # Non-verbal action micro-clip (0 spoken words, visual + acoustic spike)
    db_nv = np.full(40, -32.0, dtype=np.float32)
    db_nv[14:17] = -4.0
    nv_snaps = [{"timestamp": 14.0, "time": 14.0, "motion": 0.29, "edge_density": 0.24, "blurriness": 0.08, "is_scene_boundary": True}]
    orig_ask = brain.ask_ai
    brain.ask_ai = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("No AI"))
    try:
        nv_clips, _ = brain.find_clips([], db_nv, 40.0, max_clips=2, visual_metrics=nv_snaps)
    finally:
        brain.ask_ai = orig_ask
    check(
        "Non-verbal 0-word gameplay action clip is found in 8-15s range instead of returning []",
        len(nv_clips) == 1 and 7.5 <= (nv_clips[0]["end"] - nv_clips[0]["start"]) <= 15.0,
        str(nv_clips),
    )

    # _auto_hook deduplication + profanity censoring
    stutter_w = make_words("Hold on. Hold on, hold", start=0.1, step=0.4)
    hk_stutter, _ = brain._auto_hook(stutter_w, 0.0, 10.0)
    follow_w = make_words("Hold on. Hold on look at that insane explosion", start=0.1, step=0.4)
    hk_follow, _ = brain._auto_hook(follow_w, 0.0, 10.0)
    check(
        "_auto_hook collapses stutter repeats and skips leading filler before action words",
        hk_stutter == "HOLD ON" and hk_follow == "LOOK AT THAT INSANE EXPLOSION",
        f"stutter={hk_stutter!r}, follow={hk_follow!r}",
    )
    cens = brain.censor_profanity("WHAT THE FUCK BRO THAT SHIT WAS WILD yaar bhai")
    check(
        "censor_profanity masks explicit profanity while preserving Hinglish and normal speech",
        "FUCK" not in cens and "SHIT" not in cens and "F***" in cens and "S***" in cens and "yaar bhai" in cens,
        cens,
    )

    # ---- 3. Full-video-first global viral selection (NOT clip as you go!) in studio/jobs.py
    from studio.pipeline.stage import StageContext  # noqa: E402
    from studio.pipeline.artifacts import ArtifactStore  # noqa: E402

    fv_dir = Path(tempfile.mkdtemp(dir=config.DATA_DIR, prefix="job-fv-"))
    (fv_dir / "clips").mkdir(parents=True, exist_ok=True)
    job = {
        "id": "jobfvtest01",
        "user": "u_fv",
        "creator": None,
        "kind": "video",
        "dir": str(fv_dir),
        "title": "GTA 6 Full Stream Test",
        "uploader": "Streamer",
        "duration": 1500.0,  # 25 minutes -> 3 progressive blocks
        "width": 1280,
        "height": 720,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "source": {"type": "file", "path": str(scan_src), "name": "stream.mp4"},
        "options": {"max_clips": 2, "min_len": 15, "max_len": 35, "style": "gta_viral", "layout": "blur", "quality": 720, "range_start": None, "range_end": None},
        "clips": [],
        "ai": {},
        "done_blocks": 0,
    }
    with jobs._lock:
        jobs._jobs[job["id"]] = job

    orig_probe = media.probe
    orig_ext_aud = media.extract_audio
    orig_load_aud = media.load_audio
    orig_loud = media.loudness_per_second
    orig_scan = media.scan_frame_metrics
    orig_thumb = media.save_thumb
    orig_trans = transcribe.transcribe
    orig_mready = transcribe.model_ready
    orig_find = brain.find_clips
    orig_render_one = jobs.render_one

    events_log = []
    block_words = {
        0: make_words("hey guys welcome back so anyway today we look slowly at nothing much at all right now", start=10.0, step=1.2),
        1: make_words("maybe we should drive over to the beach soon and check out the map later today", start=10.0, step=0.9),
        2: make_words("WAIT did you just see those physics?! No way he clutched that one HP getaway before GTA 6!", start=10.0, step=0.42),
    }
    call_idx = {"t": 0}

    def fake_probe(p):
        return {"width": 1280, "height": 720, "duration": 1500.0}

    def fake_extract_audio(s, wav):
        Path(wav).write_bytes(b"")

    def fake_load_audio(wav):
        return np.zeros(1500 * 16000, dtype=np.int16)

    def fake_loud(arr):
        db_arr = np.full(1500, -30.0, dtype=np.float32)
        db_arr[1210:1228] = -4.0  # strongest burst is in Block 2 (t=1210..1228)!
        return db_arr

    def fake_scan(vp, **kw):
        events_log.append(("scan", len(events_log)))
        return [
            {"timestamp": 20.0, "time": 20.0, "motion": 0.002, "edge_density": 0.1, "blurriness": 0.1, "is_scene_boundary": False, "faces": None},
            {"timestamp": 1215.0, "time": 1215.0, "motion": 0.30, "edge_density": 0.25, "blurriness": 0.05, "is_scene_boundary": True, "faces": 1},
        ]

    def fake_trans(aud_slice, cb=None):
        idx = call_idx["t"]
        call_idx["t"] += 1
        events_log.append(("transcribe", idx))
        return list(block_words.get(idx, []))

    find_calls = []

    def spy_find(w_all, db_all, span_all, *a, **kw):
        events_log.append(("find_clips", len(w_all)))
        find_calls.append((len(w_all), float(span_all), kw.get("visual_metrics")))
        return [
            {
                "start": 1210.0, "end": 1228.0, "score": 92,
                "scores": {"hook": 90, "pacing": 90, "payoff": 92, "shareability": 88, "focus": 90, "clarity": 88},
                "tier": "viral", "audience_reach": "1M+ Strangers", "scroll_stopper": "WAIT physics?!",
                "share_trigger": "Wait, what?!", "loop_seamless": True, "category": "clutch",
                "hook": "THESE PHYSICS ARE UNREAL", "title": "Block 2 Viral Peak", "description": "",
                "hashtags": ["gta6"], "highlight_words": ["physics"], "reason": "Global top peak in block 2", "shots": [],
            },
            {
                "start": 610.0, "end": 628.0, "score": 76,
                "scores": {"hook": 74, "pacing": 78, "payoff": 75, "shareability": 72, "focus": 80, "clarity": 74},
                "tier": "strong", "audience_reach": "100K+ Audience", "scroll_stopper": "Beach drive",
                "share_trigger": "High energy", "loop_seamless": False, "category": "other",
                "hook": "BLOCK 1 MOMENT", "title": "Block 1 Clip", "description": "",
                "hashtags": ["gta6"], "highlight_words": [], "reason": "Second best", "shots": [],
            },
        ], {"model": "mock", "tokens": 50, "note": ""}

    def fake_render_one(j, clip, words_in):
        events_log.append(("render", clip["id"], clip["start"]))
        clip["status"] = "done"
        clip["version"] = clip.get("version", 0) + 1

    try:
        media.probe = fake_probe
        media.extract_audio = fake_extract_audio
        media.load_audio = fake_load_audio
        media.loudness_per_second = fake_loud
        media.scan_frame_metrics = fake_scan
        media.save_thumb = lambda *a, **k: False
        transcribe.transcribe = fake_trans
        transcribe.model_ready = lambda: True
        brain.find_clips = spy_find
        jobs.render_one = fake_render_one

        sctx_fv = StageContext(job=job, user_dir=fv_dir.parent.parent, user_id="u_fv", store=ArtifactStore(user_dir=fv_dir.parent.parent, user_id="u_fv"))
        jobs._run_job_body(job, fv_dir, job["options"], job["source"], 0.0, [], sctx_fv)

        trans_indices = [i for i, ev in enumerate(events_log) if ev[0] == "transcribe"]
        scan_indices = [i for i, ev in enumerate(events_log) if ev[0] == "scan"]
        find_indices = [i for i, ev in enumerate(events_log) if ev[0] == "find_clips"]
        render_indices = [i for i, ev in enumerate(events_log) if ev[0] == "render"]
        check(
            "Full-video-first pipeline transcribes ALL 3 blocks & checks ALL frames BEFORE calling find_clips ONCE and rendering",
            len(find_calls) == 1
            and len(trans_indices) == 3
            and len(scan_indices) == 1
            and max(trans_indices) < min(scan_indices) < min(find_indices) < min(render_indices)
            and job["clips"][0]["start"] == 1210.0
            and job["clips"][0]["id"] == "clip01"
            and (fv_dir / "frame_metrics.json").is_file(),
            f"events={events_log} clips={job['clips']}",
        )

        # Resume after selection: skips transcribe, scan, and find_clips; renders only unfinished clip
        events_log.clear()
        job["resume"] = True
        job["clips"][0]["status"] = "queued"
        job["clips"][1]["status"] = "done"
        jobs._run_job_body(job, fv_dir, job["options"], job["source"], 0.0, [], sctx_fv)
        check(
            "Resume after global selection skips transcribe/scan/find_clips and only renders unfinished clip01",
            events_log == [("render", "clip01", 1210.0)],
            str(events_log),
        )
    finally:
        media.probe = orig_probe
        media.extract_audio = orig_ext_aud
        media.load_audio = orig_load_aud
        media.loudness_per_second = orig_loud
        media.scan_frame_metrics = orig_scan
        media.save_thumb = orig_thumb
        transcribe.transcribe = orig_trans
        transcribe.model_ready = orig_mready
        brain.find_clips = orig_find
        jobs.render_one = orig_render_one
        with jobs._lock:
            jobs._jobs.pop(job["id"], None)

    # ---- 4. Mid-Left & Mid-Right auto facecam detection
    def _make_mid_frame(key: str) -> np.ndarray:
        fh, fw = 180, 320
        frame = np.zeros((fh, fw, 3), dtype=np.uint8)
        frame[:, :] = (35, 85, 135)
        cx, cy, cw, ch = render.FACECAM_CORNERS[key]
        x0, x1 = int(round(cx * fw)), int(round((cx + cw) * fw))
        y0, y1 = int(round(cy * fh)), int(round((cy + ch) * fh))
        frame[y0:y1, x0:x1] = (24, 24, 30)
        if cx < 0.5:
            frame[y0:y1, max(0, x1 - 2):min(fw, x1 + 1)] = (230, 180, 70)
        else:
            frame[y0:y1, max(0, x0 - 1):min(fw, x0 + 2)] = (230, 180, 70)
        frame[max(0, y0 - 1):min(fh, y0 + 2), x0:x1] = (230, 180, 70)
        frame[max(0, y1 - 2):min(fh, y1 + 1), x0:x1] = (230, 180, 70)
        pw, ph = x1 - x0, y1 - y0
        frame[y0 + int(ph * 0.22):y0 + int(ph * 0.80), x0 + int(pw * 0.25):x0 + int(pw * 0.75)] = (212, 152, 122)
        return frame

    fr_ml = render.analyze_framing_from_frames([_make_mid_frame("cam_ml"), _make_mid_frame("cam_ml")])
    fr_mr = render.analyze_framing_from_frames([_make_mid_frame("cam_mr"), _make_mid_frame("cam_mr")])
    check(
        "analyze_framing_from_frames auto-detects Mid-Left (cam_ml) & Mid-Right (cam_mr) streamer webcams",
        fr_ml["mode"] == "facecam" and fr_ml["facecam_box"] is not None
        and fr_ml["facecam_box"][0] < 0.05 and 0.25 <= fr_ml["facecam_box"][1] <= 0.45
        and fr_mr["mode"] == "facecam" and fr_mr["facecam_box"] is not None
        and fr_mr["facecam_box"][0] > 0.65 and 0.25 <= fr_mr["facecam_box"][1] <= 0.45,
        f"ml={fr_ml} mr={fr_mr}",
    )

    # ---- 5. Render V2: gta_viral, vice_neon, sidechain BGM ducking, and 50/50 comparison split
    vid_10s = work / "comp_src.mp4"
    bgm_wav = work / "bgm.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:duration=10:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-c:v", "libx264", "-c:a", "aac", str(vid_10s)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=110:duration=3", str(bgm_wav)],
        check=True,
    )
    cap_words = make_words("what the fuck look at this detail", start=0.2, step=0.32)

    flags.override("PEAKCUT_ASSETS", True)
    try:
        res_gta = render.render_clip(
            vid_10s, work / "out", "gta_style",
            {"start": 0.0, "end": 2.5, "layout": "blur", "style": "gta_viral", "hook": "NEVER DO THIS FUCK", "bgm_path": str(bgm_wav)},
            cap_words, 1280, 720,
        )
        ass_gta = (work / "out" / "gta_style.ass").read_text(encoding="utf-8")
        meta_gta = media.probe(work / "out" / "gta_style.mp4")
        check(
            "gta_viral + sidechain BGM ducking renders 1080x1920 MP4 with custom yellow hook pill and censored profanity",
            meta_gta["width"] == 1080 and meta_gta["height"] == 1920
            and "\\c&H00EAFF&" in ass_gta and "&H000A0A0A" in ass_gta
            and "FUCK" not in ass_gta and "F***" in ass_gta,
            ass_gta[:400],
        )

        res_comp = render.render_clip(
            vid_10s, work / "out", "ab_comp",
            {"start": 0.0, "end": 2.5, "compare_start": 5.0, "compare_end": 7.5,
             "style": "vice_neon", "hook": "2013 VS 2026 GRAPHICS",
             "top_label": "GTA 5 (2013)", "bottom_label": "GTA 6 (2026)"},
            cap_words, 1280, 720,
        )
        ass_comp = (work / "out" / "ab_comp.ass").read_text(encoding="utf-8")
        meta_comp = media.probe(work / "out" / "ab_comp.mp4")
        check(
            "Comparison split (2013 vs 2026) + vice_neon renders 1080x1920 MP4 with top/bottom pill labels",
            meta_comp["width"] == 1080 and meta_comp["height"] == 1920
            and "GTA 5 (2013)" in ass_comp and "GTA 6 (2026)" in ass_comp
            and "&H00B469FF" in ass_comp and "\\c&H140F1E&" in ass_comp,
            ass_comp[:400],
        )
    finally:
        flags.reset_overrides()

    check("web/index.html CAP_LOOK includes gta_viral and vice_neon previews",
          "gta_viral" in (ROOT / "web" / "index.html").read_text(encoding="utf-8")
          and "vice_neon" in (ROOT / "web" / "index.html").read_text(encoding="utf-8"))

    print("ALL PASSED" if not fails else f"{len(fails)} failed")
finally:
    db.engine.dispose()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

sys.exit(1 if fails else 0)
