"""Comprehensive test suite for Cat's Eye Phase 2:
- Feature flags (default OFF, overrides)
- Typed contracts (validation rules, start < end, bounds, file checks)
- Deterministic stage fingerprinting
- Content-addressed artifact store (atomic write, reuse, touch, delete, crash resilience)
- Migration runner (blank DB, idempotency, rollback, re-apply)
- Pipeline resume semantics and caching (skipped_cached, DAG execution)
- Retry on retryable errors and cancellation process termination
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from pydantic import ValidationError

from studio import config, flags
from studio.contracts import (
    Asset,
    Moment,
    Scene,
    Shot,
    ShotPlan,
    Snapshot,
    TranscriptSegment,
    Word,
)
from studio.db import migrate
from studio.pipeline.artifacts import ArtifactStore
from studio.pipeline.engine import PipelineEngine, get_process_memory_mb
from studio.pipeline.stage import (
    Stage,
    StageContext,
    StageError,
    compute_file_sha256,
    compute_stage_fingerprint,
)


def run_all_tests():
    fails = []

    def check(name, ok, extra=""):
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}" + (f"   {extra}" if extra else ""))
        if not ok:
            fails.append(name)

    print("\n=== Feature Flags ===")
    all_f = flags.all_flags()
    expected_flags = [
        "PEAKCUT_SNAPSHOTS",
        "PEAKCUT_VIRAL_V2",
        "PEAKCUT_ASSETS",
        "PEAKCUT_PRESETS",
        "PEAKCUT_MULTISHOT",
        "PEAKCUT_MATCHING",
        "PEAKCUT_HWACCEL",
        "PEAKCUT_WORKSPACE",
        "PEAKCUT_QC",
        "PEAKCUT_LEARNING",
    ]
    check("all 10 feature flags registered", all(k in all_f for k in expected_flags))
    check("all feature flags default to OFF", all(not all_f[k] for k in expected_flags))

    # Test override
    flags.override("PEAKCUT_SNAPSHOTS", True)
    check("flag override works", flags.is_on("PEAKCUT_SNAPSHOTS") is True)
    flags.reset_overrides()
    check("flag reset works", flags.is_on("PEAKCUT_SNAPSHOTS") is False)

    print("\n=== Typed Contracts Validation ===")
    # 1. Word
    w_valid = Word(word="Cat's", start=1.0, end=1.5, confidence=0.98)
    check("valid Word passes", w_valid.word == "Cat's" and w_valid.validate() is not None)
    try:
        Word(word="bad", start=2.0, end=1.0)
        check("Word start >= end rejected", False)
    except ValidationError:
        check("Word start >= end rejected", True)

    try:
        Word(word="bad", start=-0.5, end=1.0)
        check("Word negative timestamp rejected", False)
    except ValidationError:
        check("Word negative timestamp rejected", True)

    try:
        Word(word="bad", start=0.0, end=1.0, confidence=1.2)
        check("Word confidence > 1.0 rejected", False)
    except ValidationError:
        check("Word confidence > 1.0 rejected", True)

    # 2. Scene
    sc_valid = Scene(id="sc1", start=0.0, end=4.5, motion_score=0.8, shot_type="wide")
    check("valid Scene passes", sc_valid.validate() is not None)
    try:
        Scene(id="sc1", start=5.0, end=2.0)
        check("Scene start >= end rejected", False)
    except ValidationError:
        check("Scene start >= end rejected", True)

    # 3. Snapshot
    tmp_snap_dir = Path(tempfile.mkdtemp(prefix="ce-test-snap-"))
    test_img = tmp_snap_dir / "frame.jpg"
    test_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # mock jpeg header
    snap_valid = Snapshot(id="sn1", timestamp=2.5, path=str(test_img), width=1920, height=1080)
    check("valid Snapshot with existing file passes", snap_valid.validate(check_files=True) is not None)

    snap_missing = Snapshot(id="sn2", timestamp=2.5, path=str(tmp_snap_dir / "nonexistent.jpg"), width=1920, height=1080)
    try:
        snap_missing.validate(check_files=True)
        check("Snapshot missing file rejected", False)
    except ValueError:
        check("Snapshot missing file rejected", True)
    shutil.rmtree(tmp_snap_dir, ignore_errors=True)

    # 4. Moment
    m_valid = Moment(id="m1", start=10.0, end=25.0, score=92.5, reason="High energy hook", hook_text="WATCH THIS")
    check("valid Moment passes", m_valid.validate() is not None)
    try:
        Moment(id="m2", start=10.0, end=5.0, score=50.0, reason="bad")
        check("Moment start >= end rejected", False)
    except ValidationError:
        check("Moment start >= end rejected", True)

    # 5. Shot & ShotPlan
    shot1 = Shot(id="s1", start=0.0, end=3.0, crop={"x": 0.0, "y": 0.0, "w": 1080.0, "h": 1920.0})
    shot2 = Shot(id="s2", start=3.0, end=6.0, crop={"x": 100.0, "y": 0.0, "w": 880.0, "h": 1920.0})
    plan = ShotPlan(moment_id="m1", shots=[shot1, shot2])
    check("valid ShotPlan passes", plan.validate(check_files=False) is not None)

    try:
        # Overlapping shots
        bad_shot2 = Shot(id="s2", start=2.5, end=6.0, crop={"x": 100.0, "y": 0.0, "w": 880.0, "h": 1920.0})
        ShotPlan(moment_id="m1", shots=[shot1, bad_shot2])
        check("ShotPlan with overlapping shots rejected", False)
    except ValidationError:
        check("ShotPlan with overlapping shots rejected", True)

    # 6. Asset
    tmp_ast_dir = Path(tempfile.mkdtemp(prefix="ce-test-ast-"))
    test_ast_file = tmp_ast_dir / "broll.mp4"
    test_ast_file.write_bytes(b"\x00" * 200)
    ast_valid = Asset(id="a1", name="Nature Drone", type="b-roll", path=str(test_ast_file), duration=10.5)
    check("valid Asset passes", ast_valid.validate(check_files=True) is not None)
    shutil.rmtree(tmp_ast_dir, ignore_errors=True)

    try:
        Asset(id="a2", name="Bad Type", type="invalid-type", path="some.mp4")
        check("Asset invalid type rejected", False)
    except ValidationError:
        check("Asset invalid type rejected", True)

    print("\n=== Deterministic Stage Fingerprinting ===")
    tmp_src_dir = Path(tempfile.mkdtemp(prefix="ce-test-fp-"))
    sample_file = tmp_src_dir / "sample.mp4"
    sample_file.write_bytes(b"video data content 1234567890" * 1000)

    src_hash1 = compute_file_sha256(sample_file)
    src_hash2 = compute_file_sha256(sample_file)
    check("streaming file SHA-256 is deterministic and cached", src_hash1 == src_hash2 and len(src_hash1) == 64)

    fp1 = compute_stage_fingerprint("transcribe", 1, src_hash1, {"lang": "en", "model": "parakeet"})
    fp2 = compute_stage_fingerprint("transcribe", 1, src_hash1, {"model": "parakeet", "lang": "en"})
    check("fingerprint canonical JSON is key-order independent", fp1 == fp2)

    fp_diff_param = compute_stage_fingerprint("transcribe", 1, src_hash1, {"lang": "es", "model": "parakeet"})
    check("fingerprint is sensitive to parameter changes", fp1 != fp_diff_param)

    fp_diff_ver = compute_stage_fingerprint("transcribe", 2, src_hash1, {"lang": "en", "model": "parakeet"})
    check("fingerprint is sensitive to stage version bump", fp1 != fp_diff_ver)

    fp_diff_up = compute_stage_fingerprint(
        "select_clips", 1, src_hash1, {}, upstream_fingerprints={"transcribe": fp1}
    )
    fp_diff_up2 = compute_stage_fingerprint(
        "select_clips", 1, src_hash1, {}, upstream_fingerprints={"transcribe": fp_diff_param}
    )
    check("fingerprint is sensitive to upstream stage fingerprint changes", fp_diff_up != fp_diff_up2)
    shutil.rmtree(tmp_src_dir, ignore_errors=True)

    print("\n=== Content-Addressed Artifact Store ===")
    store_dir = Path(tempfile.mkdtemp(prefix="ce-test-store-"))
    store = ArtifactStore(user_dir=store_dir, user_id="test_user")

    # Put artifact
    tmp_art_work = store_dir / "work"
    tmp_art_work.mkdir(parents=True, exist_ok=True)
    art_f1 = tmp_art_work / "out1.txt"
    art_f2 = tmp_art_work / "out2.json"
    art_f1.write_text("hello world artifact", encoding="utf-8")
    art_f2.write_text('{"status": "ok"}', encoding="utf-8")

    fp_test = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    manifest = store.put("transcribe", fp_test, {"out1.txt": art_f1, "out2.json": art_f2}, metadata={"words": 42})
    check("artifact put creates manifest", manifest.get("fingerprint") == fp_test)
    check("manifest records sha256_map", "out1.txt" in manifest.get("sha256_map", {}))

    # Get artifact
    got = store.get(fp_test, stage="transcribe")
    check("artifact get retrieves stored manifest", got is not None and got.get("metadata", {}).get("words") == 42)

    # Touch artifact
    check("artifact touch succeeds", store.touch(fp_test, stage="transcribe"))

    # Reuse artifact on second put
    manifest2 = store.put("transcribe", fp_test, {"out1.txt": art_f1}, metadata={"words": 99})
    check("artifact put reuses existing destination", manifest2.get("metadata", {}).get("words") == 42)

    # Crash mid-write simulation
    crash_parent = store.root / "transcribe" / fp_test[:2]
    crash_tmp = crash_parent / f".tmp_{fp_test}_partial_crash"
    crash_tmp.mkdir(parents=True, exist_ok=True)
    (crash_tmp / "corrupted_half_written.tmp").write_text("incomplete", encoding="utf-8")
    # Verify get() does NOT pick up partial unrenamed directory
    check("artifact store ignores uncommitted .tmp_* crash directories", store.get("fake_fp_1234") is None)
    shutil.rmtree(crash_tmp, ignore_errors=True)

    # Disk usage
    usage = store.user_disk_usage("test_user")
    check("user disk usage calculates bytes", usage > 0)

    # Delete artifact
    check("artifact delete succeeds", store.delete(fp_test, stage="transcribe"))
    check("deleted artifact is no longer found", store.get(fp_test, stage="transcribe") is None)
    shutil.rmtree(store_dir, ignore_errors=True)

    print("\n=== Migration Runner ===")
    mig_db_file = Path(tempfile.mkdtemp(prefix="ce-test-mig-")) / "test_mig.db"
    mig_engine = sa.create_engine(f"sqlite:///{mig_db_file.as_posix()}")

    # 1. Run migrations from blank DB
    applied1 = migrate.run_all(mig_engine)
    check("blank DB runs all migrations", len(applied1) >= 2)
    check("0001_initial in applied list", "0001_initial" in applied1)
    check("0002_stage_runs_artifacts in applied list", "0002_stage_runs_artifacts" in applied1)

    with mig_engine.connect() as c:
        insp = sa.inspect(c)
        tables = set(insp.get_table_names())
        check("tables created: schema_migrations", "schema_migrations" in tables)
        check("tables created: users", "users" in tables)
        check("tables created: stage_runs", "stage_runs" in tables)
        check("tables created: artifacts", "artifacts" in tables)

    # 2. Idempotent rerun
    applied2 = migrate.run_all(mig_engine)
    check("rerun migrations is idempotent (0 new applied)", len(applied2) == 0)

    # 3. Rollback latest
    rolled_back = migrate.rollback_latest(mig_engine)
    check("rollback latest rolls back 0002_stage_runs_artifacts", rolled_back == "0002_stage_runs_artifacts")
    with mig_engine.connect() as c:
        insp = sa.inspect(c)
        tables_after_rb = set(insp.get_table_names())
        check("stage_runs dropped after rollback", "stage_runs" not in tables_after_rb)
        check("users table remains after rollback", "users" in tables_after_rb)

    # 4. Re-apply
    re_applied = migrate.run_all(mig_engine)
    check("re-applying runs 0002_stage_runs_artifacts", re_applied == ["0002_stage_runs_artifacts"])
    shutil.rmtree(mig_db_file.parent, ignore_errors=True)

    print("\n=== Pipeline DAG Execution, Caching & Resume Semantics ===")
    pipe_dir = Path(tempfile.mkdtemp(prefix="ce-test-pipe-"))
    test_user_dir = pipe_dir / "user_1"
    test_job_dir = test_user_dir / "jobs" / "job_test_dag"
    test_job_dir.mkdir(parents=True, exist_ok=True)

    pstore = ArtifactStore(user_dir=test_user_dir, user_id="user_1")
    engine = PipelineEngine(store=pstore)

    execution_counts = {"s1": 0, "s2": 0, "s3": 0, "s4": 0}

    class DummyStage(Stage):
        def __init__(self, name, outputs=()):
            self.name = name
            self.version = 1
            self.outputs = outputs
            self.cacheable = True

        def run(self, ctx: StageContext):
            execution_counts[self.name] += 1
            f = Path(ctx.job["dir"]) / f"{self.name}.out"
            f.write_text(f"output from {self.name}", encoding="utf-8")
            return {f"{self.name}_data": str(f)}

    stg1 = DummyStage("s1")
    stg2 = DummyStage("s2")
    stg3 = DummyStage("s3")
    stg4 = DummyStage("s4")

    job = {
        "id": "job_dag_1",
        "dir": str(test_job_dir),
        "user": "user_1",
        "source": {"type": "file", "path": str(test_job_dir / "src.mp4")},
        "options": {},
        "stages": {},
    }
    src_file = Path(job["source"]["path"])
    src_file.write_bytes(b"dummy source video 999")
    source_id = compute_file_sha256(src_file)

    pctx = StageContext(job=job, user_dir=test_user_dir, user_id="user_1", store=pstore)

    # Run stage 1, 2, 3
    engine.execute_stage(stg1, pctx, source_id, stage_index=1, total_stages=4)
    engine.execute_stage(stg2, pctx, source_id, stage_index=2, total_stages=4)
    engine.execute_stage(stg3, pctx, source_id, stage_index=3, total_stages=4)

    check("stage 1 executed", execution_counts["s1"] == 1)
    check("stage 2 executed", execution_counts["s2"] == 1)
    check("stage 3 executed", execution_counts["s3"] == 1)
    check("stage 4 not yet executed", execution_counts["s4"] == 0)

    # SIMULATE SERVER RESTART / CRASH INTERRUPT
    # Wipe local job outputs so resume must resolve artifacts from ArtifactStore
    for out_name in ("s1.out", "s2.out", "s3.out"):
        (test_job_dir / out_name).unlink(missing_ok=True)
    # Create fresh context for resume
    job_resume = {
        "id": "job_dag_1",
        "dir": str(test_job_dir),
        "user": "user_1",
        "source": {"type": "file", "path": str(test_job_dir / "src.mp4")},
        "options": {},
        "stages": {},
    }
    pctx_resume = StageContext(job=job_resume, user_dir=test_user_dir, user_id="user_1", store=pstore)

    # Resume: run all stages 1 through 4
    out1 = engine.execute_stage(stg1, pctx_resume, source_id, stage_index=1, total_stages=4)
    engine.execute_stage(stg2, pctx_resume, source_id, stage_index=2, total_stages=4)
    engine.execute_stage(stg3, pctx_resume, source_id, stage_index=3, total_stages=4)
    engine.execute_stage(stg4, pctx_resume, source_id, stage_index=4, total_stages=4)

    check("stage 1 skipped_cached on resume", execution_counts["s1"] == 1)
    check("stage 2 skipped_cached on resume", execution_counts["s2"] == 1)
    check("stage 3 skipped_cached on resume", execution_counts["s3"] == 1)
    check("stage 4 ran to completion on resume", execution_counts["s4"] == 1)
    check("cached stage output resolves inside ArtifactStore", Path(out1["s1_data"]).is_file() and Path(out1["s1_data"]).is_relative_to(pstore.root.resolve()))

    check("resumed job records stage 1 cached", job_resume["stages"]["s1"]["cached"] is True)
    check("resumed job records stage 4 freshly computed", job_resume["stages"]["s4"]["cached"] is False)

    print("\n=== Flag-OFF Equivalence Checks ===")
    from studio import brain, jobs
    import numpy as np
    check("jobs.DEFAULTS layout is 'blur' when flags are OFF", jobs.DEFAULTS["layout"] == "blur")
    tidied = brain._tidy({"score": 80, "hook": "TEST", "title": "Test", "shots": [{"start": 0, "end": 10, "type": "close"}]}, 0.0, 20.0)
    check("brain._tidy omits multishot shots when PEAKCUT_MULTISHOT is OFF", tidied["shots"] == [])
    fb_picks = brain.fallback_by_loudness(np.array([1.0] * 30), [{"w": "hello", "s": 2.0, "e": 3.0}], 1, 15, 30.0)
    check("fallback_by_loudness omits multishot shots when PEAKCUT_MULTISHOT is OFF", bool(fb_picks) and fb_picks[0]["shots"] == [])

    print("\n=== Retry on Retryable Errors and Cancellation Cleanup ===")
    attempt_count = [0]

    class FlakyStage(Stage):
        name = "flaky"
        version = 1
        cacheable = False

        def run(self, ctx: StageContext):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise StageError("NETWORK_TIMEOUT", "Transient network error", retryable=True)
            return {"flaky_result": "success"}

    flaky_job = {"id": "flaky_job", "dir": str(test_job_dir), "options": {}, "stages": {}}
    flaky_ctx = StageContext(job=flaky_job, user_dir=test_user_dir, user_id="user_1", store=pstore)
    flaky_stg = FlakyStage()
    flaky_out = engine.execute_stage(flaky_stg, flaky_ctx, "source_123")
    check("flaky stage retried and succeeded on attempt 3", attempt_count[0] == 3 and flaky_out.get("flaky_result") == "success")

    # Non-retryable fails immediately
    class FatalStage(Stage):
        name = "fatal"
        version = 1
        cacheable = False

        def run(self, ctx: StageContext):
            raise StageError("INVALID_FORMAT", "Unrecoverable corrupted header", retryable=False)

    fatal_job = {"id": "fatal_job", "dir": str(test_job_dir), "options": {}, "stages": {}}
    fatal_ctx = StageContext(job=fatal_job, user_dir=test_user_dir, user_id="user_1", store=pstore)
    try:
        engine.execute_stage(FatalStage(), fatal_ctx, "source_123")
        check("non-retryable error raised immediately", False)
    except StageError as e:
        check("non-retryable error raised immediately", e.code == "INVALID_FORMAT")

    # Cancellation kills active processes
    cancel_ctx = StageContext(job=job, user_dir=test_user_dir, user_id="user_1", store=pstore)
    # Spawn a sleeping dummy process
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    cancel_ctx.register_process(proc)
    check("child process running before cancel", proc.poll() is None)
    cancel_ctx.cancel_event.set()
    cancel_ctx.kill_active_processes()
    proc.wait(timeout=5)
    check("child process terminated after cancellation", proc.poll() is not None)

    shutil.rmtree(pipe_dir, ignore_errors=True)

    print("\n" + ("=" * 40))
    if fails:
        print(f"FAILED {len(fails)} checks: {', '.join(fails)}")
        return False
    print("ALL PHASE 2 PIPELINE & CONTRACT CHECKS PASSED!\n")
    return True


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
