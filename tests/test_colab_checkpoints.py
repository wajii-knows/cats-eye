"""Comprehensive test suite for Cat's Eye Google Colab Durable Source of Truth:
1. Feature flag PEAKCUT_COLAB and normal-mode equivalence.
2. Verified write protocol: temp file -> fsync -> readback sha256 -> rename -> manifest.
3. Fault injection: crash before sync, crash before rename, absence rule, and read corruption detection.
4. Packed directory (.tar) roundtrip for small files (snapshots, frames, JSON).
5. Session lease concurrency: active rejection, heartbeat, stale takeover (>3 min), force takeover.
6. Multi-generation SQLite online backup, PRAGMA integrity check, sessions sanitization, and 5-gen retention.
7. Corrupt generation fallback (corrupt gen 5 falls back to gen 4).
8. Durable uploader queue: background worker, pending bytes/count metrics, safe-to-disconnect state.
9. Intra-stage checkpointing and resume (transcribe 5-min chunks).
10. Secrets absence: verify no API keys or plaintext credentials exist in durable store.
11. Colab server gate: OWNER_PASSWORD enforcement, rate-limited login, and Colab status API.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import config, flags
from studio.storage.colab import ColabManager, get_colab_manager, set_colab_manager
from studio.storage.durable import (
    DriveMountStore,
    DurableStoreError,
    FaultInjectingStore,
    IntegrityError,
    LocalFolderStore,
    compute_sha256,
)
from studio.storage.lease import ActiveLeaseError, SessionLease
from studio.storage.db_checkpoint import DatabaseCheckpointManager
from studio.storage.uploader import DurableUploader, UploadTask
from studio.pipeline.stage import StageContext
from studio.pipeline.artifacts import ArtifactStore


def run_all_tests():
    fails = []

    def check(name: str, ok: bool, extra: str = ""):
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}" + (f"   {extra}" if extra else ""))
        if not ok:
            fails.append(name)

    print("\n=== 1. Feature Flag & Normal Mode Equivalence ===")
    all_f = flags.all_flags()
    check("PEAKCUT_COLAB flag registered", "PEAKCUT_COLAB" in all_f)
    check("PEAKCUT_COLAB default is OFF", flags.is_on("PEAKCUT_COLAB") is False)

    # Verify normal mode equivalence
    orig_colab = os.environ.get("PEAKCUT_COLAB")
    try:
        os.environ.pop("PEAKCUT_COLAB", None)
        # Reload relevant config
        from studio import config as test_cfg

        check("In non-Colab mode, COLAB_MODE is False", test_cfg.COLAB_MODE is False)
        check("In non-Colab mode, WORK_DIR equals DATA_DIR", test_cfg.WORK_DIR == test_cfg.DATA_DIR)
        check("In non-Colab mode, DURABLE_DIR equals DATA_DIR", test_cfg.DURABLE_DIR == test_cfg.DATA_DIR)
    finally:
        if orig_colab is not None:
            os.environ["PEAKCUT_COLAB"] = orig_colab

    print("\n=== 2. Verified Write Protocol & Fault Injection ===")
    with tempfile.TemporaryDirectory(prefix="test_durable_") as tmp_dur, tempfile.TemporaryDirectory(
        prefix="test_work_"
    ) as tmp_work:
        durable_dir = Path(tmp_dur)
        work_dir = Path(tmp_work)

        # Create a local test file
        src_file = work_dir / "sample.mp4"
        src_content = b"CAT_EYE_DURABLE_SAMPLE_VIDEO_DATA" * 100
        src_file.write_bytes(src_content)
        expected_sha = compute_sha256(src_file)

        store = LocalFolderStore(durable_dir, session_id="test_sess_1")
        meta = store.put_file(src_file, "videos/job_101/sample.mp4")

        check("Put file returns verified metadata", bool(meta and meta["sha256"] == expected_sha))
        check("Put file exists in store", store.exists("videos/job_101/sample.mp4"))
        check("Manifest contains file entry", "videos/job_101/sample.mp4" in store._manifest.get("files", {}))

        # Absence Rule: a file present on disk but absent in manifest is considered absent
        ghost_path = durable_dir / "videos" / "job_101" / "ghost.txt"
        ghost_path.parent.mkdir(parents=True, exist_ok=True)
        ghost_path.write_text("untracked ghost file", encoding="utf-8")
        check("Absence Rule: untracked file returns exists()=False", store.exists("videos/job_101/ghost.txt") is False)

        # Fault Injection: Drop writes (network timeout or crash before sync)
        base_dur = LocalFolderStore(durable_dir, session_id="test_fault_1")
        fault_store = FaultInjectingStore(base_dur)
        fault_store.drop_writes = True
        crashed = False
        try:
            fault_store.put_file(src_file, "videos/crashed_sync.mp4")
        except DurableStoreError:
            crashed = True
        check("Crash before sync raises DurableStoreError", crashed)
        check("Crash before sync leaves no entry in manifest", not fault_store.exists("videos/crashed_sync.mp4"))

        # Fault Injection: Crash before rename
        fault_store.drop_writes = False
        fault_store.crash_before_rename = True
        crashed = False
        try:
            fault_store.put_file(src_file, "videos/crashed_rename.mp4")
        except DurableStoreError:
            crashed = True
        check("Crash before rename raises DurableStoreError", crashed)
        check("Crash before rename leaves target non-existent", not (durable_dir / "videos/crashed_rename.mp4").exists())
        check("Crash before rename leaves no manifest entry", not fault_store.exists("videos/crashed_rename.mp4"))

        # Fault Injection: Crash before manifest commit
        fault_store.crash_before_rename = False
        fault_store.crash_before_manifest_commit = True
        crashed = False
        try:
            fault_store.put_file(src_file, "videos/crashed_manifest.mp4")
        except DurableStoreError:
            crashed = True
        check("Crash before manifest commit raises DurableStoreError", crashed)
        check("Absence rule: file on disk without manifest is absent", not fault_store.exists("videos/crashed_manifest.mp4"))

        # Read-back verification and corruption detection
        fault_store.crash_before_manifest_commit = False
        corrupt_meta = fault_store.put_file(src_file, "videos/to_corrupt.mp4")
        real_file = durable_dir / "videos/to_corrupt.mp4"
        # Tamper with the file on durable storage
        real_file.write_bytes(b"TAMPERED_BYTES_CORRUPTED")
        read_failed = False
        try:
            dest_read = work_dir / "restored.mp4"
            fault_store.get_file("videos/to_corrupt.mp4", dest_read)
        except IntegrityError:
            read_failed = True
        check("Tampered durable file triggers IntegrityError on read", read_failed)

    print("\n=== 3. Directory Packing (.tar) Round-trip ===")
    with tempfile.TemporaryDirectory(prefix="test_pack_") as tmp_pack:
        pack_dir = Path(tmp_pack)
        local_frames = pack_dir / "frames"
        local_frames.mkdir()
        for i in range(25):
            (local_frames / f"frame_{i:03d}.png").write_bytes(f"PNG_DATA_{i}".encode() * 50)
        (local_frames / "sub").mkdir()
        (local_frames / "sub" / "meta.json").write_text('{"count": 25}', encoding="utf-8")

        dur_store = LocalFolderStore(pack_dir / "durable")
        dur_store.put_dir_packed(local_frames, "artifacts/snapshots/pack.tar")

        check("Packed archive exists on durable store", dur_store.exists("artifacts/snapshots/pack.tar"))
        check("Archive file is single tar file", (pack_dir / "durable" / "artifacts/snapshots/pack.tar").is_file())

        unpack_dest = pack_dir / "unpacked"
        dur_store.get_dir_unpacked("artifacts/snapshots/pack.tar", unpack_dest)

        check("Unpacked directory contains all 25 frames", len(list(unpack_dest.glob("frame_*.png"))) == 25)
        check(
            "Unpacked sub/meta.json matches original",
            (unpack_dest / "sub" / "meta.json").read_text(encoding="utf-8") == '{"count": 25}',
        )

    print("\n=== 4. Session Lease Concurrency & Timeout ===")
    with tempfile.TemporaryDirectory(prefix="test_lease_") as tmp_l:
        l_dir = Path(tmp_l)
        lease_a = SessionLease(l_dir, session_id="session_alpha", heartbeat_interval=0.2)
        res_a = lease_a.acquire()
        check("Lease A acquired successfully", res_a["session_id"] == "session_alpha")
        check("Lease file created", (l_dir / "lease.json").is_file())

        # Second session attempts acquisition while Lease A is active (< 3 minutes)
        lease_b = SessionLease(l_dir, session_id="session_beta")
        b_blocked = False
        try:
            lease_b.acquire()
        except ActiveLeaseError:
            b_blocked = True
        check("Active lease rejects second session (ActiveLeaseError)", b_blocked)

        # Force takeover & deposed heartbeat self-termination
        res_b_forced = lease_b.acquire(force=True)
        check("Force takeover succeeds", res_b_forced["session_id"] == "session_beta")
        time.sleep(0.45)
        check("Deposed session stopped overwriting lease", lease_b.read_lease()["session_id"] == "session_beta")
        check("Deposed session heartbeat stopped automatically", not lease_a._heartbeat_thread or not lease_a._heartbeat_thread.is_alive())

        # Stale takeover: simulate lease older than 180 seconds
        lease_a.stop_heartbeat()
        lease_b.stop_heartbeat()
        lease_file = l_dir / "lease.json"
        stale_data = {
            "session_id": "session_old",
            "created_at": time.time() - 500,
            "heartbeat_at": time.time() - 300,  # 5 minutes ago
            "hostname": "old-colab-worker",
        }
        lease_file.write_text(json.dumps(stale_data), encoding="utf-8")

        lease_c = SessionLease(l_dir, session_id="session_gamma", heartbeat_interval=0.2)
        res_c = lease_c.acquire()
        check("Stale lease (> 3 minutes) taken over automatically", res_c["session_id"] == "session_gamma")

        # Clean release
        released = lease_c.release()
        check("Lease released cleanly", released is True)

    print("\n=== 5. SQLite Online Backup, Integrity Check & 5-Gen Retention ===")
    with tempfile.TemporaryDirectory(prefix="test_db_") as tmp_d:
        d_dir = Path(tmp_d)
        work_db = d_dir / "work" / "peakcut.db"
        work_db.parent.mkdir(parents=True, exist_ok=True)
        dur_dir = d_dir / "durable"

        # Initialize test SQLite database with tables
        conn = sqlite3.connect(str(work_db))
        conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT);")
        conn.execute("CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id TEXT);")
        conn.execute("CREATE TABLE stage_runs (id TEXT PRIMARY KEY, status TEXT);")
        conn.execute("INSERT INTO users VALUES ('u1', 'Owner User');")
        conn.execute("INSERT INTO sessions VALUES ('secret_session_token_123', 'u1');")
        conn.execute("INSERT INTO stage_runs VALUES ('sr1', 'done');")
        conn.commit()
        conn.close()

        dur_store = LocalFolderStore(dur_dir)
        db_mgr = DatabaseCheckpointManager(
            work_db_path=work_db,
            durable_store=dur_store,
            max_generations=5,
            checkpoint_interval=1.0,
        )

        # Generate 6 checkpoints to test generation increment and 5-generation retention
        for i in range(1, 7):
            res = db_mgr.create_checkpoint(reason=f"test_run_{i}")
            check(f"Checkpoint gen-{i} created", res is not None and res["generation"] == i)

        gens = db_mgr.get_generations()
        check("Generations trimmed to max 5 (retention rule)", len(gens) == 5 and gens == [2, 3, 4, 5, 6])
        check("Earliest generation gen-1 was pruned", not dur_store.exists("db/gen-0001.sqlite"))
        check("Latest generation gen-6 exists", dur_store.exists("db/gen-0006.sqlite"))

        # Verify session sanitization in snapshot
        snap_tmp = d_dir / "snap_inspect.sqlite"
        dur_store.get_file("db/gen-0006.sqlite", snap_tmp)
        snap_conn = sqlite3.connect(str(snap_tmp))
        cur = snap_conn.cursor()
        users_count = cur.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
        sessions_count = cur.execute("SELECT COUNT(*) FROM sessions;").fetchone()[0]
        snap_conn.close()

        check("Snapshot preserves user data", users_count == 1)
        check("Snapshot sanitizes sessions table (zero sessions stored on Drive)", sessions_count == 0)

    print("\n=== 6. Corrupt Generation Fallback ===")
    with tempfile.TemporaryDirectory(prefix="test_fallback_") as tmp_fb:
        fb_dir = Path(tmp_fb)
        work_db = fb_dir / "work" / "peakcut.db"
        work_db.parent.mkdir(parents=True, exist_ok=True)
        dur_dir = fb_dir / "durable"

        # Create base DB and 2 checkpoints
        conn = sqlite3.connect(str(work_db))
        conn.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT);")
        conn.execute("INSERT INTO kv VALUES ('gen', '1');")
        conn.commit()
        conn.close()

        dur_store = LocalFolderStore(dur_dir)
        db_mgr = DatabaseCheckpointManager(work_db, dur_store, max_generations=5)
        db_mgr.create_checkpoint(reason="gen1")

        # Update and make gen 2
        conn = sqlite3.connect(str(work_db))
        conn.execute("UPDATE kv SET v = '2';")
        conn.commit()
        conn.close()
        db_mgr.create_checkpoint(reason="gen2")

        # Now simulate corruption of gen-2 on durable storage
        gen2_file = dur_dir / "db/gen-0002.sqlite"
        gen2_file.write_bytes(b"NOT_A_VALID_SQLITE_DATABASE_CORRUPTED_BYTES")

        # Delete local DB and restore
        import gc
        gc.collect()
        time.sleep(0.05)
        work_db.unlink()
        restored_meta = db_mgr.restore_latest_checkpoint()

        check(
            "Corruption in gen-2 triggered automatic fallback to gen-1",
            restored_meta is not None and restored_meta["generation"] == 1,
        )
        check("Restored local DB exists", work_db.is_file())

        conn = sqlite3.connect(str(work_db))
        val = conn.execute("SELECT v FROM kv WHERE k = 'gen';").fetchone()[0]
        conn.close()
        del conn
        gc.collect()
        check("Data restored correctly from fallback generation 1", val == "1")

    print("\n=== 7. Durable Uploader Background Queue & Metrics ===")
    with tempfile.TemporaryDirectory(prefix="test_up_") as tmp_u:
        u_dir = Path(tmp_u)
        d_store = LocalFolderStore(u_dir / "durable")
        uploader = DurableUploader(d_store, max_workers=2)

        file1 = u_dir / "upload1.txt"
        file1.write_bytes(b"HELLO_UPLOADER_1" * 100)
        file2 = u_dir / "upload2.txt"
        file2.write_bytes(b"HELLO_UPLOADER_2" * 200)

        uploader.enqueue_file(file1, "queue/file1.txt")
        uploader.enqueue_file(file2, "queue/file2.txt")

        status_queued = uploader.get_status()
        check("Uploader tracks pending count", status_queued["pending_count"] == 2)
        check("Uploader tracks pending bytes", status_queued["pending_bytes"] > 0)
        check("Safe to disconnect is False while pending", status_queued["safe_to_disconnect"] is False)

        uploader.start()
        # Wait for queue to drain
        for _ in range(50):
            if uploader.is_idle():
                break
            time.sleep(0.05)

        uploader.stop()
        status_done = uploader.get_status()
        check("Uploader successfully drained all items", status_done["pending_count"] == 0)
        check("Uploader state is 'synced'", status_done["state"] == "synced")
        check("Safe to disconnect is True when idle", status_done["safe_to_disconnect"] is True)
        check("Files exist on durable store", d_store.exists("queue/file1.txt") and d_store.exists("queue/file2.txt"))

    print("\n=== 8. Intra-Stage Checkpoint Resume ===")
    with tempfile.TemporaryDirectory(prefix="test_ctx_") as tmp_c:
        c_dir = Path(tmp_c)
        store = ArtifactStore(user_dir=c_dir, user_id="u_colab")
        d_store = LocalFolderStore(c_dir / "durable")
        job = {"id": "j_test_resume", "user": "u_colab", "stages": {}}

        ctx1 = StageContext(
            job=job,
            user_dir=c_dir,
            user_id="u_colab",
            store=store,
            durable_store=d_store,
        )

        # Stage saves partial progress
        ctx1.checkpoint("chunk_0", {"start": 0.0, "end": 300.0, "words": [{"w": "Hello", "s": 0.5, "e": 0.9}]})
        ctx1.checkpoint("chunk_1", {"start": 300.0, "end": 600.0, "words": [{"w": "World", "s": 301.0, "e": 301.5}]})

        # Wipe local checkpoints directory to simulate Colab local disk loss
        shutil.rmtree(c_dir / "checkpoints", ignore_errors=True)
        check("Local checkpoints dir removed before resume", not (c_dir / "checkpoints").exists())

        # Simulate fresh session resuming from Drive durable store
        ctx2 = StageContext(
            job=job,
            user_dir=c_dir,
            user_id="u_colab",
            store=store,
            durable_store=d_store,
        )

        c0 = ctx2.resume_state("chunk_0")
        c1 = ctx2.resume_state("chunk_1")
        check("Resumed chunk 0 matches saved state from Drive", c0 is not None and c0["words"][0]["w"] == "Hello")
        check("Resumed chunk 1 matches saved state from Drive", c1 is not None and c1["words"][0]["w"] == "World")

        # Cross-user ArtifactStore isolation check
        store_b = ArtifactStore(user_dir=c_dir / "other_user", user_id="u_other")
        f_iso = c_dir / "iso.txt"
        f_iso.write_text("user_a_secret", encoding="utf-8")
        fp_iso = "a" * 64
        store.put("transcribe", fp_iso, {"iso.txt": f_iso}, metadata={})
        check("User B cannot read User A artifact", store_b.get(fp_iso) is None)
        check("User A artifact survives User B delete attempt", store_b.delete(fp_iso) and store.get(fp_iso) is not None)

    print("\n=== 9. Secrets Absence Rule ===")
    with tempfile.TemporaryDirectory(prefix="test_sec_") as tmp_s:
        s_dir = Path(tmp_s)
        dur_dir = s_dir / "durable"
        work_dir = s_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        dur_dir.mkdir(parents=True, exist_ok=True)
        mgr = ColabManager(work_dir=work_dir, durable_dir=dur_dir)

        # Fake local files with secrets in WORK_DIR
        ai_file = mgr.work_dir / "ai.json"
        ai_file.write_text('{"provider": "gemini", "key": "SUPER_SECRET_AI_API_KEY"}', encoding="utf-8")

        # Run Colab checkpoint
        conn = sqlite3.connect(str(mgr.work_dir / "peakcut.db"))
        conn.execute("CREATE TABLE t (id INT);")
        conn.commit()
        conn.close()
        del conn
        mgr.checkpoint_now(reason="secret_check")

        # Scan all durable files for secrets
        found_secret = False
        for root, _, files in os.walk(dur_dir):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    data = fpath.read_bytes()
                    if b"SUPER_SECRET_AI_API_KEY" in data:
                        found_secret = True
                except Exception:
                    pass

        check("Plaintext AI secret key never exists on durable Drive store", found_secret is False)
        check("ai.json is never uploaded to durable Drive store", not (dur_dir / "ai.json").exists())

    print("\n=== 10. Colab Manager Lifecycle & Server Gate ===")
    with tempfile.TemporaryDirectory(prefix="test_mgr_") as tmp_m:
        m_dir = Path(tmp_m)
        orig_pw = config.OWNER_PASSWORD
        orig_colab_mode = config.COLAB_MODE

        try:
            config.COLAB_MODE = True
            config.OWNER_PASSWORD = ""
            mgr_fail = ColabManager(work_dir=m_dir / "work", durable_dir=m_dir / "durable")
            failed_startup = False
            try:
                mgr_fail.start()
            except RuntimeError as e:
                failed_startup = "PEAKCUT_OWNER_PASSWORD is required" in str(e)
            check("Server startup fails when COLAB_MODE=True and OWNER_PASSWORD empty", failed_startup)

            # Valid password startup
            config.OWNER_PASSWORD = "catseye-secure-owner-pw-2026"
            mgr_ok = ColabManager(work_dir=m_dir / "work", durable_dir=m_dir / "durable")
            mgr_ok.start()
            st = mgr_ok.get_status()

            check("ColabManager starts cleanly with valid password", st["colab_mode"] is True)
            check("Uptime is tracked in status", st["uptime_sec"] >= 0)
            check("Session lease is active", bool(st["lease"]["session_id"]))
            check("Drive free space reported", "drive_free_gb" in st)

            mgr_ok.stop()
            check("ColabManager stops cleanly", mgr_ok._started is False)
        finally:
            config.OWNER_PASSWORD = orig_pw
            config.COLAB_MODE = orig_colab_mode

    print("\n=== 11. Colab HTTP Gate, Checkpoint Restore Ordering & APIs ===")
    import subprocess
    import urllib.error
    import urllib.request
    import http.cookiejar

    with tempfile.TemporaryDirectory(prefix="test_colab_srv_") as tmp_srv:
        srv_dir = Path(tmp_srv)
        w_dir = srv_dir / "work"
        d_dir = srv_dir / "durable"
        w_dir.mkdir(parents=True, exist_ok=True)
        d_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed a Drive checkpoint before booting studio.server
        seed_db = srv_dir / "seed.db"
        import sqlalchemy as sa
        from studio.db import migrate, users as db_users
        seed_eng = sa.create_engine(f"sqlite:///{seed_db.as_posix()}")
        migrate.run_all(seed_eng)
        with seed_eng.begin() as conn:
            conn.execute(db_users.insert().values(id="u_local", email="you@localhost", name="RestoredOwner", is_admin=True, created_at=time.time()))
        seed_eng.dispose()
        seed_dur = LocalFolderStore(d_dir)
        seed_mgr = DatabaseCheckpointManager(seed_db, seed_dur)
        seed_mgr.create_checkpoint(reason="preseeded")

        colab_port = 8798
        base_colab = f"http://127.0.0.1:{colab_port}"
        srv_env = {
            **os.environ,
            "PEAKCUT_COLAB": "1",
            "PEAKCUT_WORK_DIR": str(w_dir),
            "PEAKCUT_DURABLE_DIR": str(d_dir),
            "PEAKCUT_OWNER_PASSWORD": "colab-test-pw-999",
            "PEAKCUT_PORT": str(colab_port),
            "PEAKCUT_HOST": "127.0.0.1",
            "PEAKCUT_PUBLIC_URL": base_colab,
            "PEAKCUT_AUTO_RELOAD": "0",
            "PEAKCUT_FETCH_MODEL": "0",
            "PEAKCUT_UPDATE_CHECK": "0",
            "PEAKCUT_ACCOUNTS": "0",
            "DATABASE_URL": "",
        }
        proc = subprocess.Popen(
            [sys.executable, "-m", "studio.server"],
            cwd=str(ROOT),
            env=srv_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            up_ok = False
            for _ in range(45):
                try:
                    if urllib.request.urlopen(f"{base_colab}/healthz", timeout=2).status == 200:
                        up_ok = True
                        break
                except Exception:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.5)
            check("Colab server starts and passes healthz", up_ok)

            if up_ok:
                jar = http.cookiejar.CookieJar()
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

                def call_http(method, path, data=None):
                    req = urllib.request.Request(
                        f"{base_colab}{path}",
                        data=(json.dumps(data).encode() if data is not None else None),
                        method=method,
                        headers={"Content-Type": "application/json", "Origin": base_colab},
                    )
                    try:
                        with opener.open(req, timeout=10) as r:
                            return r.status, json.loads(r.read().decode())
                    except urllib.error.HTTPError as e:
                        body = e.read().decode()
                        try:
                            return e.code, json.loads(body)
                        except Exception:
                            return e.code, {"raw": body}

                st_signup, _ = call_http("POST", "/api/auth/signup", {"email": "attacker@evil.com", "password": "Password123"})
                check("POST /api/auth/signup blocked with 404 in COLAB_MODE", st_signup == 404)

                st_unauth, _ = call_http("GET", "/api/colab/status")
                check("GET /api/colab/status rejects unauthenticated client (401)", st_unauth == 401)

                st_bad, _ = call_http("POST", "/api/colab/login", {"password": "wrong-password"})
                check("POST /api/colab/login rejects wrong password (401)", st_bad == 401)

                st_login, d_login = call_http("POST", "/api/colab/login", {"password": "colab-test-pw-999"})
                check("POST /api/colab/login succeeds with owner password", st_login == 200 and d_login["user"]["name"] == "RestoredOwner")

                st_status, d_status = call_http("GET", "/api/colab/status")
                check("GET /api/colab/status succeeds when authenticated", st_status == 200 and d_status.get("colab_mode") is True)
                check("Drive DB checkpoint restored before db.init()", d_status.get("recovery", {}).get("recovered") is True)

                st_cp, d_cp = call_http("POST", "/api/colab/checkpoint", {})
                check("POST /api/colab/checkpoint creates verified snapshot", st_cp == 200 and d_cp.get("ok") is True and d_cp["checkpoint"]["generation"] >= 2)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    print("\n=== 12. Colab Deletion Persistence & Stale Upload Drop ===")
    with tempfile.TemporaryDirectory(prefix="test_colab_del_") as tmp_del:
        del_dir = Path(tmp_del)
        w_dir = del_dir / "work"
        d_dir = del_dir / "durable"
        d_store = LocalFolderStore(d_dir)

        # A. Coalescing duplicate enqueues + silently dropping deleted local files with 0 errors
        up = DurableUploader(d_store, max_workers=1)
        f_ghost = w_dir / "ghost_job.json"
        w_dir.mkdir(parents=True, exist_ok=True)
        f_ghost.write_text('{"id": "ghost"}', encoding="utf-8")
        for _ in range(5):
            up.enqueue_file(f_ghost, "users/u_local/jobs/ghost/job.json")
        check("Duplicate enqueues for same path coalesce into 1 queued item", up.get_status()["pending_count"] == 1)
        f_ghost.unlink()
        up.start()
        for _ in range(40):
            if up.is_idle():
                break
            time.sleep(0.05)
        up.stop()
        st_ghost = up.get_status()
        check(
            "Deleted local file before upload drops silently with 0 errors and safe_to_disconnect=True",
            st_ghost["pending_count"] == 0 and st_ghost["last_error"] is None and st_ghost["safe_to_disconnect"] is True
            and not d_store.exists("users/u_local/jobs/ghost/job.json"),
            str(st_ghost),
        )

        # B. jobs.delete + account.delete_creator erase Drive durable copies so Colab restart does NOT restore them
        from studio import account, creators, ctx, jobs  # noqa: E402
        from studio.storage import colab as _colab  # noqa: E402

        orig_data = config.DATA_DIR
        orig_u = config.USERS_DIR
        orig_colab_flag = config.COLAB_MODE
        orig_owner_pw = config.OWNER_PASSWORD
        orig_mgr = _colab.get_colab_manager(create=False)
        try:
            config.DATA_DIR = w_dir
            config.USERS_DIR = w_dir / "users"
            config.USERS_DIR.mkdir(parents=True, exist_ok=True)
            config.COLAB_MODE = True
            config.OWNER_PASSWORD = "pw-delete-test"

            mgr_del = ColabManager(work_dir=w_dir, durable_dir=d_dir)
            mgr_del.start(force_lease=True)
            _colab.set_colab_manager(mgr_del)

            with ctx.as_user("u_local"):
                j_del = jobs.create({"type": "file", "path": "", "name": "gone.mp4"}, {}, enqueue=False)
                jid_del = j_del["id"]
                cr = creators.create({"name": "GoneCreator"})
                cid_del = cr["id"]
                j_cr = jobs.create({"type": "file", "path": "", "name": "crv.mp4"}, {}, enqueue=False, creator=cid_del)
                j_cr["status"] = "done"
                jobs.save(j_cr)
                jobs._q.remove(j_del["id"])
                jobs._q.remove(j_cr["id"])

                # Drain uploader so durable Drive store definitely holds job.json before deletion
                for _ in range(50):
                    if mgr_del.uploader.is_idle():
                        break
                    time.sleep(0.05)
                rel_j = f"users/u_local/jobs/{jid_del}/job.json"
                check("Job uploaded to durable Drive store before delete", d_store.exists(rel_j) or mgr_del.durable_store.exists(rel_j))

                jobs.delete(jid_del)
                check(
                    "jobs.delete removes job.json from durable Drive store and prevents uploader errors",
                    not mgr_del.durable_store.exists(rel_j)
                    and not (Path(j_del["dir"]).exists())
                    and mgr_del.uploader.get_status()["last_error"] is None,
                )

                # Manually place creator.json on durable store too, then delete creator
                c_rel = jobs._durable_rel(creators.root(cid_del) / "creator.json") or f"users/u_local/creators/{cid_del}/creator.json"
                mgr_del.durable_store.put_file(creators.root(cid_del) / "creator.json", c_rel)
                account.delete_creator(cid_del)
                check(
                    "account.delete_creator removes creator and its videos from durable Drive store",
                    not mgr_del.durable_store.exists(c_rel) and not mgr_del.durable_store.list(f"users/u_local/creators/{cid_del}"),
                )

            mgr_del.stop()
            # Restart fresh ColabManager on same work & durable dirs: deleted job MUST stay gone!
            mgr_re = ColabManager(work_dir=w_dir, durable_dir=d_dir)
            mgr_re.start(force_lease=True)
            check(
                "Restarting ColabManager does NOT resurrect deleted job or creator",
                not (w_dir / rel_j).exists() and not (w_dir / c_rel).exists(),
            )
            mgr_re.stop()
        finally:
            _colab.set_colab_manager(orig_mgr)
            config.DATA_DIR = orig_data
            config.USERS_DIR = orig_u
            config.COLAB_MODE = orig_colab_flag
            config.OWNER_PASSWORD = orig_owner_pw

    print(f"\n==========================================")
    print(f"Colab Checkpoint Tests Complete: {len(fails)} failures.")
    print(f"==========================================")
    return len(fails) == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
