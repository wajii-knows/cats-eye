"""Verify Pause, Resume, and Remove controls for currently processing and paused video jobs."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-jctl-")
os.environ["DATABASE_URL"] = ""
os.environ["PEAKCUT_ACCOUNTS"] = "1"
os.environ["PEAKCUT_VERIFY_EMAIL"] = "0"
os.environ["PEAKCUT_AI_KEY"] = ""
os.environ["PEAKCUT_FETCH_MODEL"] = "0"
sys.path.insert(0, ".")

from fastapi.testclient import TestClient  # noqa: E402
from studio import auth, config, ctx, db, jobs, live, server  # noqa: E402
from studio.pipeline.stage import StageContext  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    if not ok:
        fails.append(name)


db.init()
u_a, _ = auth._create("ctl-a@example.com", "User A", password_hash="x")
u_b, _ = auth._create("ctl-b@example.com", "User B", password_hash="x")

# ---- Part A: Unit tests for queue removal, pause, resume, restart persistence, and safe deletion
with ctx.as_user(u_a.id):
    j_q = jobs.create({"type": "file", "path": "", "name": "q.mp4"}, {}, enqueue=True)
    j_q["clips"] = [{"id": "clip01", "status": "rendering", "start": 0, "end": 10}]
    jobs.save(j_q)
    check("queued job starts in waiting queue", jobs.waiting_total() == 1 and j_q["status"] == "queued")

    ok_p = jobs.pause(j_q["id"])
    check(
        "jobs.pause pauses queued job, clears queue, resets rendering clip, and persists",
        ok_p
        and j_q["status"] == "paused"
        and j_q["stage"] == "Paused"
        and j_q["resume"] is True
        and j_q["clips"][0]["status"] == "queued"
        and jobs.waiting_total() == 0
        and json.loads((Path(j_q["dir"]) / "job.json").read_text(encoding="utf-8"))["status"] == "paused",
        str((ok_p, j_q["status"], j_q["stage"], jobs.waiting_total())),
    )

    # Stale worker updates must not overwrite paused state
    jobs._set(j_q, status="running", stage="Stale update", progress=0.77)
    check("stale _set does not overwrite paused job status/stage", j_q["status"] == "paused" and j_q["stage"] == "Paused")

    # Resuming requeues the job cleanly
    check("pausing an already paused job returns False", not jobs.pause(j_q["id"]))
    ok_r = jobs.resume(j_q["id"])
    check(
        "jobs.resume puts paused job back onto queue",
        ok_r and j_q["status"] == "queued" and j_q["stage"] == "Resuming where it paused" and jobs.waiting_total() == 1,
        str((ok_r, j_q["status"], jobs.waiting_total())),
    )
    check("resuming a non-paused job returns False", not jobs.resume(j_q["id"]))

    # Pause persists across simulated server restart and is NOT auto-requeued
    jobs.pause(j_q["id"])
    jobs._jobs.clear()
    jobs._by_user.clear()
    jobs._paused_ids.clear()
    jobs._q = jobs._FairQueue()
    jobs.load_all()
    jobs._pick_up_after_restart()
    reloaded = jobs.get(j_q["id"])
    check(
        "paused job remains paused across restart and is not auto-requeued",
        reloaded is not None and reloaded["status"] == "paused" and jobs.waiting_total() == 0,
        str(reloaded),
    )

    # Active StageContext cooperative cancellation on pause and delete
    j_run = jobs.create({"type": "file", "path": "", "name": "run.mp4"}, {}, enqueue=False)
    j_run["status"] = "running"
    j_run["stage"] = "Writing the transcript"
    jobs.save(j_run)
    from studio.pipeline.artifacts import ArtifactStore
    sctx = StageContext(
        job=j_run,
        user_dir=Path(j_run["dir"]).parent.parent,
        user_id=u_a.id,
        store=ArtifactStore(user_dir=Path(j_run["dir"]).parent.parent, user_id=u_a.id),
    )
    jobs._running_contexts[j_run["id"]] = sctx

    ev = threading.Event()
    started = threading.Event()
    pool = ThreadPoolExecutor(1)
    fut = pool.submit(lambda: (started.set(), ev.wait(30)))
    t_w0 = time.time()

    def _trigger_pause():
        started.wait(5.0)
        jobs.pause(j_run["id"])

    t_p = threading.Thread(target=_trigger_pause, daemon=True)
    t_p.start()
    jobs._wait_for_clips(j_run, [(fut, {"id": "clip01"})], {}, 1, sctx=sctx)
    elapsed = time.time() - t_w0
    ev.set()
    t_p.join(timeout=2.0)
    pool.shutdown(wait=True)
    check(
        "pause immediately cancels StageContext and aborts _wait_for_clips",
        elapsed < 3.0 and sctx.is_cancelled() and j_run["status"] == "paused",
        f"elapsed={elapsed:.2f}s status={j_run['status']}",
    )
    jobs._running_contexts.pop(j_run["id"], None)

    # Deleting a running/processing video kills active context, removes dir, and prevents stale recreation
    j_del = jobs.create({"type": "file", "path": "", "name": "del.mp4"}, {}, enqueue=False)
    j_del["status"] = "running"
    jobs.save(j_del)
    del_dir = Path(j_del["dir"])
    sctx2 = StageContext(
        job=j_del,
        user_dir=del_dir.parent.parent,
        user_id=u_a.id,
        store=ArtifactStore(user_dir=del_dir.parent.parent, user_id=u_a.id),
    )
    jobs._running_contexts[j_del["id"]] = sctx2
    check("jobs.delete removes active running video and directory", jobs.delete(j_del["id"]) and sctx2.is_cancelled() and not del_dir.exists() and jobs.get(j_del["id"]) is None)
    jobs._set(j_del, status="done")
    jobs.save(j_del)
    check("stale _set/save after delete does not recreate deleted folder", not del_dir.exists())
    jobs._running_contexts.pop(j_del["id"], None)

# ---- Part B: Full FastAPI HTTP endpoint & ownership isolation tests
client_a = TestClient(server.app, headers={"Origin": "http://testserver"})
client_b = TestClient(server.app, headers={"Origin": "http://testserver"})

r_la = client_a.post("/api/auth/signup", json={"email": "http-a@example.com", "password": "test-pass-123", "name": "A"})
r_lb = client_b.post("/api/auth/signup", json={"email": "http-b@example.com", "password": "test-pass-123", "name": "B"})
check("HTTP test users sign up cleanly", r_la.status_code == 200 and r_lb.status_code == 200)
uid_http_a = r_la.json()["user"]["id"]

with ctx.as_user(uid_http_a):
    http_job = jobs.create({"type": "file", "path": "", "name": "stream.mp4"}, {}, enqueue=True)
    http_live = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=x", "name": "Live"}, {}, enqueue=False, kind="live")
    http_live["status"] = "live"
    jobs.save(http_live)
    jobs._q.remove(http_job["id"])

jid = http_job["id"]
# User B cannot pause, resume, or remove User A's job (404)
check("User B gets 404 on pause", client_b.post(f"/api/jobs/{jid}/pause").status_code == 404)
check("User B gets 404 on resume", client_b.post(f"/api/jobs/{jid}/resume").status_code == 404)
check("User B gets 404 on DELETE", client_b.delete(f"/api/jobs/{jid}").status_code == 404)

# User A pauses queued job -> resumes -> removes while processing (queued/running)
r_p = client_a.post(f"/api/jobs/{jid}/pause")
check("POST /api/jobs/{jid}/pause returns 200 & paused status", r_p.status_code == 200 and r_p.json()["status"] == "paused", r_p.text)
check("POST /api/jobs/{jid}/pause twice returns 409", client_a.post(f"/api/jobs/{jid}/pause").status_code == 409)

r_res = client_a.post(f"/api/jobs/{jid}/resume")
check("POST /api/jobs/{jid}/resume returns 200 & queued status", r_res.status_code == 200 and r_res.json()["status"] == "queued", r_res.text)
jobs._q.remove(jid)

# Set status to running to verify removing a currently processing video via DELETE works directly
http_job["status"] = "running"
jobs.save(http_job)
r_del = client_a.delete(f"/api/jobs/{jid}")
check("DELETE /api/jobs/{jid} removes currently running video (200)", r_del.status_code == 200 and r_del.json().get("ok") is True, r_del.text)
check("removed processing video is gone from GET /api/jobs", all(x["id"] != jid for x in client_a.get("/api/jobs").json()))

# Live recording still refuses pause/delete with 409
check("POST pause on active live job returns 409", client_a.post(f"/api/jobs/{http_live['id']}/pause").status_code == 409)
check("DELETE on active live job returns 409", client_a.delete(f"/api/jobs/{http_live['id']}").status_code == 409)

# ---- Part C: UI checks on web/index.html
index_html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
check(
    "UI renders Pause, Resume, and Remove controls for processing/paused videos",
    all(tok in index_html for tok in ('data-act="pausejob"', 'data-act="resumejob"', 'data-act="remjob"', "/pause", "/resume", "Remove this video?")),
)

print("ALL PASSED" if not fails else f"{len(fails)} failed")
db.engine.dispose()
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
