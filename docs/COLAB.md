# Cat's Eye on Google Colab Architecture & Recovery Specification

## 1. Executive Summary & Goals

Cat's Eye runs seamlessly in Google Colab, leveraging free or Pro GPU runtimes while using **Google Drive as the durable source of truth**.

Because Google Colab sessions can terminate abruptly (runtime recycling, idle timeouts, maximum 12/24 hour limits, or browser tab closure), **Cat's Eye guarantees automatic recovery with zero data loss, no duplicate clips, and no corrupted SQLite databases**.

---

## 2. Architecture: Disposable Compute vs. Durable Storage

Running live databases directly on Google Drive via FUSE mount (`/content/drive`) causes severe I/O latency, lock failures, and corrupted WAL logs.

Cat's Eye solves this with strict compute-storage isolation:

```
+-----------------------------------------------------------------------------------------+
|                                GOOGLE COLAB INSTANCE                                    |
|                                                                                         |
|   Fast Local NVMe Disk (WORK_DIR = /content/cats_eye_work)                              |
|   +---------------------------------------------------------------------------------+   |
|   | - Live SQLite Database: peakcut.db (WAL mode, fast local I/O)                   |   |
|   | - Raw working files: audio.wav, video parts, unpacked stage frames              |   |
|   | - Cached models: Parakeet 0.6b ASR (local RAM/GPU)                              |   |
|   | - Sensitive local files: ai.json (secrets NEVER committed to Drive)             |   |
|   +---------------------------------------------------------------------------------+   |
|                                          |                                              |
|                                          | Sync / Checkpoint protocol                   |
|                                          v                                              |
|   Google Drive FUSE Mount (DURABLE_DIR = /content/drive/MyDrive/CatsEye)                |
|   +---------------------------------------------------------------------------------+   |
|   | - lease.json (Concurrency lock, 30s heartbeat, 3m timeout)                      |   |
|   | - manifest.json (Verified atomic file index with SHA-256)                       |   |
|   | - db/gen-{1..5}.sqlite (Verified online SQLite backups, sanitized)              |   |
|   | - artifacts/{stage}/{fp[:2]}/{fp}/{fp}.tar (Packed artifact archives)           |   |
|   | - users/{uid}/jobs/{jid}/clips/*.mp4 (Finished durable clip renders)            |   |
|   | - models/ (Cached model files to eliminate reconnect downloads)                 |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Recovery Point Objectives (RPO)

| Subsystem | RPO / Maximum Loss | Mechanism |
|---|---|---|
| **Database State** | **≤ 30 seconds** | SQLite online backup API (`sqlite3.backup`) debounced timer & per-stage trigger. |
| **Audio Transcription** | **≤ 5 minutes** | Audio chunked in 5-minute blocks; partial transcript chunks committed to `StageContext.checkpoint()`. |
| **Rendered Clips** | **0 seconds (Instant)** | Clips uploaded to Drive immediately upon render completion before marking job state. |
| **Intermediate Artifacts** | **Stage boundary** | Packed `.tar` archives committed to durable store; cache miss checks Drive before recomputing. |

---

## 4. The Verified Write Protocol

Writing directly to Google Drive FUSE is prone to silent truncation under network drops or abrupt notebook termination. Cat's Eye enforces a strict 5-step write protocol:

1. **Unique Temp File**: Write to `<target>.tmp.<session_uuid>`.
2. **Flush and Fsync**: Force all OS and FUSE buffers to storage.
3. **Read-back Verification**: Re-read file from disk, calculate SHA-256 hash, and check exact byte size.
4. **Atomic Rename**: Rename `<target>.tmp.<session_uuid>` to `<target>`.
5. **Manifest Commit**: Append entry to `manifest.json`.

> **Absence Rule**: Any file present in Drive that is **not** indexed in `manifest.json` is treated as absent and will never be read as a checkpoint.

---

## 5. Multi-Generation SQLite Snapshot Management

Cat's Eye uses `DatabaseCheckpointManager` (`studio/storage/db_checkpoint.py`):

1. **Non-blocking Backup**: Uses Python's `sqlite3.Connection.backup()` to take a snapshot without locking the active database or copying volatile `-wal` / `-shm` files.
2. **Integrity Verification**: Executes `PRAGMA integrity_check` on the backup file. If integrity fails, the backup is discarded.
3. **Session Sanitization**: Prior to cloud upload, the `sessions` table is emptied in the snapshot so stolen Drive files cannot authenticate requests.
4. **Generations & Fallback**: Retains the last 5 verified database generations (`gen-1.sqlite` .. `gen-5.sqlite`). If the latest generation is unreadable or fails verification during restore, it automatically falls back to `gen-4`, etc.

---

## 6. Concurrency & Session Lease Protection

To prevent multiple running Colab instances (e.g. from multiple browser tabs or accidental multiple runtimes) from corrupting the Drive store:

- `lease.json` is placed at the root of `DURABLE_DIR`.
- Contains: `session_id`, `created_at`, `last_heartbeat`, `hostname`.
- A background daemon updates `last_heartbeat` every 30 seconds.
- If another notebook starts and the existing lease heartbeat is `< 3 minutes` old, it raises `ActiveLeaseError` and refuses to start.
- If the lease is `> 3 minutes` old (stale from an abandoned session), the new session automatically takes over ownership.
- Force takeover is supported via `--force-lease` / `PEAKCUT_FORCE_LEASE=1`.

---

## 7. Web Access & Google Colab Terms of Service Analysis

### Official Solution: `google.colab.kernel.proxyPort(8765)`
Google Colab's Free-Tier Terms of Service explicitly prohibit unauthorized proxies and public tunnels (such as `cloudflared` and `ngrok`), using automated heuristic abuse detection that can terminate notebooks or ban user accounts.

**Cat's Eye uses Google Colab's native authorized proxy API:**
```python
from google.colab.output import eval_js
proxy_url = eval_js("google.colab.kernel.proxyPort(8765)")
```

### Advantages of `proxyPort`:
1. **100% Colab ToS Compliant**: Authorized Google infrastructure; zero risk of account ban.
2. **Authenticated by Google**: Requests are routed through the user's logged-in Google session.
3. **No External Dependencies**: Requires no third-party binaries or API accounts.
4. **Single-Owner Protection**: Gated additionally by `PEAKCUT_OWNER_PASSWORD`.

*(For Colab Pro+ or private VMs, an optional cloudflared tunnel remains supported by passing `--tunnel cloudflared`).*

---

## 8. Security Model

1. **Owner Password**: When `PEAKCUT_COLAB=1`, the server strictly refuses to start without `PEAKCUT_OWNER_PASSWORD`.
2. **Rate Limiting**: Brute-force attempts against the owner login endpoint are throttled using `auth.throttle`.
3. **Zero Secrets in Cloud**: API keys configured locally (`ai.json`) are stored only on the disposable `WORK_DIR`. Users configure their API key via Colab Secrets or the web settings panel.
4. **Cross-Site & Origin Protection**: All requests validate session tokens using HTTP-only cookies and SameSite restrictions.
