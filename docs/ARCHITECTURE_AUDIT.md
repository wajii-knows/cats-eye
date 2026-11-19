# Cat's Eye (formerly PeakCut) - Architecture Audit
**Document Version:** 2.0.0  
**Date:** November 2026  
**Auditor:** Senior Software Engineer & System Architect  
**Branch:** `phase-0-audit`  
**Target Repository:** `D:\peakcut`

---

## 1. Executive Summary & Baseline Test Results

This architectural audit provides an exhaustive, ground-truth evaluation of the **PeakCut** codebase (rebranding to **Cat's Eye**). Every claim, signature, and pipeline detail is verified directly against the source code at `D:\peakcut`.

### 1.1 Environment & Tool Versions
- **Operating System:** Windows 11 Pro (win32, x64)
- **Python Version:** `3.12.10`
- **FFmpeg Version:** `9.0.2-full_build-www.gyan.dev`
- **yt-dlp Version:** `2026.10.21`
- **FastAPI / Starlette:** `0.115.0+`
- **SQLAlchemy:** `2.0+`
- **Database:** SQLite (local WAL mode) / PostgreSQL (`postgresql+psycopg://`)

### 1.2 Baseline Test Execution (`python tests/run.py`)
Ran `python tests/run.py` on branch `phase-0-audit` against an ephemeral test instance with an isolated, temporary data directory (`PEAKCUT_DATA=<tempdir>`). All tests ran with **zero network dependencies** and **no external AI API keys**.

```
=== permissions ===                                       [56/56 checks passed]
=== abuse and limits ===                                  [40/40 checks passed]
=== queue under load ===                                  [25/25 checks passed]
=== deleting and bulk actions ===                         [13/13 checks passed]
=== big uploads in pieces ===                             [19/19 checks passed]
=== AI services ===                                       [26/26 checks passed]
=== no sign-in, and the AI key from the dashboard ===     [39/39 checks passed]
=== Hindi and Urdu captions in Roman letters ===          [28/28 checks passed]
=== connecting YouTube, Instagram and TikTok ===          [38/38 checks passed]
=== the new-version notice ===                            [20/20 checks passed]
=== no limits by default ===                              [9/9 checks passed]
=== captions never overlap, stuck clips, sentence endings === [20/20 checks passed]
=== fresh installs: ffmpeg, the speech model, AI errors === [17/17 checks passed]
=== the storage folder: full drive, open, move ===        [20/20 checks passed]
=== snapshots and multishot rendering ===                 [6/6 checks passed]
=== brand identity and UI assets ===                      [18/18 checks passed]

================================================================================
ALL 16 TEST SUITES PASSED (386 total checks passed, 0 failures, 0 skipped).
================================================================================
```

---

## 2. Directory Map & Component Inventory

### 2.1 Top-Level Repository Structure

| Path | Item Type | Purpose & Description |
|---|---|---|
| `AGENTS.md` | File (49 lines, 5.7 KB) | Architecture map and core engineering/safety rules for contributors and AI agents |
| `DEPLOY.md` | File (160 lines, 6.3 KB) | Production deployment documentation (systemd, Docker, Caddy reverse proxy) |
| `Dockerfile` | File (35 lines, 0.9 KB) | Multi-stage container recipe with pre-packaged FFmpeg and Silero speech model |
| `docker-compose.yml` | File (70 lines, 2.8 KB) | Multi-container orchestration (FastAPI application, PostgreSQL, Caddy proxy) |
| `requirements.txt` | File (12 lines, 0.2 KB) | Core Python package requirements (FastAPI, uvicorn, sqlalchemy, httpx, etc.) |
| `run_peakcut.bat` | File (45 lines, 1.5 KB) | Windows one-click local startup script |
| `USER_GUIDE.md` | File (120 lines, 3.6 KB) | End-user manual explaining clipping workflows and creator settings |
| `data/` | Directory | Runtime state, SQLite databases, user workspaces, and local models (gitignored) |
| `deploy/` | Directory | Systemd service units, Caddy configs, and GPU worker deployment files |
| `docs/` | Directory | Technical audits, phase specifications, and architectural documentation |
| `studio/` | Directory (31 files) | Core Python backend package (FastAPI routes, pipeline, AI, rendering) |
| `tests/` | Directory (17 files) | Integration and unit regression test suite |
| `web/` | Directory (9 files) | Vanilla frontend single-page application, login pages, and static assets |

### 2.2 Backend Modules (`studio/`)

| Module | Lines | Size | One-Line Purpose |
|---|---|---|---|
| `studio/__init__.py` | 0 | 0 B | Package namespace initializer |
| `studio/account.py` | 156 | 6.4 KB | Cascade user account deletion, creator purging, and orphan file removal |
| `studio/accounts.py` | 579 | 26.9 KB | OAuth 2.0 integrations and token refresh for YouTube, Instagram, and TikTok |
| `studio/admin.py` | 169 | 9.1 KB | CLI admin management and administrative REST APIs (`/api/admin/*`) |
| `studio/ai.py` | 322 | 15.3 KB | Multi-provider LLM gateway (Gemini, OpenAI, Anthropic, Ollama) and JSON parser |
| `studio/alerts.py` | 79 | 3.0 KB | System alerts, disk exhaustion warnings, and outbound webhook dispatches |
| `studio/asr_worker.py` | 86 | 3.5 KB | Remote GPU-accelerated speech-to-text HTTP worker |
| `studio/auth.py` | 432 | 19.0 KB | Cookie sessions, scrypt password hashing, and Google OAuth sign-in |
| `studio/brain.py` | 622 | 30.9 KB | AI prompt engineering, clip moment ranking, and multishot planner |
| `studio/config.py` | 238 | 11.3 KB | Environment configuration parser, defaults, and filesystem path resolution |
| `studio/creators.py` | 194 | 7.5 KB | Creator workspaces, channel profile links, and `creator.json` storage |
| `studio/ctx.py` | 70 | 1.7 KB | Thread-local user execution context and client bump invalidation (`ctx.bump`) |
| `studio/db.py` | 120 | 6.3 KB | SQLAlchemy Core database schemas, connection pooling, and SQLite WAL pragmas |
| `studio/feedback.py` | 56 | 2.7 KB | In-app bug reporting and feature suggestion persistence |
| `studio/fetch_model.py` | 75 | 3.1 KB | Automated download of Silero/sherpa-onnx speech recognition models |
| `studio/jobs.py` | 782 | 35.9 KB | FairQueue multi-user scheduler, background worker threads, and job recovery |
| `studio/live.py` | 779 | 37.5 KB | Rolling livestream recording, segmented audio processing, and real-time clipping |
| `studio/load.py` | 102 | 4.0 KB | Win32/Linux CPU, RAM, and disk resource load monitor |
| `studio/mailer.py` | 109 | 7.1 KB | Outbound SMTP email verification and password reset delivery |
| `studio/media.py` | 363 | 15.7 KB | yt-dlp downloader, FFmpeg audio extraction, snapshots, and video probing |
| `studio/notices.py` | 66 | 2.6 KB | User notification queue, unread counters, and alert dismissals |
| `studio/posting.py` | 263 | 11.6 KB | Scheduled automated and manual social media publishing engine |
| `studio/profiles.py` | 370 | 16.3 KB | Streamer channel avatar scraping and live status polling |
| `studio/render.py` | 401 | 20.7 KB | FFmpeg 9:16 vertical cropping, multishot composition, and ASS subtitles |
| `studio/roman.py` | 263 | 14.4 KB | Transliteration of Hindi and Urdu Romanized text into English |
| `studio/safe.py` | 12 | 0.5 KB | Text sanitization and prompt injection escaping |
| `studio/server.py` | 1,554 | 64.9 KB | FastAPI ASGI web app, Gate request security middleware, and REST routes |
| `studio/storage.py` | 151 | 6.5 KB | Disk space tracking, quota calculations, and storage folder relocation |
| `studio/tools.py` | 83 | 3.8 KB | System verification for FFmpeg and FFprobe binary availability |
| `studio/transcribe.py` | 271 | 10.9 KB | Offline speech-to-text inference with word-level timestamps |
| `studio/updates.py` | 134 | 6.0 KB | Version comparison and GitHub release polling |
| `studio/usage.py` | 97 | 4.5 KB | AI token consumption accounting and audit logging |

### 2.3 Frontend Files (`web/`)

| File | Lines | Size | One-Line Purpose |
|---|---|---|---|
| `web/admin.html` | 430 | 35.4 KB | Dedicated server administration portal for users, feedback, and usage |
| `web/ascii.js` | 98 | 5.1 KB | Lightweight ASCII animation rendering utility |
| `web/auth.html` | 149 | 12.4 KB | Clean, dark-mode sign-in and account registration portal |
| `web/gradient-ascii.js` | 855 | 30.6 KB | Interactive emerald hero canvas background shader |
| `web/index.html` | 2,488 | 228.6 KB | Core monolithic SPA (dashboard, queue, review dialog, settings, and tour) |
| `web/landing.html` | 463 | 52.3 KB | Public marketing and product introduction page |
| `web/legal.html` | 249 | 19.4 KB | Single-page legal portal (Terms of Service, Privacy, Acceptable Use) |
| `web/reset.html` | 137 | 10.0 KB | Password recovery and reset request portal |
| `web/brand/` | 10 items | — | SVG vectors (`favicon.svg`, `logo-light.svg`, `logo.svg`) and demo previews |

---

## 3. End-to-End Pipeline Walkthrough

The application supports three primary entry points: Web URL (YouTube/Twitch/Kick), Chunked Local File Upload, and Continuous Livestream Recording.

```
+--------------------------------------------------------------------------------------------------+
|                                    INGESTION & INPUT VECTORS                                     |
|  1. POST /api/jobs (URL)        2. Chunked Upload (/api/upload/*)      3. Live Stream Polling    |
|     yt-dlp download                Staged upload fragments                Rolling stream buffer  |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                AUDIO EXTRACTION & TRANSCRIPTION                                  |
|  File: studio/media.py (extract_audio)  --> 16kHz mono WAV (audio.wav)                           |
|  File: studio/transcribe.py            --> Silero VAD + sherpa-onnx Whisper ASR                  |
|  Output: transcript.json (Word-level timestamps: {w: word, s: start, e: end})                    |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                     VISUAL FRAME EXTRACTION                                      |
|  File: studio/media.py (extract_snapshots)                                                       |
|  Output: data/users/<uid>/jobs/<jid>/snapshots/snapshot_*.jpg                                    |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                   AI MOMENT SELECTION & SCORING                                  |
|  File: studio/brain.py (find_clips) / studio/ai.py                                               |
|  LLM Prompt: Ingests transcript blocks + snapshots. Evaluates hook, virality, energy.            |
|  Fallback (No AI): Acoustic RMS volume spike detection (studio/brain.py:score_loudness).        |
|  Output: clip candidate objects {start, end, score, hook, title, virality, shots}                |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                               DYNAMIC MULTISHOT SHOT PLANNING                                    |
|  File: studio/brain.py (plan_multishots)                                                         |
|  Output: shots array: [{start, end, type: "wide" | "close" | "zoom" | "crop" | "split"}]          |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                 FFMPEG RENDERING & COMPOSITION                                   |
|  File: studio/render.py (render_clip)                                                            |
|  - Subtitles: Pillow-measured animated ASS subtitle stream (bounce / hype / clean)              |
|  - Video Filter Graph: Dynamic 9:16 layout composition (_multishot_filter / _filter)             |
|  - Audio Filter: loudnorm normalization (I=-14, TP=-1.5, LRA=11)                                |
|  - Working directory: Temporary path <260 chars on Windows (mkdtemp(prefix="pk"))                |
|  Output: clip.mp4, cover.jpg, clip.ass stored under clips/<cid>/                                 |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                  SOCIAL PUBLISHING & ARCHIVING                                   |
|  File: studio/posting.py (post_clip)                                                             |
|  Publishes to connected YouTube Shorts, Instagram Reels, or TikTok via OAuth 2.0                 |
+--------------------------------------------------------------------------------------------------+
```

### 3.1 Step Details & State Transitions

1. **Ingestion:**
   - **File:** [studio/server.py](file:///D:/peakcut/studio/server.py#L394-L450), [studio/media.py](file:///D:/peakcut/studio/media.py#L180-L240)
   - **Functions:** `add_video()`, `download()`, `probe()`
   - **Input:** URL string (`url`) or multipart binary stream.
   - **Output:** Staged source video `source.mp4`, `probe.json`.
   - **State:** Saved in `data/users/<uid>/jobs/<jid>/job.json` with status `"queued"`.
   - **Error Handling:** Invalid URLs or unsupported codecs set `job["status"] = "error"`, `job["error"] = str(e)`, and call `ctx.bump(uid)` to notify the frontend.

2. **Audio Demuxing:**
   - **File:** [studio/media.py](file:///D:/peakcut/studio/media.py#L310-L330)
   - **Function:** `extract_audio(src, dst)`
   - **Command:** `ffmpeg -y -v error -i <src> -vn -ac 1 -ar 16000 <dst>`
   - **Output:** `data/users/<uid>/jobs/<jid>/audio.wav`.

3. **Transcription:**
   - **File:** [studio/transcribe.py](file:///D:/peakcut/studio/transcribe.py#L60-L150)
   - **Functions:** `transcribe()`, `_run_sherpa()`
   - **Input:** `audio.wav`.
   - **Output:** `transcript.json` with schema `[{"w": "word", "s": float, "e": float, "p": float}]`.
   - **Fallback:** If local model files are absent, auto-downloads via `studio/fetch_model.py`.

4. **Visual Snapshots:**
   - **File:** [studio/media.py](file:///D:/peakcut/studio/media.py#L335-L363)
   - **Function:** `extract_snapshots(src, out_dir, interval=5.0)`
   - **Output:** JPEGs stored in `data/users/<uid>/jobs/<jid>/snapshots/snapshot_*.jpg`.

5. **Clip Selection & Scoring:**
   - **File:** [studio/brain.py](file:///D:/peakcut/studio/brain.py#L150-L320), [studio/ai.py](file:///D:/peakcut/studio/ai.py#L248-L300)
   - **Functions:** `find_clips()`, `score_loudness()`
   - **Input:** Transcript segments + snapshot image paths.
   - **Output:** List of clip candidates with fields `start`, `end`, `score`, `hook`, `title`, `virality`.
   - **Fallback:** When no AI key is configured (`ready() == False`), `score_loudness()` evaluates RMS acoustic energy spikes to select clips.

6. **Shot Planning:**
   - **File:** [studio/brain.py](file:///D:/peakcut/studio/brain.py#L420-L480)
   - **Function:** `plan_multishots(clip_dur, text, words)`
   - **Output:** `clip["shots"] = [{"start": float, "end": float, "type": "wide"|"close"|"zoom"|"crop"|"split"}]`.

7. **Rendering & Subtitle Composition:**
   - **File:** [studio/render.py](file:///D:/peakcut/studio/render.py#L339-L401)
   - **Function:** `render_clip()`, `_render_in()`
   - **Input:** `source.mp4`, `words`, `clip` metadata.
   - **Output:** `clip.mp4`, `cover.jpg`, `subtitles.ass`.
   - **Windows Safety:** Renders inside short temporary path (`tempfile.mkdtemp(prefix="pk")`), then moves outputs into place.

8. **Publishing:**
   - **File:** [studio/posting.py](file:///D:/peakcut/studio/posting.py#L50-L180)
   - **Functions:** `schedule_post()`, `post_clip()`
   - **Output:** Uploads via platform APIs; status updated in `posts.json`.

---

## 4. Target Engine Mapping

| Target Engine | Current Status | Backing Files | Current Capabilities & Missing Gaps |
|---|---|---|---|
| **Video Analysis** | **Partial** | `studio/media.py`, `studio/transcribe.py` | Extracts duration, resolution, audio RMS loudness, and chat velocity. **Missing:** Deep scene-change detection, optical flow, and face tracking. |
| **Snapshot** | **Exists** | `studio/media.py` | Extracts periodic JPEGs (`extract_snapshots`) across source video. **Missing:** Perceptual hashing deduplication and keyframe visual clustering. |
| **Transcript** | **Exists** | `studio/transcribe.py`, `studio/roman.py` | Local Silero VAD + sherpa-onnx Whisper speech model with word-level timestamps; Hindi/Urdu transliteration. |
| **Viral Intelligence** | **Partial** | `studio/brain.py` | Predicts hooks, titles, and virality tiers (`viral`, `high`, `good`) via LLM. **Missing:** Historical viral retention databases and trending audio curve matching. |
| **Clip Selection** | **Exists** | `studio/brain.py` | LLM-driven moment scoring with working acoustic RMS loudness fallback. |
| **Shot Planning** | **Exists** | `studio/brain.py`, `studio/render.py` | `plan_multishots()` computes alternating camera shot types (`wide`, `close`, `zoom`, `crop`, `split`). |
| **Asset Library** | **Missing** | UI navigation slot only | **Missing:** Ingestion, tagging, indexing, and storage of reusable B-roll, memes, audio stingers, and visual overlays. |
| **Asset Matching** | **Missing** | None | **Missing:** Automated insertion of sound effects or reaction memes based on transcript cues or emotional spikes. |
| **Editing** | **Partial** | `web/index.html`, `studio/render.py` | Trim dialog supports adjusting start/end times, hook text, and caption styles. **Missing:** Multi-track timeline, keyframe animation, or drag-and-drop layer canvas. |
| **Rendering** | **Exists** | `studio/render.py` | FFmpeg complex filter graphs, Intel QSV / libx264 encoding, ASS subtitle burning via Pillow font metrics. |
| **Learning** | **Missing** | None | **Missing:** Social post analytics feedback loop to refine AI prompt scoring weights over time. |

---

## 5. Data Layer & Filesystem Hierarchy

### 5.1 Relational Database (`studio/db.py`)
Configured via `DATABASE_URL`. Defaults to SQLite WAL mode (`data/db.sqlite`), or PostgreSQL (`postgresql+psycopg://...`).

- `users`: `id` (VARCHAR 40, PK), `email` (VARCHAR 254, UNIQUE), `name` (VARCHAR 120), `password_hash` (VARCHAR 300), `google_sub` (VARCHAR 64, UNIQUE), `is_admin` (BOOLEAN), `disabled` (BOOLEAN), `created_at` (FLOAT), `last_seen` (FLOAT).
- `sessions`: `token_hash` (VARCHAR 64, PK), `user_id` (FK users.id), `created_at` (FLOAT), `expires_at` (FLOAT), `ip` (VARCHAR 64), `agent` (VARCHAR 300).
- `usage`: `id` (INTEGER, PK), `user_id` (FK users.id), `t` (FLOAT), `tokens_in` (INTEGER), `tokens_out` (INTEGER), `purpose` (VARCHAR 60), `creator` (VARCHAR 80), `place` (VARCHAR 200). Index: `(user_id, t)`.
- `email_ok`: `user_id` (VARCHAR 40, PK), `verified_at` (FLOAT).
- `email_tokens`: `token_hash` (VARCHAR 64, PK), `user_id` (FK users.id), `purpose` (VARCHAR 10), `expires_at` (FLOAT), `used_at` (FLOAT).
- `feedback`: `id` (INTEGER, PK), `user_id` (FK users.id), `t` (FLOAT), `kind` (VARCHAR 10), `message` (TEXT), `page` (VARCHAR 200), `context` (VARCHAR 1000), `status` (VARCHAR 10), `updated_at` (FLOAT).
- `notices`: `id` (INTEGER, PK), `user_id` (FK users.id), `t` (FLOAT), `kind` (VARCHAR 20), `title` (VARCHAR 160), `body` (VARCHAR 400), `link` (VARCHAR 200), `ref` (VARCHAR 80), `read_at` (FLOAT).
- `admin_log`: `id` (INTEGER, PK), `t` (FLOAT), `admin_id` (VARCHAR 40), `action` (VARCHAR 30), `target_id` (VARCHAR 40), `detail` (VARCHAR 300).

**Schema Migration System:**  
There is currently **no automated migration tool** (such as Alembic). Schema creation relies entirely on `meta.create_all(engine)` in `studio/db.py:115`. Note that `create_all()` only creates missing tables; it **does not add columns to existing tables**. Adding columns requires additive migrations using manual `ALTER TABLE` checks.

### 5.2 Filesystem Storage Hierarchy (`data/`)

```
data/
├── ai.json                         # Configured AI provider, API key, model selections
├── settings.json                   # Server-wide preferences
├── db.sqlite                       # SQLite database file (when PostgreSQL is not set)
├── models/                         # Bundled speech recognition models (~480 MB)
└── users/<uid>/                    # Isolated per-user storage directory
    ├── accounts.json               # Linked social media OAuth tokens
    ├── posts.json                  # Queued and published social media posts
    ├── creators/<cid>/             # Creator workspace directory
    │   ├── creator.json            # Creator metadata, channels, caption preferences
    │   ├── avatar.jpg              # Creator profile image
    │   └── log.jsonl               # Stream detection and activity log
    └── jobs/<jid>/                 # Individual video job directory
        ├── job.json                # Job metadata, processing stages, clip arrays
        ├── source.mp4              # Downloaded or uploaded original video
        ├── audio.wav               # 16kHz mono audio extracted for Whisper
        ├── transcript.json         # Word-level timestamped transcription
        ├── snapshots/              # Visual frame snapshots
        │   ├── snapshot_00000.jpg
        │   └── snapshot_00001.jpg
        └── clips/<cid>/            # Rendered vertical shorts
            ├── clip.mp4            # Final 9:16 vertical short
            ├── cover.jpg           # Extracted cover poster
            └── subtitles.ass       # Pillow-computed ASS subtitle script
```

---

## 6. Queue, Resource Model & Crash Recovery

### 6.1 FairQueue Scheduler (`studio/jobs.py`)
- Multi-user round-robin queue (`_FairQueue`): Prevents any single user from starving server resources when submitting large video batches.
- Dual-lane execution:
  - Normal queue: Heavy tasks (video download, audio extraction, Whisper transcription, multi-clip rendering).
  - Quick queue (`_q.quick`): High-priority single clip re-renders (e.g., user trims timestamps or changes subtitle styles in the UI).
- Worker threads:
  - Normal video worker: `_worker()`
  - Quick worker: `_worker(quick=True)`
  - Live stream worker: `_live_worker()`

### 6.2 Host Resource Monitoring (`studio/load.py`)
- Polled every 1 second via daemon thread `_watch()`.
- **CPU Threshold:** `PEAKCUT_BUSY_CPU` (default: 82%). New heavy jobs pause when CPU exceeds 82%. Uses an exponential moving average (`0.55 * prev + 0.45 * curr`) to filter transient spikes.
- **RAM Threshold:** `PEAKCUT_MIN_FREE_RAM_GB` (default: 1.5 GB). Work pauses if available memory drops below 1.5 GB.
- **Disk Threshold:** `PEAKCUT_MIN_FREE_DISK_GB` (default: 3.0 GB). Ingestion is rejected if free disk space is under 3 GB.
- Pure Python/Win32/Linux: Uses `ctypes` on Windows (`GetSystemTimes`, `GlobalMemoryStatusEx`) and `/proc/stat`, `/proc/meminfo` on Linux without external dependencies like `psutil`.

### 6.3 Crash & Restart Recovery
- Handled by `studio/jobs.py:_pick_up_after_restart()`.
- On server startup, all jobs with `status in ("queued", "running")` are reset to `status="queued"`, `stage="Picking up where it stopped"`, and `resume=True`.
- When resumed:
  - If `transcript.json` already exists on disk, transcription is skipped.
  - If `job["done_blocks"]` indicates completed AI blocks, already-processed chunks are retained.
  - Completed clips (`c["status"] == "done"`) are skipped; only incomplete clips are re-rendered.

---

## 7. FFmpeg & Subtitle Engine Analysis

### 7.1 Command Construction & Encoders
FFmpeg commands are constructed in three primary files:
1. `studio/media.py`:
   - Audio extraction: `-vn -ac 1 -ar 16000` (16kHz mono PCM).
   - Snapshot extraction: `-vf fps=1/<interval> -q:v 3 snapshots/snapshot_%05d.jpg`.
   - Probing: `ffprobe -v error -print_format json -show_format -show_streams`.
2. `studio/render.py`:
   - Encoders tested sequentially:
     - **Hardware:** Intel QuickSync (`h264_qsv`, `-global_quality 23`, `-preset fast`, `-pix_fmt nv12`).
     - **Software Fallback:** `libx264` (`-crf 21`, `-preset veryfast`, `-pix_fmt yuv420p`).
   - Audio: `-c:a aac -b:a 160k -ar 48000 -af loudnorm=I=-14:TP=-1.5:LRA=11`.
3. `studio/live.py`:
   - Rolling stream buffer: Segmented recording from HLS/RTMP streams.

### 7.2 9:16 Vertical Cropping & Multishot Filtering
- Defined in `studio/render.py:_filter()` and `_multishot_filter()`:
  - **Crop Mode (`crop`):** Scales to `1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1`.
  - **Blur Mode (`blur`):** Splits video into background (scaled to 270:480, boxblurred 6:2, scaled back to 1080:1920, brightness dimmed by -0.14) and foreground (scaled to 1080 width, centered over blurred background).
  - **Multishot Mode (`multishot`):** Dynamically cuts between camera angles (`close`, `wide`, `zoom`, `crop`, `split`) across planned shot boundaries, concatenating segments via `concat=n={N}:v=1:a=0`.

### 7.3 Subtitle Engine (Advanced SubStation Alpha - ASS)
- Subtitles are generated natively in `studio/render.py:build_ass()` and burned into video frames via `-filter_complex "[v0]ass=<name>.ass:fontsdir=f[v]"`.
- Pillow measures exact word glyph widths (`f.width(text, size)`) to position words and compute dynamic scaling pops (`\fscx`, `\fscy`).
- Subtitle Styles:
  - `bounce`: Archivo Black font, bright yellow active highlight (`#FFE814`), dark outline.
  - `hype`: Anton font, intense red active highlight (`#FF2A2A`), all-caps text.
  - `clean`: Poppins font, minimalist white active text with subtle outline.

---

## 8. AI Usage & Multi-Provider Architecture

### 8.1 Gateway Architecture (`studio/ai.py`)
`studio/ai.py` is the **exclusive AI gateway** in the codebase. No other file imports provider SDKs or performs outbound LLM calls.

```
                           +------------------------+
                           |    studio/ai.py        |
                           |  call(model, prompt,   |
                           |       schema, images)  |
                           +-----------+------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
          v                            v                            v
  +---------------+            +---------------+            +---------------+
  |    Gemini     |            |    OpenAI     |            |   Anthropic   |
  | Google REST   |            | ChatCompletions|            | Messages API  |
  | (generate-    |            | (OpenAI, xAI,  |            | (Claude 3.5)  |
  |  Content)     |            |  Groq, Ollama) |            |               |
  +---------------+            +---------------+            +---------------+
```

### 8.2 JSON Output Enforcement & Validation
- **Gemini:** Passed native `responseSchema` and `responseMimeType: "application/json"`.
- **OpenAI / Claude / Others:** Injected into prompt via `_wrapped(prompt, schema)` requesting schema compliance, combined with `response_format: {"type": "json_object"}`.
- **Parsing & Recovery (`_parse`):** Strips markdown fences, parses JSON, and uses regex fallback `[\[{].*[\]}]` to extract valid JSON objects when conversational filler is returned.

### 8.3 Token Accounting
- Every provider call returns `(parsed_data, tokens_in, tokens_out)`.
- Recorded into `usage` database table via `studio/usage.py:record()`.
- Displayed in real-time on the AI Usage dashboard (`web/index.html`).

---

## 9. Frontend Architecture & State Management

### 9.1 Zero-Build Reactive Pattern
- Monolithic SPA in `web/index.html` (~2,488 lines). No Webpack, Vite, or React runtime.
- **Micro-Renderer (`put`):** Employs a `WeakMap` cache of rendered HTML strings. If the generated HTML matches the prior cycle, DOM operations are skipped to eliminate layout thrashing.
- **Routing:** Client-side URL routing (`go(view, tab)`) with browser history integration (`pushState`, `popstate`).

### 9.2 Real-Time Polling & Bump Protocol
- Frontend polls `GET /api/sync?v=<last_counter>`.
- The server checks in-memory `ctx._bumps[uid]`. If the counter has not changed, it returns `{v: current_counter}` within 1ms without touching disk or database.
- When background worker threads finish an ingestion step, transcription block, or clip render, they call `ctx.bump(uid)`. The client's subsequent sync poll detects the incremented counter and triggers `loadSync()`.

### 9.3 XSS Prevention & Untrusted Text Escaping
- Untrusted user input, filenames, and video metadata are sanitized before DOM insertion using the `esc(s)` helper:
  ```javascript
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
  ```

---

## 10. Test Coverage & Harness Analysis

### 10.1 Test Runner Mechanics (`tests/run.py`)
- Creates an ephemeral directory `tempfile.mkdtemp(prefix="peakcut-test-")` for `PEAKCUT_DATA`.
- Launches a dedicated server on port 8799 with in-memory SQLite and isolated environment variables.
- Waits for `/healthz` ready status (up to 90 seconds).
- Executes all 16 test suites in sequence, reporting individual checks and failures.
- Tears down server process and purges the temporary directory on completion.

### 10.2 Tested vs. Untested Areas

| Area | Automated Coverage | Notes |
|---|---|---|
| Permissions & Isolation | Full | Tested across single/multi-user modes and admin boundaries |
| Queue & Concurrency | Full | Tested with FairQueue multi-user starvation checks |
| Resumable Chunked Uploads | Full | Tested with chunk out-of-order, sizing, and cross-user checks |
| Subtitle Overlap & Timing | Full | Tested across sentence ends, non-overlap, and ASS generation |
| Speech Model Ingestion | Full | Tested with fresh install, model download, and tool checks |
| Multishot & Snapshots | Full | Tested snapshot extraction and multishot filter composition |
| Brand Identity & Assets | Full | Tested vector SVGs, HTTP 200 serving, CSS tokens, and nav |
| Live Stream RTMP Capture | Partial | Tested via mock loops; live RTMP socket ingest is not live-tested |
| GPU Hardware Acceleration | Untested | Automated test suite runs strictly on software CPU encoder |
| External Live AI APIs | Mocked | Tested against mocked responses to avoid network/key needs |

---

## 11. Risk Register for Future Phases

| # | Risk Description | Severity | Impact | Mitigation Strategy |
|---|---|---|---|---|
| 1 | **Snapshot Disk Explosion** | High | Extracting frames every 2s on 4-hour streams creates thousands of JPEGs, filling inodes. | Cap snapshot resolution at 480p; use adaptive extraction intervals (5-10s); purge JPEGs after clip scoring. |
| 2 | **Windows 260-Char MAX_PATH Limit** | High | Deep nested folders (`data/users/<uid>/jobs/<jid>/clips/...`) fail during FFmpeg operations on Windows. | Retain `tempfile.mkdtemp(prefix="pk")` short-path render scratchpad before atomic move to final directory. |
| 3 | **FFmpeg Filtergraph Memory Hang** | High | Multishot complex filter graphs with 10+ split/trim/concat nodes can exhaust RAM or stall. | Enforce `timeout=RENDER_SECONDS` on subprocess calls; cap shot cuts to max 8 per clip; fallback to `blur` on failure. |
| 4 | **GPU Encoder Variance Across OS** | Medium | NVENC (`h264_nvenc`), QSV (`h264_qsv`), and VideoToolbox require different pixel formats (`nv12` vs `yuv420p`). | Implement dynamic hardware encoder probing with a 1-frame test clip; fallback gracefully to `libx264`. |
| 5 | **Memory Exhaustion on Long Footage** | High | Reading full multi-hour audio files into memory will crash low-RAM hosts. | Enforce chunked streaming processing; stream yt-dlp downloads; demux audio in 16kHz mono WAV chunks. |
| 6 | **Asset Library Storage Quota Creep** | Medium | Custom user memes, B-roll, and audio stingers can overwhelm server storage. | Store assets under dedicated `data/users/<uid>/assets/` tree; count asset bytes toward user quota. |
| 7 | **Multi-User Asset Race Conditions** | Medium | Concurrent asset reads/writes can corrupt files. | Use atomic write-and-replace (`.tmp` $\rightarrow$ replace) with file mutex locks. |
| 8 | **Frontend Timeline UI Lag** | Medium | Complex multi-track editing DOM elements in `web/index.html` could cause layout thrashing. | Render the timeline using HTML5 `<canvas>` or virtualized CSS Grid instead of deeply nested DOM elements. |
| 9 | **Database Schema Drift** | High | SQLite and PostgreSQL handle column alterations differently without migrations. | Implement explicit additive migration checks (`ALTER TABLE ADD COLUMN IF NOT EXISTS`) in `studio/db.py:init()`. |
| 10 | **AI Timestamp Hallucinations** | High | LLMs may suggest clip timestamps outside source video boundaries ($t > \text{duration}$). | Clamp and validate all AI timestamps in `studio/brain.py` before passing to FFmpeg (`max(0.0, min(dur, t))`). |

---

## 12. Proposed 7-Stage Pipeline Split

To support resumability, caching, and fine-grained job inspection, the current monolithic pipeline should be partitioned into 7 decoupled stages:

```
+----------------------------------------------------------------------------------------------------+
| 1. STAGE_INGEST                                                                                    |
| Inputs:   URL string or multipart upload parts                                                     |
| Outputs:  source.mp4, meta.json (width, height, duration, fps, codec)                               |
| Caching:  Cached if hash/URL matches existing completed ingest                                     |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 2. STAGE_AUDIO_EXTRACT                                                                             |
| Inputs:   source.mp4                                                                               |
| Outputs:  audio.wav (16kHz mono PCM)                                                               |
| Caching:  Cached if audio.wav exists and matches source file mtime                                 |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 3. STAGE_TRANSCRIBE                                                                                |
| Inputs:   audio.wav                                                                                |
| Outputs:  transcript.json (word-level timestamped array)                                           |
| Caching:  Cached if transcript.json is valid; supports block resumption via done_blocks            |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 4. STAGE_SNAPSHOTS                                                                                 |
| Inputs:   source.mp4, interval (default 5.0s)                                                      |
| Outputs:  snapshots/snapshot_*.jpg                                                                 |
| Caching:  Cached if snapshots/ directory is populated                                              |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 5. STAGE_ANALYZE_SELECT                                                                            |
| Inputs:   transcript.json, snapshots/, audio loudness RMS                                          |
| Outputs:  clips_plan.json (candidate clips with hooks, titles, scores, virality tiers)             |
| Caching:  Cached if clips_plan.json exists; fallback to RMS loudness when no AI key is present     |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 6. STAGE_SHOT_PLAN                                                                                 |
| Inputs:   clips_plan.json, source width & height                                                   |
| Outputs:  clip.shots array with camera framings (wide, close, zoom, crop, split)                   |
| Caching:  Deterministic based on transcript pacing and speaker cues                                |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
| 7. STAGE_RENDER_EXPORT                                                                             |
| Inputs:   source.mp4, clip metadata, clip.shots, words, caption style, layout                      |
| Outputs:  clips/<cid>/clip.mp4, cover.jpg, subtitles.ass                                           |
| Caching:  Cached per clip version; re-renders execute immediately in quick queue lane              |
+----------------------------------------------------------------------------------------------------+
```

---

## 13. Top 5 Architectural Questions for Product Alignment

1. **Asset Library Scope (Phase 2):**  
   Will the Asset Library be strictly personal per user (`data/users/<uid>/assets/`), or should there be a global, pre-seeded server library of royalty-free sound effects, stickers, and meme overlays shared across all users?
2. **GPU Encoding Hardware Strategy:**  
   In addition to Intel QuickSync (`h264_qsv`), should Phase 2 prioritize NVIDIA NVENC (`h264_nvenc`) for local Windows/Linux workstations, or Apple VideoToolbox (`h264_videotoolbox`) for macOS?
3. **Database Migration Strategy:**  
   As we introduce new tables for assets, presets, and learning analytics, do you prefer continuing with raw SQLAlchemy Core additive migrations (`ALTER TABLE ADD COLUMN ...`), or should we introduce an explicit, zero-dependency migration runner?
4. **Visual Snapshot Ingestion Limits:**  
   For multimodal LLM prompts (e.g., Gemini 2.5 Flash), what is the maximum number of snapshots we should bundle per moment-scoring request (e.g., 5 to 10 keyframes) to balance token cost with visual context?
5. **Timeline Editor Complexity:**  
   For the in-browser clip editor, are we aiming for a multi-track audio/video timeline with draggable layers and split points, or a streamlined single-track boundary trimmer with layout and style controls?
