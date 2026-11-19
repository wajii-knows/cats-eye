# Working on PeakCut

A short map for contributors and AI coding assistants.

## Run and test

- Run: `python -m studio.server` (settings from `.env`; see `.env.example`). Needs Python 3.10+ and `ffmpeg`.
- The speech model (about 480 MB) downloads by itself on first start; `python -m studio.fetch_model` gets it ahead of time.
- Tests: `python tests/run.py`. It starts a throwaway copy on a spare port with an empty data folder. Run it
  before and after any change to `studio/`. No keys needed.
- To try a risky change by hand without touching real data:
  `PEAKCUT_DATA=<temp folder> PEAKCUT_PORT=8766 python -m studio.server`.

## Where things are

| File | What it does |
|---|---|
| `studio/server.py` | Every web route, and the gate that runs before each request (who is asking, limits, headers) |
| `studio/config.py` | Every setting, read from environment variables |
| `studio/ai.py` | The only code that talks to an AI service, and the AI settings saved from the dashboard (`ai.json` in the data folder) |
| `studio/brain.py` | The prompts: finding clips, live judging, hooks and titles, caption fixes |
| `studio/usage.py` | The record of AI calls and tokens shown on the AI usage page |
| `studio/jobs.py` | The video pipeline and the fair queue |
| `studio/live.py` | Live recording and clipping |
| `studio/transcribe.py`, `studio/asr_worker.py` | Speech to text, here or on a second machine |
| `studio/render.py` | Cutting, layouts and captions (ffmpeg and Pillow) |
| `studio/media.py` | Downloading (yt-dlp), probing, audio |
| `studio/auth.py`, `studio/account.py`, `studio/admin.py` | Sign-in, deleting things, the admin dashboard |
| `studio/db.py` | Database tables (SQLAlchemy core; SQLite or PostgreSQL) |
| `web/index.html` | The whole signed-in app, one file, no build step |

## Rules worth knowing

- **Sign-in is off by default** (`config.ACCOUNTS`): every request then belongs to one built-in owner
  (`auth.local_user()`). With `PEAKCUT_ACCOUNTS=1` it is a normal multi-user site. Both paths are tested.
- **Who is asking** always comes from the session (`ctx.uid()`), never from the request body. Background threads
  set it themselves with `ctx.as_user(...)`.
- **Anything that changes what a user sees must call `ctx.bump()`**. The page polls `/api/sync`, which answers
  from memory when that user's counter has not moved.
- **Text from videos is untrusted**: it goes into prompts through `brain._safe` and out through `brain._text`.
- **Adding an AI provider**: add a row to `PROVIDERS` in `studio/ai.py`. If it speaks the OpenAI chat format that
  is all; otherwise add one function that returns `(parsed JSON, tokens in, tokens out)`.
- **Clips are rendered in a short temporary folder** and moved into place (`render.render_clip`): ffmpeg's caption
  step cannot open paths longer than 260 characters on Windows.
- **Secrets** are read from the environment or from files named by `*_FILE` settings. Never write a key, a real
  address or a personal path into a tracked file.
- **Pages** are plain HTML, CSS and JavaScript. Keep the existing palette, no shadows, and hover states that
  change colour only.
- Comments and messages are written in plain language for people who do not program. Keep that tone.

---

## PROJECT RULES: apply to every task

### STACK
- Python 3.10+, FastAPI, vanilla HTML/CSS/JavaScript with no build step, ffmpeg, yt-dlp, sherpa-onnx, SQLite (local) and PostgreSQL (server).
- No new framework or heavy dependency without asking. If you want one, write a short justification (what, size, license, alternative considered) and wait.

### AI USAGE
- All model calls go through `studio/ai.py`. No other file may import a provider SDK or call a provider URL.
- Every AI-powered feature must have a working no-AI fallback. The product already works with no API key and must keep doing so.
- Model output is DATA. It is validated against a schema and never interpolated into shell commands, ffmpeg arguments, file paths or SQL.
- Provider capabilities (JSON mode, image input, context size) are looked up, never assumed.

### SAFETY AND ISOLATION
- User identity comes from the session only, never from request bodies or paths.
- Every file path derived from user input is normalized and checked to stay inside the user's data folder.
- Uploads: validate type with ffprobe/magic bytes, cap size, sanitize names.
- Text from videos, transcripts and filenames is untrusted when shown in the UI (escape it) and when sent to an AI (treated as data).

### CHANGE DISCIPLINE
- Every new feature sits behind a flag documented in `.env.example`, default OFF until its phase is merged and verified.
- DB changes are additive migrations that work on SQLite and PostgreSQL. Never drop or rewrite existing columns.
- The existing pipeline keeps working untouched until a replacement is proven side by side.
- No drive-by refactors. Unrelated cleanup goes in a separate commit or is listed under known gaps.

### CONVENTIONS
- All timestamps are float seconds in source-video time, handled by one helper module (`studio/timeutil.py` or the audit's equivalent). Never store frame numbers as the primary time.
- Heavy work goes through the existing queue and respects its CPU/RAM/disk checks.
- Long videos: stream and chunk, never load all frames or the full audio into memory.
- Logging is structured and never contains API keys, tokens or full transcripts.

### DEFINITION OF DONE
- `python tests/run.py` passes with no API keys and no network.
- New behavior has tests, including at least one failure-path test.
- `.env.example` and README sections updated for new flags.
- Final message contains: summary, files changed, how to test manually, known gaps, decisions made.
