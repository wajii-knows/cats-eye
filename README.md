# Cat's Eye

Turn long videos and live streams into captioned vertical clips, ready to post. Free and open source: run it on
your own computer, server, or Google Colab, with your own AI key.

Paste a YouTube, Twitch or Kick link (or upload a file) and Cat's Eye finds the moments worth clipping, cuts them to
9:16, adds word-by-word captions and a hook, and can post them to your connected accounts. It can also record a
creator's live stream and clip the highlights while the stream is still running.

*(Built on PeakCut.)*

## What it does

- **Clips from any video.** Transcribes the audio locally with `sherpa-onnx`, asks your AI which moments would hold a
  stranger's attention, and renders each one with captions in one of three styles.
- **Resumable stage pipeline.** Stage execution (`fetch`, `probe`, `extract_audio`, `transcribe`, `select_clips`,
  `render_clip`, `post`) with content-addressed caching and typed contracts under `studio/pipeline/` and `studio/contracts.py`.
- **Google Colab + Drive source of truth.** Optional Colab mode (`PEAKCUT_COLAB=1`) isolates fast scratch state in
  `WORK_DIR`, commits verified multi-generation SQLite checkpoints and artifacts to Google Drive (`DURABLE_DIR`), and
  auto-resumes interrupted jobs after disconnects ([`docs/COLAB.md`](docs/COLAB.md)).
- **Live clipping.** Records a stream into a rolling buffer and clips as it happens, using loudness, chat activity
  and the transcript.
- **Workspaces per creator.** Channels, clip settings, posting rules and an activity log for each creator you clip.
- **Posting.** Connect YouTube, Instagram and TikTok in one click, then queue or auto-post finished clips, with
  spacing and daily caps (see [`DEPLOY.md`](DEPLOY.md#connecting-youtube-instagram-and-tiktok) for the one-time setup).
- **Your own AI.** Gemini, OpenAI, Grok, Groq, OpenRouter, Claude, a local model through Ollama, or anything that
  speaks the OpenAI chat format. No key at all also works: clips are then picked by loudness.

## Quick start (Local / Windows)

You need Python 3.10 or newer and `ffmpeg`. On Windows: `winget install Gyan.FFmpeg`. On a Mac: `brew install ffmpeg`.
On Linux: `sudo apt install ffmpeg`. Cat's Eye finds `ffmpeg` by itself and tells you in the dashboard if it is missing.

```bash
git clone https://github.com/wajii-knows/cats-eye.git cats-eye
cd cats-eye
pip install -r requirements.txt
python -m studio.server
```

On Windows, you can also double-click `run_peakcut.bat`.

The first start downloads the speech model by itself (about 480 MB, once). Open http://127.0.0.1:8765 — in default local
mode there is no sign-in, so it opens straight into the dashboard. Go to **Settings, AI**, choose your AI service,
paste your key and press **Save and test**.

## Google Colab Durable Mode

1. Open [`colab/CatsEye.ipynb`](colab/CatsEye.ipynb) in Google Colab and mount Google Drive (`/content/drive`).
2. Step 2 clones `https://github.com/wajii-knows/cats-eye.git` and installs dependencies.
3. Optional: add `PEAKCUT_OWNER_PASSWORD` and `PEAKCUT_AI_KEY` in Colab Secrets (key icon on the left).
4. Run Step 3 (`scripts/colab_bootstrap.py`) and open the printed `proxyPort(8765)` URL in the same Google account browser session.

All database snapshots (`db/gen-*.sqlite`), user metadata, rendered clips, and the cached speech model live safely in
`/content/drive/MyDrive/CatsEye` with verified SHA-256 commits and session lease protection. See [`docs/COLAB.md`](docs/COLAB.md)
and [`docs/COLAB_MANUAL_TEST.md`](docs/COLAB_MANUAL_TEST.md).

## Choose your AI

Cat's Eye needs a chat model that can return JSON. In **Settings, AI**, pick the service, paste your key, press
**Find models**, then **Save and test**. Or configure `.env`:

```ini
PEAKCUT_AI_PROVIDER=gemini
PEAKCUT_AI_MODELS=gemini-2.5-flash
PEAKCUT_AI_KEY=your-key-here
```

| `PEAKCUT_AI_PROVIDER` | Service | Where to get a key |
|---|---|---|
| `gemini` | Google Gemini | https://aistudio.google.com/apikey |
| `openai` | OpenAI | https://platform.openai.com/api-keys |
| `xai` | Grok (xAI) | https://console.x.ai |
| `groq` | Groq | https://console.groq.com/keys |
| `openrouter` | OpenRouter (many models, one key) | https://openrouter.ai/keys |
| `anthropic` | Claude | https://console.anthropic.com |
| `ollama` | A model on your own machine, no key | https://ollama.com |
| `custom` | Any OpenAI-compatible address: also set `PEAKCUT_AI_BASE_URL` | |

- Speech to text runs locally with `sherpa-onnx`, so audio never leaves your machine or Colab runtime.
- The **AI usage** page shows requests and tokens per video.

## Settings & Feature Flags

All settings use `PEAKCUT_*` environment variables (in `.env`). [`.env.example`](.env.example) documents every setting:

| Setting | What it is for |
|---|---|
| `PEAKCUT_AI_PROVIDER`, `PEAKCUT_AI_MODELS`, `PEAKCUT_AI_KEY` | The AI that picks clips and writes titles |
| `PEAKCUT_ACCOUNTS` | `1` turns on multi-user sign-in and `/admin` |
| `PEAKCUT_COLAB` | `1` enables Google Colab mode + Drive durable checkpoints |
| `PEAKCUT_OWNER_PASSWORD` | Required owner password when `PEAKCUT_COLAB=1` |
| `PEAKCUT_SNAPSHOTS`, `PEAKCUT_MULTISHOT` | Optional pipeline flags (`0` by default) |
| `PEAKCUT_DATA` | Where videos, clips and the database are kept (default `./data`) |

## Architecture

| Part | What it is |
|---|---|
| `studio/` | FastAPI server, clip pipeline, live recording, accounts, storage |
| `studio/ai.py` | Single gateway for all AI model calls |
| `studio/pipeline/` | Stage contract, DAG runner, and content-addressed artifact store |
| `studio/storage/` | Storage manager, Colab Drive durable store, DB checkpointing, lease |
| `web/` | Obsidian & Amber Gold UI (vanilla HTML/CSS/JS, no build step) |
| `colab/`, `scripts/` | Colab notebook and bootstrap script |
| `docs/` | Architecture audit, contracts, design system, and Colab guides |
| `tests/` | Automated suite (`python tests/run.py`) |

[`AGENTS.md`](AGENTS.md) is the project engineering and safety contract.

## Deploy

[`DEPLOY.md`](DEPLOY.md) walks through deploying Cat's Eye on a Linux server with Docker, PostgreSQL, and Caddy HTTPS:

```bash
docker compose up -d --build
```

## Tests

```bash
python tests/run.py
```

Starts a throwaway server on a temporary data directory, runs all test suites without external network or AI keys, and cleans up.

## Visual Understanding (`PEAKCUT_SNAPSHOTS=1`)

When `PEAKCUT_SNAPSHOTS=1` is set in `.env` (off by default), Cat's Eye exposes five cacheable visual analysis stages (`detect_scenes`, `extract_snapshots`, `analyze_snapshots`, `caption_snapshots`, `build_moments`) with Drive-packed archives and batch checkpoint resume in Colab mode, alongside:

- `POST /api/videos/{id}/analyze` — Queue or retry visual analysis for a video
- `GET /api/videos/{id}/scenes` — Paginated scenes with `method`, `layout`, `confidence`, and `facecam_box`
- `GET /api/videos/{id}/snapshots?from=&to=&limit=&cursor=` — Time-filtered, paginated snapshots with local metrics
- `GET /api/videos/{id}/snapshots/{sid}/image?size=thumb|analysis|full` — Lazy-unpacked WebP thumbnails, JPEG analysis frames, and on-demand full-resolution frames
- `GET /api/videos/{id}/moments?from=&to=` — Half-open `[start, end)` timeline moments joining transcript segments, scenes, and snapshots

## Licence

[MIT](LICENSE) — Copyright (c) 2026 PeakCut contributors; Copyright (c) 2026 Cat's Eye contributors.
Caption fonts in `studio/fonts/` (Anton, Archivo Black, Poppins) and bundled UI fonts in `web/fonts/` (Inter, Space Grotesk, JetBrains Mono) are licensed under the SIL Open Font License; licence texts sit beside them.
