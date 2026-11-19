# PeakCut User Guide (with Video Snapshots & Multishot System)

This modified version of **PeakCut** includes two major upgrades:
1. **Periodic Video Snapshots for AI Vision**: The AI no longer relies solely on audio/text transcripts. It inspects periodic video frames to identify visual action, facial reactions, gameplay moments, and physical comedy.
2. **Dynamic Multishots System**: Clips are no longer rendered as a single static angle for the entire duration. The editor cuts between multiple framing angles (`wide`, `close` punch-ins, `crop`, `zoom`, `split`) every 3–5 seconds on speech beats and punchlines.

---

## 1. Quick Start (How to Run)

### Method A: One-Click Launcher (Recommended on Windows)
Double-click:
```
D:\peakcut\run_peakcut.bat
```
This automatically starts the PeakCut server and opens your web browser to **http://127.0.0.1:8765**.

### Method B: Terminal / Command Prompt
Open PowerShell or Command Prompt:
```powershell
cd D:\peakcut
python -m studio.server
```
Then navigate to **http://127.0.0.1:8765** in any browser.

---

## 2. How to Clip Videos

1. **Paste a Video Link or Upload a File**:
   - Paste a YouTube, Twitch, or Kick URL, OR drag and drop an `.mp4` / `.mov` file into the upload box.
2. **Select Layout (Multishot is Default)**:
   - Under **Options**, the layout is set to **`Multishot (dynamic cuts)`** by default.
   - If desired, you can still choose classic layouts:
     - `Multishot (dynamic cuts)`: Multiple camera angles switching every 3–5s
     - `Full frame + blurred fill`: Single wide frame with blurred background
     - `Fill the screen (centre crop)`: Full vertical 9:16 crop
     - `Zoomed in + blurred fill`: Single zoomed-in shot
3. **Set Clip Length & Preferences**:
   - Shortest clip (e.g. 15s) and Longest clip (e.g. 60s).
   - Caption style: `Bounce`, `Hype`, or `Clean`.
4. **Click "Find Clips"**:
   - PeakCut transcribes the speech.
   - PeakCut extracts periodic video snapshots (~every 4s).
   - Gemini evaluates both visual snapshots and the transcript to select the highest-energy moments.
   - Clips are rendered with dynamic shot cuts and word-by-word animated subtitles.

---

## 3. How the Multishot System Works

Each clip generated contains an internal shot timeline:
- **`wide`**: Establishing shot showing full frame with blurred background (great for setup and context).
- **`close`**: Punch-in close-up shot focused tightly on the speaker/action (great for punchlines, comedic expressions, emotional reactions).
- **`crop`**: Vertical 9:16 fill (tight gameplay or mobile action).
- **`zoom`**: Medium zoom.
- **`split`**: Stacked two-shot layout.

### Inspecting and Editing Shots:
1. Click on any finished clip in the dashboard to open the **Clip Editor**.
2. Under **On the video**, you will see the **Shots breakdown** (e.g., `wide (3.5s) → close (4.0s) → zoom (3.8s) → close (4.2s)`).
3. You can edit timestamps, caption style, or switch layouts at any time and click **Save and re-make clip**.

---

## 4. AI Configuration

Your AI settings are stored locally in `D:\peakcut\data\ai.json`.
- Current Provider: **Google Gemini** (`gemini-3.5-flash-lite`)
- You can change models or providers at any time in **Settings → AI** directly from the dashboard.

---

## 5. Where Files Are Saved

- **Clips and Videos**: `D:\peakcut\data\users\u_local\jobs\`
- **Exported Clips**: In each job's `clips/` subfolder (`.mp4`, `.jpg` cover, `.ass` subtitles).
- **Snapshots**: Extracted thumbnail snapshots used by AI are cached under each job's `snapshots/` folder.
