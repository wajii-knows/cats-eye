# Cat's Eye: Google Colab Manual Verification Guide

Follow this guide to verify that Cat's Eye runs smoothly in Google Colab, persists all state to Google Drive, and recovers automatically after sudden runtime disconnections.

---

## Prerequisites
- A Google account with Google Colab access (free tier or Colab Pro).
- At least 3 GB of free storage in your Google Drive.
- (Optional) An AI provider API key (Gemini, OpenAI, Groq, etc.) configured in Colab Secrets or added in Settings.

---

## Step 1: Open the Colab Notebook

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Choose **Upload** and upload `colab/CatsEye.ipynb` from the repository.
3. In the Colab menu, go to **Runtime** → **Change runtime type** → select **T4 GPU** (or CPU if GPU is unavailable).

---

## Step 2: Configure Secrets & Mount Drive

1. Click the **Key icon** (Secrets) in the left sidebar:
   - Add Secret: `PEAKCUT_OWNER_PASSWORD` → set to your desired password (e.g. `mypassword123`).
   - (Optional) Add Secret: `PEAKCUT_AI_KEY` → set to your Gemini or OpenAI API key.
2. Run **Cell 1 & 2** to mount your Google Drive:
   - When prompted, grant Colab permission to connect to your Google Drive.
   - Verify that `/content/drive/MyDrive` is accessible.

---

## Step 3: Launch Cat's Eye

1. Run **Cell 3 & 4**:
   - Dependencies will install automatically.
   - You will see console output showing:
     ```
     ========================================================
       CAT'S EYE: GOOGLE COLAB RUNTIME BOOTSTRAP
     ========================================================
       Google Drive durable store: /content/drive/MyDrive/CatsEye (12.4 GB free)
       Fast local work store:      /content/cats_eye_work
       Acquiring session lease and verifying durable checkpoints...
       >>> Clean database initialized. Ready for new jobs.
       Starting Cat's Eye web server on http://127.0.0.1:8765 ...
     ```
   - The cell will print a clickable port proxy link:
     ```
     🐾 CLICK TO OPEN CAT'S EYE IN YOUR BROWSER:
     https://<unique-id>-8765.colab.googleusercontent.com/
     ```

---

## Step 4: Sign in & Test Web UI

1. Click the proxy link to open Cat's Eye in your browser.
2. You will see the **Colab Owner Access** screen.
3. Enter your owner password and press **Unlock Cat's Eye**.
4. You will arrive at the Cat's Eye dashboard.
5. In the top bar, observe the **Cloud Sync chip**:
   - Displays a green dot and **"Synced"**.
   - Click the chip to open the **Google Drive Cloud Sync** modal:
     - Check Google Drive free space (GB).
     - Check verified database generation count.
     - Click **"Checkpoint now"** and verify that a new verified DB generation is created.

---

## Step 5: Start a Video Job

1. On the home page, paste a video URL (e.g. a YouTube video) or upload a test video.
2. Click **Make clips**.
3. Observe the progress:
   - Audio extraction → Transcription chunks → AI clip selection.
   - Notice the top bar chip briefly changes to **"Syncing X MB"** (amber spinner) and returns to **"Synced"** once durable commits complete.

---

## Step 6: Simulate Sudden Colab Disconnection (Crash Test)

1. While the job is running (or after clips are rendered), simulate a total session wipe:
   - Go to Google Colab menu: **Runtime** → **Disconnect and delete runtime**.
   - Confirm deletion.
   - Notice: **All local disk files (`/content/cats_eye_work`) are now completely destroyed.**

---

## Step 7: Reconnect & Verify 100% Automatic Recovery

1. In the same notebook, run **Step 1, 2, and 3** again.
2. Watch the bootstrap console output:
   ```
   Acquiring session lease and verifying durable checkpoints...
   >>> CHECKPOINT RESTORED: Generation 3 verified.
   >>> Auto-resuming 1 unfinished video job(s).
   ```
3. Open the new proxy link in your browser:
   - Log in with your owner password.
   - Observe the amber banner at the top of the dashboard:
     **"Checkpointed: Recovered from Drive checkpoint at 14:35:12 (Gen 3). Resuming 1 job."**
   - Click **Videos**:
     - The interrupted job is resumed automatically from its last completed stage/chunk!
     - No clips are duplicated, and no state is corrupted.

Congratulations! The Google Colab durable sync and recovery pipeline is fully verified.
