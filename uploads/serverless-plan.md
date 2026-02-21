# Plan: Serverless Transcription Pipeline

## Overview
Two-part system: a **Local Web UI** for managing files and reviewing results, and a **Serverless Bridge** that connects to RunPod's WhisperX endpoint for GPU transcription.

**Core Idea:** Your local app becomes a "control panel." It never transcribes anything itself — it sends audio to the cloud, polls for results, and displays them for review.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              YOUR LOCAL MACHINE                  │
│                                                  │
│   Browser (localhost:8000)                        │
│     ├── Drop audio file                          │
│     ├── See upload progress to S3                │
│     ├── Click "Start Transcription"              │
│     ├── See live status (queued/processing/done) │
│     └── Review finished text + audio player      │
│                                                  │
│   server.py (FastAPI)                            │
│     ├── POST /upload  → saves locally + pushes   │
│     │                   to S3 bucket             │
│     ├── POST /transcribe/{id} → sends S3 URL    │
│     │                   to RunPod serverless     │
│     ├── GET /status/{id} → polls RunPod job      │
│     │                   status, caches result    │
│     └── Serves static UI                         │
└──────────────┬───────────────┬───────────────────┘
               │               │
          S3 Upload       RunPod API
               │               │
               ▼               ▼
┌──────────────────┐  ┌────────────────────┐
│  RunPod S3       │  │  RunPod Serverless  │
│  Network Volume  │  │  WhisperX Worker    │
│                  │  │                     │
│  ez2d4o9xmt      │  │  rr2b5frt7aqoi2     │
│  Stores .m4a     │──│  Reads audio URL    │
│  files           │  │  Returns segments   │
└──────────────────┘  └────────────────────┘
```

---

## What Changes

### 1. `server.py` — Backend (modify existing)

#### A. New: S3 Upload on file drop
When user drops a file:
- Save to local `uploads/` (same as now)
- **Also upload to S3 bucket** in background
- Generate a **presigned URL** (temporary public link) so the serverless worker can access it
- Return `task_id` to frontend

#### B. New: `/transcribe/{task_id}` — calls RunPod Serverless API
Instead of starting a local thread, this endpoint now:
1. Generates a presigned S3 URL for the audio file
2. Sends a `POST` to `https://api.runpod.ai/v2/rr2b5frt7aqoi2/run` with:
   ```json
   {
     "input": {
       "audio": "https://s3api-us-wa-1.runpod.io/ez2d4o9xmt/Interview1.m4a",
       "language": "ru",
       "diarize": true,
       "align": true,
       "batch_size": 16,
       "model": "large-v2"
     }
   }
   ```
3. Stores the returned `job_id` in the task dict
4. Returns success to frontend

#### C. New: `/status/{task_id}` — polls RunPod job
Instead of reading from local `transcriptions` dict, this endpoint:
1. Checks if we already have a cached result → return it
2. Otherwise, calls `GET https://api.runpod.ai/v2/rr2b5frt7aqoi2/status/{job_id}`
3. If `status == "COMPLETED"`:
   - Parse the WhisperX response into our segment format (speaker, timestamp, text)
   - Apply our hallucination filters
   - Save `.json`, `.md`, `.docx`
   - Cache in `transcriptions` dict
4. Returns status + segments to frontend

#### D. Keep: All existing review features
- Audio player, speaker renaming, clickable timestamps — unchanged
- The UI doesn't care WHERE the transcription came from

### 2. `static/app.js` — Frontend (minor tweaks)

#### A. Upload progress
- Show a "Uploading to cloud..." status while the file is being pushed to S3
- The "Start Transcription" button becomes active only after S3 upload is confirmed

#### B. Status polling changes
- Instead of showing "Diarizing... / Transcribing...", show cloud-appropriate statuses:
  - "Uploading to cloud..."
  - "Queued (waiting for GPU)..."
  - "Transcribing on cloud GPU..."
  - "Processing results..."
  - "Done ✓"

#### C. No other UI changes needed
- The transcription display, speaker pills, audio player — all stay the same

### 3. `.env` — Credentials (new file, already gitignored)

```text
RUNPOD_ACCESS_KEY=your_s3_access_key
RUNPOD_SECRET_KEY=your_s3_secret_key
RUNPOD_API_KEY=your_runpod_api_key
```

### 4. `requirements.txt` — Add `requests` (if not present)

Already have `boto3`. Just need `requests` for the RunPod API calls.

---

## Files Changed

| File | Action | What |
|------|--------|------|
| `server.py` | **Modify** | Add S3 upload + RunPod API integration to existing endpoints |
| `static/app.js` | **Modify** | Update status messages, add upload progress |
| `static/index.html` | **No change** | Button already exists from earlier work |
| `static/style.css` | **No change** | Styling already in place |
| `.env` | **Create** | Store 3 keys (S3 access, S3 secret, RunPod API) |
| `requirements.txt` | **Modify** | Add `requests` |

---

## What We Can Remove

Since transcription no longer happens locally, we can **optionally** remove:
- `whisper` import and model loading (saves 2GB RAM on your PC)
- `pyannote` import and pipeline loading (saves another 1GB+ RAM)
- `run_live_transcription()` function
- `run_diarization()` function
- `remote_worker.py` (no longer needed)
- `cloud_sync.py` (no longer needed — S3 sync is built into the app)

Your local `server.py` becomes **lightweight** — just FastAPI + boto3 + requests. No GPU needed locally at all.

---

## User Flow (End Result)

1. Open `localhost:8000`
2. Drop an `.m4a` file
3. See "Uploading to cloud..." progress bar
4. Audio player loads immediately (you can listen while it uploads)
5. Click **"Start Transcription"**
6. See "Queued... → Transcribing... → Done ✓" (takes 3-5 min for 1hr audio)
7. Transcription appears in the right panel
8. Review, edit speakers, click timestamps — all local, all free

**Cost per 1-hour interview: ~$0.03-0.05 (serverless billing)**
**Cost while reviewing: $0.00**
