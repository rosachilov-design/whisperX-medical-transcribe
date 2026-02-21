# Transcriber Pro — Pod Workflow

## Architecture
```
Your PC (Free)                    RunPod Pod (Pay per minute)
┌──────────────┐     S3 Sync      ┌──────────────────────┐
│ server.py    │ ───────────────▶ │ remote_worker.py     │
│ (Dashboard)  │                  │ (GPU Engine)          │
│              │ ◀─────────────── │                      │
│ Review text  │     S3 Sync      │ Whisper + Diarization│
│ Edit speakers│                  │ Saves .json results  │
│ Play audio   │                  └──────────────────────┘
│ Save .md/.docx                   
└──────────────┘                   
```

## File Structure
```
transcriber/
├── server.py          ← Local dashboard (no GPU needed)
├── engine.py          ← GPU transcription engine (runs on Pod only)
├── remote_worker.py   ← Headless worker script (runs on Pod only)
├── cloud_sync.py      ← S3 upload/download bridge
├── .env               ← Your RunPod API keys (gitignored)
├── uploads/           ← Audio files + JSON results
├── static/            ← Web UI files
└── requirements.txt   ← Lightweight local deps only
```

## Step-by-Step Workflow

### 1. Upload Audio to Cloud (Pod OFF — $0)
```bash
python cloud_sync.py
```
This pushes any new `.m4a` / `.mp3` / `.wav` files from your local `uploads/` folder to the RunPod Network Volume via S3.

### 2. Start Pod & Transcribe (~$0.70/hr, takes 5 min)
1. Go to RunPod Dashboard → Deploy a Pod (RTX 4090 or similar)
2. Attach your Network Volume (`ez2d4o9xmt`)
3. SSH into the Pod
4. Run:
```bash
cd /workspace
git clone https://github.com/YOUR_REPO/transcriber.git  # (first time only)
cd transcriber
pip install -r requirements_gpu.txt                       # (first time only)
python remote_worker.py
```
5. Wait for "All done" message
6. **STOP the Pod immediately**

### 3. Download Results (Pod OFF — $0)
```bash
python cloud_sync.py
```
This pulls the `.json`, `.md`, and `.docx` files back to your local `uploads/` folder.

### 4. Review Locally (Forever — $0)
```bash
python server.py
```
Open `http://localhost:8000` in your browser:
- Drop the audio file → transcription loads automatically from the .json
- Play audio with synced timeline
- Click timestamps to jump
- Rename speakers with the pill system
- Download as .md or .docx

## Tips & Troubleshooting

### 1. Force Re-transcribe
If you want to re-run a file (e.g., to use the new Large-v3 model), you must delete the old JSON result on the Pod:
```bash
rm uploads/Interview1.json
python remote_worker.py
```

### 2. Auto-moving files
The worker now automatically looks for files in `/workspace/` and moves them into `transcriber/uploads/`. If your files are missing, just run the worker and it will find them.

### 3. Session Reliability
Use `screen` to keep your worker alive if your terminal disconnects:
```bash
apt update && apt install -y screen
screen -S transcriber
python remote_worker.py
# (Use Ctrl+A then D to detach)
```
