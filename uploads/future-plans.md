# Future Development Plans


## ⚡ High-Speed Infrastructure
- **Cloud Migration (RunPod/Lambda)**: Move the server to high-end GPUs like **RTX 4090** or **A6000** ($0.70/hr). Reduces 90-minute transcription from 45 mins to ~5 mins.
- **WhisperX Architecture**: Replace standard Whisper with WhisperX for batched, parallel transcription and VAD-based segment alignment.

## 🛡️ Robust Hallucination Suppression
- **Prompt Sanitization**: Automatically strip detected hallucinations from the `initial_prompt` to prevent "poisoning" subsequent chunks.
- **Silero VAD Integration**: Use aggressive Voice Activity Detection to cut audio gaps entirely, preventing Whisper from "hearing" subtitle credits in background noise.

## 🤖 Advanced AI Post-Processing
- **LLM Refinement (Llama/GPT)**: Implement a final "Reviewer" pass using a secondary LLM. 
    - **Goal**: Surgical removal of all remaining hallucinations (credits, "like/subscribe").
    - **Improvement**: Correct medical/technical terminology in Russian.
    - **Summary**: Automatically generate a concise summary of the transcribed conversation.

## 🛠 UX & Interface
- **Video Integration**: Synced video player that tracks with transcription segments.
- **Keyboard Shortcuts**: Hotkeys for Play/Pause, Skip, and Speed control.
- **Search & Replace**: Global search and replace within the transcription panel.

---

## 🚀 RunPod Environment Setup (Broad Strokes)

### 1. Create a Pod
- Go to [runpod.io](https://runpod.io) → **GPU Cloud** → **Deploy**
- Pick a GPU template: **RTX A6000** (~$0.49/hr) or **RTX 6000 Ada** (~$0.77/hr)
- Use the **PyTorch 2.x** template (comes with CUDA, Python, torch pre-installed)
- Set **Container Disk** to at least 20 GB (models are large)
- Set **Volume Disk** to 20+ GB if you want persistent storage across restarts
- Under **Expose HTTP Ports**, add **8000** (our server port)

### 2. Connect & Install System Dependencies
SSH in or use the RunPod web terminal:
```bash
apt update && apt install -y ffmpeg
```

### 3. Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/transcriber.git
cd transcriber
```

### 4. Install Python Dependencies
```bash
pip install openai-whisper fastapi uvicorn python-multipart python-docx pyannote.audio soundfile
```
> **Note:** `torch` is already on the RunPod PyTorch template. Don't reinstall it — it'll break CUDA.

### 5. Hugging Face Token
The server uses `pyannote/speaker-diarization-3.1`, which requires a HF token with access granted.
- Make sure you've accepted the model terms on huggingface.co
- Token is already hardcoded in `server.py` (consider moving to env var for production)

### 6. Run the Server
```bash
python server.py
```
Server starts on `0.0.0.0:8000`. Use the RunPod **proxy URL** to access from your browser.

### 7. (Optional) Keep it Running
```bash
nohup python server.py > server.log 2>&1 &
```
This keeps the server alive if you close the terminal. Check logs with `tail -f server.log`.

### Quick Checklist
- [ ] Pod created with PyTorch template + exposed port 8000
- [ ] `ffmpeg` installed
- [ ] Repo cloned
- [ ] Python deps installed (without reinstalling torch)
- [ ] HF token has access to pyannote models
- [ ] `python server.py` runs and shows "Whisper loaded. Diarization Pipeline loaded."
- [ ] Access the web UI via RunPod's proxy URL
