# 🚀 New Flow: RunPod Serverless WhisperX + Diarization

> **Goal**: Zero-setup transcription of Russian medical interviews with speaker diarization.
> **GPU Target**: RTX 6000 Ada (48 GB VRAM)
> **Stack**: WhisperX · Pyannote 3.1 · faster-whisper · RunPod Serverless

---

## Why This Flow

| Old (Pod)                      | New (Serverless)                      |
|--------------------------------|---------------------------------------|
| Manual SSH key setup           | Send HTTP request, get JSON back      |
| Wake/Stop Pod buttons          | Auto-scales, pay per second           |
| Vanilla Whisper (slow, drifts) | WhisperX (10x speed, aligned words)   |
| Glued-on diarization           | Native WhisperX diarization pipeline  |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- [DockerHub](https://hub.docker.com/) account (free)
- [RunPod](https://runpod.io) account with credits
- [HuggingFace](https://huggingface.co/) account + access token (for Pyannote models)
  - Accept the terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
  - Accept the terms at: https://huggingface.co/pyannote/segmentation-3.0

---

## Step 1 — Create the Repo

```
mkdir whisperx-serverless && cd whisperx-serverless
```

Create these 3 files:

### 1.1 `Dockerfile`

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/m-bain/whisperx.git \
    pyannote.audio \
    faster-whisper

# Pre-download models into the image (critical for fast cold starts)
# WhisperX large-v3
ARG HF_TOKEN
RUN python -c "\
import whisperx; \
whisperx.load_model('large-v3', 'cpu', compute_type='float32', download_root='/app/models')"

# Pyannote diarization (requires HF token at build time)
RUN python -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', \
    use_auth_token='${HF_TOKEN}', cache_dir='/app/models')"

# Russian alignment model
RUN python -c "\
import whisperx; \
whisperx.load_align_model(language_code='ru', device='cpu', model_dir='/app/models')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
```

> [!IMPORTANT]
> You must pass your HuggingFace token at build time:
> `docker build --build-arg HF_TOKEN=hf_XXXX -t yourdockerhub/whisperx-serverless .`

### 1.2 `handler.py`

```python
import runpod
import whisperx
import torch
import gc
import os
import tempfile
import requests

# ─── Config ───
DEVICE = "cuda"
BATCH_SIZE = 32          # RTX 6000 Ada can handle 32+ easily with 48GB VRAM
COMPUTE_TYPE = "float16" # Native Ada support
MODEL_DIR = "/app/models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Load models once (stays in GPU memory between requests) ───
print("Loading WhisperX model...")
model = whisperx.load_model(
    "large-v3", DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_DIR
)
print("✅ WhisperX ready.")


def download_audio(url: str) -> str:
    """Download audio from URL to a temp file."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    ext = url.split('.')[-1].split('?')[0] or 'wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def handler(job):
    """
    Input:
      - audio_url: URL to audio file (S3 presigned, public link, etc.)
      - language:  Language code (default: "ru")
      - hf_token:  HuggingFace token for diarization (or set via env)
      - min_speakers / max_speakers: Optional speaker count hints
    """
    inp = job["input"]
    audio_url = inp["audio_url"]
    language = inp.get("language", "ru")
    hf_token = inp.get("hf_token", HF_TOKEN)
    min_speakers = inp.get("min_speakers", None)
    max_speakers = inp.get("max_speakers", None)

    # 1. Download audio
    audio_path = download_audio(audio_url)
    audio = whisperx.load_audio(audio_path)

    # 2. Transcribe
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)

    # 3. Align (word-level timestamps)
    model_a, metadata = whisperx.load_align_model(
        language_code=language, device=DEVICE, model_dir=MODEL_DIR
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False
    )
    # Free alignment model memory
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Diarize (speaker identification)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=DEVICE
    )
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Clean up temp file
    os.unlink(audio_path)

    # 6. Format output
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "speaker": seg.get("speaker", "Unknown"),
            "text": seg["text"].strip(),
        })

    return {
        "language": language,
        "segments": segments,
    }


runpod.serverless.start({"handler": handler})
```

### 1.3 `.dockerignore`

```
.git
.env
__pycache__
*.pyc
```

---

## Step 2 — Build & Push Docker Image

```bash
# Login to DockerHub
docker login

# Build (pass your HuggingFace token for model download)
docker build \
  --build-arg HF_TOKEN=hf_YOUR_TOKEN_HERE \
  -t yourdockerhub/whisperx-serverless:latest .

# Push
docker push yourdockerhub/whisperx-serverless:latest
```

> [!TIP]
> The build downloads ~8 GB of models. First build takes 10–15 min.
> Subsequent builds are cached and take <1 min.

---

## Step 3 — Create RunPod Serverless Endpoint

1. Go to **[runpod.io/console/serverless](https://www.runpod.io/console/serverless)**
2. Click **"+ New Endpoint"**
3. Fill in:

| Field              | Value                                          |
|--------------------|-------------------------------------------------|
| **Endpoint Name**  | `whisperx-russian`                              |
| **Docker Image**   | `yourdockerhub/whisperx-serverless:latest`      |
| **GPU**            | Select **RTX 6000 Ada** (or equivalent 48GB)    |
| **Min Workers**    | `0` (scale to zero when idle)                   |
| **Max Workers**    | `1` (or more if you need parallel processing)   |
| **Idle Timeout**   | `60` seconds                                    |
| **Container Disk** | `20 GB`                                         |

4. Under **Environment Variables**, add:

| Key        | Value               |
|------------|----------------------|
| `HF_TOKEN` | `hf_YOUR_TOKEN_HERE` |

5. Click **Create** → copy the **Endpoint ID** (e.g. `abc123def456`)

---

## Step 4 — Test the Endpoint

### Option A: cURL

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://your-s3-bucket.com/interview.m4a",
      "language": "ru",
      "min_speakers": 2,
      "max_speakers": 3
    }
  }'
```

### Option B: Python

```python
import requests

ENDPOINT_ID = "YOUR_ENDPOINT_ID"
API_KEY = "YOUR_RUNPOD_API_KEY"

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "audio_url": "https://your-s3-bucket.com/interview.m4a",
            "language": "ru",
            "min_speakers": 2,
            "max_speakers": 3,
        }
    },
    timeout=300,
)

data = response.json()
for seg in data["output"]["segments"]:
    print(f"[{seg['start']:.1f}s] {seg['speaker']}: {seg['text']}")
```

### Expected Output

```json
{
  "output": {
    "language": "ru",
    "segments": [
      {"start": 0.0,  "end": 3.42, "speaker": "SPEAKER_00", "text": "Здравствуйте, расскажите о ваших жалобах."},
      {"start": 3.80, "end": 8.15, "speaker": "SPEAKER_01", "text": "Добрый день. У меня последние две недели болит голова."},
      {"start": 8.50, "end": 12.3, "speaker": "SPEAKER_00", "text": "Какой характер боли? Давящий, пульсирующий?"}
    ]
  }
}
```

---

## Step 5 — Integrate with Local Dashboard

Update `server.py` to call the serverless endpoint instead of managing SSH:

```python
# In server.py — replace setup_pod / start_transcription with:

@app.post("/transcribe-cloud")
async def transcribe_cloud(task_id: str):
    """Send audio to RunPod Serverless for transcription."""
    audio_path = UPLOAD_DIR / task_id
    
    # 1. Upload to S3 and get a presigned URL
    s3_key = f"audio/{task_id}"
    s3.upload_file(str(audio_path), S3_BUCKET, s3_key)
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': s3_key},
        ExpiresIn=3600
    )
    
    # 2. Call RunPod Serverless
    resp = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={
            "input": {
                "audio_url": presigned_url,
                "language": "ru",
                "min_speakers": 2,
                "max_speakers": 5,
            }
        },
        timeout=600,
    )
    
    result = resp.json()
    return result["output"]
```

---

## Cost Estimate

| Audio Length | Approx. GPU Time | Cost (RTX 6000 Ada) |
|-------------|-------------------|----------------------|
| 10 min      | ~30 sec           | ~$0.02               |
| 30 min      | ~90 sec           | ~$0.06               |
| 1 hour      | ~3 min            | ~$0.12               |
| 2 hours     | ~6 min            | ~$0.24               |

> Zero idle cost. You only pay when audio is being processed.

---

## Checklist

- [ ] Create DockerHub account (or use GitHub Container Registry)
- [ ] Accept Pyannote model terms on HuggingFace
- [ ] Build & push Docker image
- [ ] Create RunPod Serverless endpoint
- [ ] Test with a sample Russian audio file
- [ ] Update local dashboard to use the new API
