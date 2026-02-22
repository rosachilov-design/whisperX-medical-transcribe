# Transcription Service: Project Architecture & Development Review

This document serves as a comprehensive overview of the current transcription and diarization stack, detailing the technologies used, recent architectural changes, critical bugs discovered, and how they were resolved to optimize both local development and cloud (RunPod Serverless) production deployments.

## 1. Project Stack & Architecture

### Core Technologies
*   **Web Framework:** `FastAPI` powered by `Uvicorn` for serving the local dashboard and handling API endpoints (Upload, Status, Download).
*   **Transcription Engine (Primary):** `faster-whisper` (powered by CTranslate2). Selected for its massive performance gains (400-500 FPS) over the native library, utilizing `FP16` compute types on CUDA-enabled GPUs.
*   **Transcription Engine (Fallback):** `openai-whisper`. Maintained as a seamless, automatic fallback engine for environments (like local Windows machines) where specific CUDA DLLs required by `faster-whisper` might be missing.
*   **Diarization Engine:** `pyannote.audio` (Speaker Diarization v3.1 model via Hugging Face). Used to map audio timelines to distinct speakers.
*   **Audio Processing:** `ffmpeg` (cli) and `soundfile` (python). Used for heavy lifting: converting compressed `.m4a` source files into uncompressed `pcm_s16le` `.wav` files, and rapidly extracting natural chunks based on speaker timelines.
*   **Deployment Target:** RunPod Serverless containerized via Docker. Base image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`.

### Specific Settings used for AI Models
*   **Faster-Whisper:** Model size `"turbo"`. Compute type explicitly defined as `"float16"` for CUDA and `"int8"` for CPU. Conditioned on previous text to maintain continuity across chunks using a sliding window of the last 50 words.
*   **OpenAI-Whisper (Fallback):** Model size `"turbo"`. First attempts `fp16=True`, with a nested `try-except` fallback to `fp16=False` if legacy PyTorch `NaN` logits errors occur.
*   **Pyannote:** Hosted Pipeline instantiated via Hugging Face Token. Pushed to `torch.device("cuda")`.

---

## 2. Recent Development & Critical Bug Fixes

The transition to a Serverless-ready, high-speed architecture surfaced several deep, systemic bottlenecks. Here are the major bugs discovered and resolved:

### Bug 1: `torchaudio.list_audio_backends` Missing Attribute Crash
**The Problem:** Newer versions of PyTorch entirely removed `torchaudio.list_audio_backends`. Because `pyannote.audio` directly depends on this attribute at initialization, the server would instantly crash on boot, bypassing diarization completely and defaulting to "Speaker 1" for the entire transcript.
**The Fix:** Implemented a runtime polyfill at the very top of `server.py` and `handler.py` before `pyannote` is imported. It injects a dummy lambda function:
```python
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
```

### Bug 2: FFmpeg scanning bottleneck (1% GPU Utilization)
**The Problem:** When extracting ~30-second audio chunks from massive (1 hour+) `.m4a` files, the script forced `ffmpeg` to linearly scan and decompress the entire source file for *every single chunk*. This resulted in heavy CPU throttling while the GPU sat idle at 1% waiting for the tiny pieces of audio.
**The Fix:** Refactored the transcription loop. At the start of the transcription phase, `ffmpeg` now does a one-time, full-file decode into an uncompressed `.wav` file (`pcm_s16le` format) stored in the `/cache` directory. Chunk extraction is now performed instantaneously from this raw `.wav` file, keeping the GPU constantly fed with transcription data.

### Bug 3: The Ghost Headers (Endless Silence Processing)
**The Problem:** When ripping chunks from the newly uncompressed `.wav` file to speed up processing, the `ffmpeg` command utilized `-c copy`. While this instantly copied the audio data, it also copied the original 1-hour metadata header. The AI received a 30-second chunk that *claimed* to be 1-hour long, causing the VAD (Voice Activity Detection) to endlessly process 59 minutes of silent void data, effectively halting transcription progress exactly at 10%.
**The Fix:** Changed the chunking argument from `-c copy` to `-c:a pcm_s16le`. This forces `ffmpeg` to properly re-encode the headers for each specific extracted chunk, restoring lightning-fast processing speeds.

### Bug 4: Faster-Whisper CUDA DLL Collision on Local Windows
**The Problem:** The local Windows machine was updated to PyTorch with **CUDA 13**, housing `cublas64_13.dll`. However, the compiled Python binaries for `faster-whisper` explicitly searched for **CUDA 12** (`cublas64_12.dll`). When the file was fundamentally missing from the operating system, `faster-whisper` silently failed and skipped chunks entirely. 
**The Fix:** Rather than downgrading the local Windows environment, a robust fallback mechanism was built into `server.py`. The transcription loop catches the explicit DLL/cuDNN errors. If triggered, it dynamically imports `openai-whisper`, loading it into memory, and routes the chunk to the fallback engine.
1. Local Windows machines seamlessly fall back to standard `openai-whisper` (so the dashboard works without crashing).
2. RunPod Serverless instances inherit the explicit CUDA 11/12 environment baked into the custom `Dockerfile`, executing perfectly on `faster-whisper` at full speed.

---

## 3. Serverless Deployment Integration

To move off local pods and into auto-scaling environments, the following files were established:
1. **`Dockerfile`**: Designed using an official RunPod PyTorch base image. Explicitly builds `ffmpeg`, `faster-whisper`, and `pyannote.audio` without version conflicts.
2. **`handler.py`**: A stateless conversion of `server.py`. It strips the FastAPI interfaces and operates as a direct RunPod Serverless endpoint. It accepts a JSON payload containing an `audio` URL, downloads it to `/tmp/`, runs the identical Diarization -> Natural Chunking -> Transcription pipeline, and immediately returns the formatted JSON payload of the speaker timeline.
