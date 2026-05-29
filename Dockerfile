# syntax=docker/dockerfile:1.7

# Modernized stack: PyTorch 2.8 / CUDA 12.8 / WhisperX 3.8.1.
# WhisperX 3.8.1 pins torch ~=2.8.0, so the base image and PyTorch
# wheels must stay on the same CUDA generation to avoid NCCL symbol errors.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV LOCAL_OCR_DEVICE=gpu
ENV ZOOM_DIARIZATION_MODE=robust
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Base utilities.
RUN pip install --no-cache-dir --upgrade runpod requests setuptools

# PaddleOCR GPU runtime for RunPod. Install this before re-pinning PyTorch so
# Paddle's CUDA dependencies cannot downgrade the NCCL library Torch imports.
RUN pip install --no-cache-dir \
    "paddlepaddle-gpu==3.3.0" \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Keep the PyTorch triplet explicit and CUDA-matched. This avoids importing a
# Torch 2.8 wheel against an older NCCL shared library from the base image or
# from another CUDA package repository.
RUN pip install --no-cache-dir --force-reinstall \
    "torch==2.8.0+cu128" \
    "torchvision==0.23.0+cu128" \
    "torchaudio==2.8.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# Core ML stack.
RUN pip install --no-cache-dir \
    "ctranslate2>=4.5.0" \
    "faster-whisper>=1.1.1" \
    "pyannote.audio>=4.0.0" \
    "whisperx==3.8.1" \
    paddleocr \
    opencv-python-headless \
    rapidfuzz

# Do not import GPU Paddle during docker build: build workers do not expose the
# host NVIDIA driver library (libcuda.so.1). Runtime OCR imports it on RunPod.
RUN python - <<'PY'
from importlib.metadata import version
import torch

print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"paddlepaddle-gpu {version('paddlepaddle-gpu')}")
PY

# Pre-download models into the image for instant cold-starts.
RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"

# Russian alignment model (essential for your medical use case).
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cpu', model_dir='/app/models')"

# Optional: pre-cache diarization with a BuildKit secret named hf_token.
RUN --mount=type=secret,id=hf_token python - <<'PY'
from pathlib import Path
from pyannote.audio import Pipeline

model_name = "pyannote/speaker-diarization-community-1"
secret_path = Path("/run/secrets/hf_token")
token = secret_path.read_text().strip() if secret_path.exists() else ""

if not token:
    print(f"Skipping diarization bake: HF_TOKEN not set for {model_name}.")
else:
    try:
        Pipeline.from_pretrained(model_name, token=token)
        print(f"Cached diarization model: {model_name}")
    except Exception as exc:
        print(f"Warning: could not pre-cache {model_name}: {exc}")
PY

COPY handler.py /app/handler.py
COPY paddle_ocr_factory.py /app/paddle_ocr_factory.py
CMD ["python", "-u", "/app/handler.py"]
