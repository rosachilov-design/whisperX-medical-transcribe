# ═══════════════════════════════════════════════════════════════
#  MODERNIZED STACK — PyTorch 2.8 / CUDA 12.8 / WhisperX 3.8.1
# ═══════════════════════════════════════════════════════════════
# Upgrade from: PyTorch 2.4 / CUDA 12.4
#
# Key changes:
#   - Base image: CUDA 12.8, PyTorch 2.8.0
#   - WhisperX: 3.8.1 (fully supported)
#   - pyannote.audio: 4.0+ (requires torch>=2.8, natively supported)
#   - transformers: 4.48+ (mandated by whisperx 3.8.1)
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# System deps (same as before)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models
ENV HF_HOME=/app/models

# Install runpod and basic utils
RUN pip install --no-cache-dir runpod requests setuptools

# Install onnxruntime for CUDA 12
RUN pip install --no-cache-dir onnxruntime-gpu

# ─── Lock the Torch ecosystem to CUDA builds ───
# The base image ships torch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0
# with CUDA 12.8. If any later pip install pulls in torchvision as a dependency,
# pip may replace the CUDA build with a CPU-only wheel from PyPI, which removes
# the compiled C++ operators (torchvision::nms, etc.) and breaks everything.
#
# Fix: force-reinstall from the official PyTorch cu128 wheel index so the
# CUDA-enabled builds are always used, even after dependency resolution.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ─── Core ML Stack ───
# CTranslate2 4.5+ requires cuDNN v9 (bundled in the CUDA 12.4 devel image)
RUN pip install --no-cache-dir "ctranslate2>=4.5.0"

# faster-whisper 1.1+ (uses ctranslate2 4.5+)
RUN pip install --no-cache-dir "faster-whisper>=1.1.1"

# pyannote.audio 4.0+ requires torch>=2.8. WhisperX 3.8.1 natively uses it.
RUN pip install --no-cache-dir "pyannote.audio>=4.0.0"

# WhisperX 3.8.1 — install without --no-deps to allow it to pull correct versions
# (like transformers>=4.48.0 and others)
RUN pip install --no-cache-dir "whisperx>=3.8.1"

# Remaining whisperx deps that might be needed explicitly
RUN pip install --no-cache-dir pandas nltk omegaconf

# ─── Re-lock torch ecosystem after all installs ───
# Safety net: if any package above quietly replaced torchvision with a CPU
# wheel, this final reinstall restores the CUDA builds.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ─── Pre-download Models ───
# Whisper large-v3 (same model, same quality — just different runtime)
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"

# Pyannote diarization (v4 uses 'token' instead of 'use_auth_token')
RUN echo "import os" > /tmp/preload.py && \
    echo "from pyannote.audio import Pipeline" >> /tmp/preload.py && \
    echo "token = os.environ.get('HF_TOKEN')" >> /tmp/preload.py && \
    echo "try:" >> /tmp/preload.py && \
    echo "    Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=token, cache_dir='/app/models')" >> /tmp/preload.py && \
    echo "    print('✅ Diarization model cached.')" >> /tmp/preload.py && \
    echo "except Exception as e:" >> /tmp/preload.py && \
    echo "    print(f'⚠️ Could not pre-cache diarization: {e}')" >> /tmp/preload.py && \
    python /tmp/preload.py && \
    rm /tmp/preload.py

# Russian alignment model
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cpu', model_dir='/app/models')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
