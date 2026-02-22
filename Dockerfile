# ═══════════════════════════════════════════════════════════════
#  MODERNIZED STACK — PyTorch 2.4 / CUDA 12.4 / WhisperX 3.8.1
# ═══════════════════════════════════════════════════════════════
# Upgrade from: PyTorch 2.1 / CUDA 11.8 / WhisperX 3.1.1
#
# Key changes:
#   - Base image: CUDA 12.4 (was 11.8)
#   - PyTorch: 2.4.0 (was 2.1.0)
#   - WhisperX: 3.8.1 (was 3.1.1) — requires pyannote v4, faster-whisper 1.1+
#   - CTranslate2: 4.5+ (was 3.24) — requires cuDNN v9 (included in base image)
#   - pyannote.audio: 4.0+ (was 3.1.1) — use_auth_token → token
#   - No more numpy<2.0 / transformers<4.45 constraints
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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

# ─── Core ML Stack ───
# No constraints file needed — PyTorch 2.4 is already in the base image
# and all modern versions are compatible with each other.

# CTranslate2 4.5+ requires cuDNN v9 (bundled in the CUDA 12.4 devel image)
RUN pip install --no-cache-dir "ctranslate2>=4.5.0"

# faster-whisper 1.1+ (uses ctranslate2 4.5+)
RUN pip install --no-cache-dir "faster-whisper>=1.1.1"

# pyannote.audio v4 (breaking change: use_auth_token → token)
RUN pip install --no-cache-dir "pyannote.audio>=4.0.0"

# Pin transformers <4.48 — versions ≥4.48 removed `Pipeline` from top-level imports,
# which breaks whisperx/asr.py: `from transformers import Pipeline`
RUN pip install --no-cache-dir "transformers>=4.40,<4.48"

# WhisperX 3.8.1 — will use the already-installed transformers
RUN pip install --no-cache-dir "whisperx>=3.8.1"

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
