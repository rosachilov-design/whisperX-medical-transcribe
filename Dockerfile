# ═══════════════════════════════════════════════════════════════
#  MODERNIZED STACK — PyTorch 2.4 / CUDA 12.4 / WhisperX 3.8.1
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models
ENV HF_HOME=/app/models

# Install runpod and basic utils
RUN pip install --no-cache-dir runpod requests setuptools onnxruntime-gpu

# ─── Core ML Stack ───
RUN pip install --no-cache-dir "ctranslate2>=4.5.0"
RUN pip install --no-cache-dir "faster-whisper>=1.1.1"
RUN pip install --no-cache-dir "pyannote.audio>=4.0.0"
RUN pip install --no-cache-dir "whisperx>=3.8.1"

# ─── Pre-download Models ───
# Bake models into the image for instant cold-starts
RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Russian alignment model (essential for your medical use case)
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cpu', model_dir='/app/models')"

# Optional: Pre-cache diarization (requires HF_TOKEN at build time or it skips)
RUN python -c "import os; from pyannote.audio import Pipeline; token=os.environ.get('HF_TOKEN'); (Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=token) if token else print('Skipping diarization bake'))"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
