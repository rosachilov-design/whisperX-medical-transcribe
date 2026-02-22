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

# ─── Core ML Stack ───
# Force matching versions for the CUDA 12.4 stack to prevent '2.8.0' type mismatch errors
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    runpod requests setuptools \
    "numpy<2" \
    "onnxruntime-gpu>=1.18.0" \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    "torchaudio==2.4.1" \
    "ctranslate2>=4.5.0" \
    "faster-whisper>=1.1.1" \
    "pyannote.audio>=3.3.1" \
    "whisperx>=3.8.1" \
    --extra-index-url https://download.pytorch.org/whl/cu124

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
