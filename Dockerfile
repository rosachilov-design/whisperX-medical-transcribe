# ═══════════════════════════════════════════════════════════════
#  MODERNIZED STACK — PyTorch 2.4 / CUDA 12.4 / WhisperX 3.8.1
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ─── Core ML Stack ───
# WhisperX 3.8.1 strictly requires Torch 2.8.0 and Pyannote 4.0+. 
# We explicitly include torchvision here so it upgrades alongside torch and doesn't get left behind (causing the nms error).
RUN pip install --no-cache-dir --upgrade \
    runpod requests setuptools \
    torch torchvision torchaudio \
    "ctranslate2>=4.5.0" \
    "faster-whisper>=1.1.1" \
    "pyannote.audio>=4.0.0" \
    "whisperx==3.8.1" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ─── Pre-download Models ───
# Bake models into the image for instant cold-starts
RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Russian alignment model (essential for your medical use case)
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cpu', model_dir='/app/models')"

# Optional: Pre-cache diarization (requires HF_TOKEN at build time or it skips)
RUN python -c "import os; from pyannote.audio import Pipeline; t=os.environ.get('HF_TOKEN'); (Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=t) if t else print('Skipping diarization bake'))"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
