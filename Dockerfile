FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Python deps with fixed versions to avoid torchaudio attribute errors
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    torchvision==0.16.0 \
    runpod \
    git+https://github.com/m-bain/whisperx.git \
    pyannote.audio==3.1.1 \
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
