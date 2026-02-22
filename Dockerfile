FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install basic dependencies (ffmpeg needed for audio conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Install python packages
COPY requirements.txt .

# 1. Force install the correct CUDA-enabled torch versions first
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. Install stack with forced stable versions
RUN pip install --no-cache-dir \
    "numpy<2" \
    "filelock<3.12" \
    "huggingface_hub>=0.17.0" \
    runpod \
    faster-whisper \
    pyannote.audio==3.1.1 \
    soundfile

# 3. Download models during build for instant cold-starts
# We download Whisper turbo and Pyannote 3.1
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('turbo', device='cpu', compute_type='int8')"

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
RUN python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='$HF_TOKEN')"

# Copy the serverless handler code inside
COPY handler.py .

# Expose port (if applicable, though runpod handles execution directly on handler)
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
