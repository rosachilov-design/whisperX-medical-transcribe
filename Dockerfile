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

# 2. Install the rest of the stack
RUN pip install --no-cache-dir \
    runpod \
    faster-whisper \
    pyannote.audio==3.1.1 \
    soundfile

# Copy the serverless handler code inside
COPY handler.py .

# Expose port (if applicable, though runpod handles execution directly on handler)
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
