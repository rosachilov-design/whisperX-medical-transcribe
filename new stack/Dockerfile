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

# We only need specific dependencies for the serverless endpoint
RUN pip install --no-cache-dir runpod faster-whisper pyannote.audio soundfile

# Copy the serverless handler code inside
COPY handler.py .

# Expose port (if applicable, though runpod handles execution directly on handler)
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
