FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models
ENV HF_HOME=/app/models

# Install runpod and basic utils
RUN pip install --no-cache-dir runpod requests setuptools
RUN pip install --no-cache-dir onnxruntime-gpu

# Install WhisperX and its hard dependencies one by one
# This prevents the resolver from getting stuck in a loop
RUN pip install --no-cache-dir faster-whisper
RUN pip install --no-cache-dir "pyannote.audio==3.1.1" "torch==2.1.0+cu118" "torchaudio==2.1.0+cu118" "torchvision==0.16.0+cu118" --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir ctranslate2
RUN pip install --no-cache-dir "git+https://github.com/m-bain/whisperx.git" "torch==2.1.0+cu118" "torchaudio==2.1.0+cu118" "torchvision==0.16.0+cu118" --extra-index-url https://download.pytorch.org/whl/cu118

# Pre-download models into the image
# Using int8 to save memory during the download phase (prevents OOM in build env)
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"

# Pyannote diarization (try to pre-download, but don't fail build if token is missing/invalid)
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
