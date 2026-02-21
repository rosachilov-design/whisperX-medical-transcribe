import runpod
import whisperx
import torch
import gc
import os
import tempfile
import requests

# ─── Config ───
DEVICE = "cuda"
BATCH_SIZE = 32          # RTX 6000 Ada can handle 32+ easily with 48GB VRAM
COMPUTE_TYPE = "float16" # Native Ada support
MODEL_DIR = "/app/models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Load models once (stays in GPU memory between requests) ───
print("Loading WhisperX transcription model...")
model = whisperx.load_model(
    "large-v3", DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_DIR
)

print("Loading Alignment model...")
model_a, metadata = whisperx.load_align_model(
    language_code="ru", device=DEVICE, model_dir=MODEL_DIR
)

print("Loading Diarization pipeline...")
diarize_model = whisperx.DiarizationPipeline(
    token=HF_TOKEN, device=DEVICE
)
print("✅ All models ready and cached in VRAM.")


def download_audio(url: str) -> str:
    """Download audio from URL to a temp file."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    ext = url.split('.')[-1].split('?')[0] or 'wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def handler(job):
    """
    Input:
      - audio_url: URL to audio file (S3 presigned, public link, etc.)
      - language:  Language code (default: "ru")
      - hf_token:  HuggingFace token for diarization (or set via env)
      - min_speakers / max_speakers: Optional speaker count hints
    """
    inp = job["input"]
    audio_url = inp["audio_url"]
    language = inp.get("language", "ru")
    min_speakers = inp.get("min_speakers", None)
    max_speakers = inp.get("max_speakers", None)

    # 1. Download audio
    audio_path = download_audio(audio_url)
    audio = whisperx.load_audio(audio_path)

    # 2. Transcribe
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)

    # 3. Align (word-level timestamps)
    # Using the globally loaded model_a and metadata
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False
    )

    # 4. Diarize (speaker identification)
    # Using the globally loaded diarize_model
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers,
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. Clean up temp file
    os.unlink(audio_path)

    # 6. Format output
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "speaker": seg.get("speaker", "Unknown"),
            "text": seg["text"].strip(),
        })

    return {
        "language": language,
        "segments": segments,
    }


runpod.serverless.start({"handler": handler})
