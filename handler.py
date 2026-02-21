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
print("Loading WhisperX model...")
model = whisperx.load_model(
    "large-v3", DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_DIR
)
print("✅ WhisperX ready.")


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
    hf_token = inp.get("hf_token", HF_TOKEN)
    min_speakers = inp.get("min_speakers", None)
    max_speakers = inp.get("max_speakers", None)

    # 1. Download audio
    audio_path = download_audio(audio_url)
    audio = whisperx.load_audio(audio_path)

    # 2. Transcribe
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)

    # 3. Align (word-level timestamps)
    model_a, metadata = whisperx.load_align_model(
        language_code=language, device=DEVICE, model_dir=MODEL_DIR
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False
    )
    # Free alignment model memory
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Diarize (speaker identification)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=DEVICE
    )
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()

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
