import os
import runpod
import whisperx
import torch
import gc
import re
import requests
import tempfile
import pandas as pd

# ─── Config & Init ───
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_DIR = "/app/models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global cache for models
MODELS = {
    "whisper": None,
    "align": {},
    "diarize": None
}

def get_whisper():
    if MODELS["whisper"] is None:
        print("🚀 Loading Whisper model...")
        MODELS["whisper"] = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
    return MODELS["whisper"]

def get_align(lang):
    if lang not in MODELS["align"]:
        print(f"🚀 Loading Alignment model ({lang})...")
        MODELS["align"][lang] = whisperx.load_align_model(language_code=lang, device=DEVICE, model_dir=MODEL_DIR)
    return MODELS["align"][lang]

def get_diarize():
    if MODELS["diarize"] is None:
        print("🚀 Loading Diarization pipeline (pyannote/speaker-diarization-3.1)...")
        # Explicitly use 3.1 model which handles overlapping/back-and-forth speech better
        model_name = "pyannote/speaker-diarization-3.1"
        try:
            MODELS["diarize"] = whisperx.DiarizationPipeline(
                model_name=model_name, 
                use_auth_token=HF_TOKEN, 
                device=DEVICE
            )
        except (AttributeError, TypeError):
            try:
                from whisperx.diarize import DiarizationPipeline
                MODELS["diarize"] = DiarizationPipeline(
                    model_name=model_name, 
                    token=HF_TOKEN, 
                    device=DEVICE
                )
            except Exception as e:
                print(f"⚠️ Fallback loading diarization: {e}")
                from whisperx.diarize import DiarizationPipeline
                MODELS["diarize"] = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    return MODELS["diarize"]

def clean_hallucinations(text: str) -> str:
    """Medical-focused Russian hallucination filter."""
    patterns = [
        r'\bРедактор субтитров\s+([А-ЯA-Z]\.?\s*){1,2}[А-ЯA-Z][а-яa-z]+',
        r'\bКорректор\s+([А-ЯA-Z]\.?\s*){1,2}[А-ЯA-Z][а-яa-z]+',
        r'\bСубтитры\s*:\s*[^\.]+',
        r'\bПеревод\s*:\s*[^\.]+',
        r'\bОзвучка\s*:\s*[^\.]+',
        r'\bРедактор субтитров\b',
        r'\bКорректор\b',
        r'\b(Все права защищены|Продолжение следует|Ставьте лайки|Подписывайтесь на канал)\b',
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned).strip()

import boto3
from botocore.config import Config

def download_file(url: str, s3_creds: dict = None) -> str:
    if s3_creds:
        print(f"📥 Downloading audio natively via boto3: {url}...")
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=s3_creds["endpoint"],
                region_name=s3_creds["region"],
                aws_access_key_id=s3_creds["access_key"],
                aws_secret_access_key=s3_creds["secret_key"],
                config=Config(signature_version="s3v4"),
            )
            suffix = "." + url.split(".")[-1] if "." in url else ".m4a"
            fd, path = tempfile.mkstemp(suffix=suffix)
            s3.download_file(s3_creds["bucket"], url, path)
            return path
        except Exception as e:
            raise Exception(f"Native S3 download failed: {e}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    print(f"📥 Downloading audio from: {url[:50]}...")
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise Exception("HTTP 403 Forbidden: The S3 URL may have expired or the worker is blocked. Please retry.")
        raise
    
    suffix = "." + url.split("?")[0].split(".")[-1] if "." in url else ".m4a"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as tmp:
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)
    return path

# ─── Handler ───

def handler(job):
    inp = job["input"]
    action = inp.get("action", "full") # default to full if not specified
    audio_url = inp.get("audio") or inp.get("audio_url")
    s3_creds = inp.get("s3_creds")
    language = inp.get("language", "ru")
    min_speakers = inp.get("min_speakers")
    max_speakers = inp.get("max_speakers")
    
    if not audio_url:
        return {"error": "Missing audio URL"}

    local_path = None
    try:
        local_path = download_file(audio_url, s3_creds)
        audio = whisperx.load_audio(local_path)
        
        response = {}

        # 1. Diarization (if requested or full)
        if action in ["diarize", "full"]:
            pipe = get_diarize()
            print(f"🎙️ Diarizing (min={min_speakers}, max={max_speakers})...")
            diarize_segments = pipe(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            
            # Format timeline for server.py compatibility
            timeline = []
            for _, row in diarize_segments.iterrows():
                timeline.append({
                    "start": round(row["start"], 3),
                    "end": round(row["end"], 3),
                    "speaker": row["speaker"]
                })
            response["timeline"] = timeline
            
            if action == "diarize":
                return response

        # 2. Transcription (if requested or full)
        if action in ["transcribe", "full"]:
            model = get_whisper()
            print("📝 Transcribing...")
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
            
            # 3. Alignment
            print("🎯 Aligning...")
            model_a, metadata = get_align(language)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            
            # 4. Assign Speakers (if we have diarization info)
            if action == "full":
                # We already have diarize_segments from step 1
                result = whisperx.assign_word_speakers(diarize_segments, result)
            elif action == "transcribe" and "timeline" in inp:
                # User provided timeline from previous step
                provided_timeline = pd.DataFrame(inp["timeline"])
                result = whisperx.assign_word_speakers(provided_timeline, result)

            # 5. Format Result for server.py compatibility
            final_segments = []
            for seg in result["segments"]:
                text = clean_hallucinations(seg["text"])
                if text:
                    final_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": text,
                        "speaker": seg.get("speaker", "Unknown")
                    })
            
            response["result"] = final_segments

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
