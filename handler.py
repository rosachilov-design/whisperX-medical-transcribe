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
COMPUTE_TYPE = "float32" # Use full precision to avoid glitches/cutoffs
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
        print("🚀 Loading Whisper model (float32, high sensitivity VAD)...")
        vad_options = {"vad_onset": 0.450, "vad_offset": 0.363}
        MODELS["whisper"] = whisperx.load_model(
            "large-v3", 
            DEVICE, 
            compute_type=COMPUTE_TYPE, 
            download_root=MODEL_DIR,
            vad_options=vad_options
        )
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
        
        # ═══ TUNE PYANNOTE HYPERPARAMETERS ═══
        # Access the underlying pyannote pipeline to adjust clustering/segmentation.
        # The DiarizationPipeline wrapper stores the pyannote Pipeline as .model
        try:
            pyannote_pipeline = MODELS["diarize"].model
            params = pyannote_pipeline.parameters(instantiated=True)
            print(f"📊 Default pyannote params: {params}")
            
            # Lower clustering threshold: default ~0.7153 is too "blind" for similar voices.
            # 0.5 is moderately aggressive - balances between separation and accuracy.
            params["clustering"]["threshold"] = 0.5
            
            # Lower min_duration_off: default 0.5s merges rapid turn-taking.
            # 0.1s preserves short back-and-forth exchanges ("Да." → response).
            params["segmentation"]["min_duration_off"] = 0.1
            
            # Lower min_duration_on: allows shorter speaker segments to be recognized.
            # 0.2s enables detection of brief 1-2 word interjections.
            params["segmentation"]["min_duration_on"] = 0.2
            
            pyannote_pipeline.instantiate(params)
            print(f"✅ Tuned pyannote params: clustering.threshold=0.5, min_duration_off=0.1, min_duration_on=0.2")
        except Exception as e:
            print(f"⚠️ Could not tune pyannote params (non-fatal): {e}")
        
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

def rescue_short_interjections(segments, max_duration=2.0):
    """
    Post-process segments to rescue short interjections that were absorbed.
    
    If a short segment (<=max_duration) is surrounded by segments from a DIFFERENT speaker,
    it's likely a real interjection and should be kept separate.
    
    Also splits segments where speaker changes mid-way based on diarization timeline.
    """
    if len(segments) < 3:
        return segments
    
    rescued = []
    
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        speaker = seg.get("speaker", "Unknown")
        
        # Check if this is a short segment
        if duration <= max_duration:
            # Look at surrounding context
            prev_speaker = segments[i-1].get("speaker", "Unknown") if i > 0 else None
            next_speaker = segments[i+1].get("speaker", "Unknown") if i < len(segments) - 1 else None
            
            # If surrounded by same speaker but we're different, keep us separate
            if prev_speaker and next_speaker and prev_speaker == next_speaker and speaker != prev_speaker:
                # This is a genuine interjection - keep it!
                rescued.append(seg)
                continue
            
            # If we match previous but next is different, and we're short,
            # we might have been mis-assigned. Check if we should belong to next.
            if prev_speaker and next_speaker and speaker == prev_speaker and speaker != next_speaker:
                # Short segment same as prev, next is different
                # Could be misclassified. Keep as-is for now (user can fix if needed)
                rescued.append(seg)
                continue
        
        rescued.append(seg)
    
    return rescued


def smooth_diarization(df):
    """
    Only merges consecutive segments of the same speaker.
    Removed 'flicker' filtering because in medical interviews, short 
    interjections ("угу", "да") between segments of another speaker 
    are actually important and shouldn't be absorbed.
    """
    if df.empty:
        return df
    
    # Sort by start time
    df = df.sort_values(by="start").reset_index(drop=True)
    
    merged_rows = []
    current_row = df.iloc[0].to_dict()
    
    for i in range(1, len(df)):
        next_row = df.iloc[i]
        # Only merge if it's the EXACT same speaker and they are consecutive or very close
        if next_row["speaker"] == current_row["speaker"]:
            current_row["end"] = next_row["end"]
        else:
            merged_rows.append(current_row)
            current_row = next_row.to_dict()
    merged_rows.append(current_row)
    
    return pd.DataFrame(merged_rows)


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
    # Default to exactly 2 speakers for medical interviews (interviewer + respondent)
    # Setting num_speakers=2 forces pyannote to find the 2 most distinct voice clusters
    min_speakers = inp.get("min_speakers") or 2
    max_speakers = inp.get("max_speakers") or 2
    num_speakers = inp.get("num_speakers") or 2
    
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
            print(f"🎙️ Diarizing (min={min_speakers}, max={max_speakers}, num={num_speakers})...")
            diarize_segments = pipe(audio, min_speakers=min_speakers, max_speakers=max_speakers, num_speakers=num_speakers)
            
            # Apply smoothing: only merge consecutive segments of the same speaker
            print("🧹 Merging consecutive same-speaker segments...")
            diarize_segments = smooth_diarization(diarize_segments)

            
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
                result = whisperx.assign_word_speakers(diarize_segments, result, fill_nearest=True)
            elif action == "transcribe" and "timeline" in inp:
                # User provided timeline from previous step
                provided_timeline = pd.DataFrame(inp["timeline"])
                result = whisperx.assign_word_speakers(provided_timeline, result, fill_nearest=True)

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
