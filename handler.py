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
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
UNKNOWN_SPEAKER = "Unknown"

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

def _build_diarization_pipeline(model_name):
    if not HF_TOKEN:
        raise RuntimeError(
            f"HF_TOKEN is required for diarization. Set HF_TOKEN and accept access to {model_name} on Hugging Face."
        )

    load_attempts = (
        (getattr(whisperx, "DiarizationPipeline", None), {"model_name": model_name, "use_auth_token": HF_TOKEN, "device": DEVICE}),
        (None, {"model_name": model_name, "token": HF_TOKEN, "device": DEVICE}),
        (None, {"model_name": model_name, "use_auth_token": HF_TOKEN, "device": DEVICE}),
    )

    last_error = None
    for loader, kwargs in load_attempts:
        try:
            if loader is None:
                from whisperx.diarize import DiarizationPipeline
                loader = DiarizationPipeline
            return loader(**kwargs)
        except (AttributeError, TypeError) as exc:
            last_error = exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not load diarization model {model_name}. "
                f"Verify HF_TOKEN is valid and that this account accepted access to {model_name}. "
                f"Original error: {exc}"
            ) from exc

    raise RuntimeError(
        f"WhisperX diarization loader is incompatible with {model_name}. "
        f"Last loader error: {last_error}"
    ) from last_error

def get_diarize():
    if MODELS["diarize"] is None:
        print(f"🚀 Loading Diarization pipeline ({DIARIZATION_MODEL})...")
        MODELS["diarize"] = _build_diarization_pipeline(DIARIZATION_MODEL)
        
        # ═══ TUNE PYANNOTE HYPERPARAMETERS ═══
        # Access the underlying pyannote pipeline to adjust clustering/segmentation.
        # The DiarizationPipeline wrapper stores the pyannote Pipeline as .model
        try:
            pyannote_pipeline = MODELS["diarize"].model
            params = pyannote_pipeline.parameters(instantiated=True)
            print(f"📊 Default pyannote params: {params}")

            changed = []
            clustering = params.get("clustering")
            if isinstance(clustering, dict) and "threshold" in clustering:
                clustering["threshold"] = 0.42
                changed.append("clustering.threshold=0.42")

            segmentation = params.get("segmentation")
            if isinstance(segmentation, dict):
                if "min_duration_off" in segmentation:
                    segmentation["min_duration_off"] = 0.05
                    changed.append("min_duration_off=0.05")
                if "min_duration_on" in segmentation:
                    segmentation["min_duration_on"] = 0.12
                    changed.append("min_duration_on=0.12")

            pyannote_pipeline.instantiate(params)
            if changed:
                print(f"✅ Tuned pyannote params: {', '.join(changed)}")
            else:
                print("⚠️ Pyannote params loaded, but overlap-specific keys were not available to tune.")
        except Exception as e:
            print(f"⚠️ Could not tune pyannote params (non-fatal): {e}")
        
    return MODELS["diarize"]


def _normalize_speaker_label(label):
    if label is None:
        return UNKNOWN_SPEAKER

    if isinstance(label, float) and pd.isna(label):
        return UNKNOWN_SPEAKER

    text = str(label).strip()
    if not text or text.lower() == "nan":
        return UNKNOWN_SPEAKER

    return text


def _fill_unlabeled_word_runs(words):
    """
    Conservatively fill runs of unlabeled words only when both surrounding
    labeled words agree on the speaker. Everything else stays Unknown.
    """
    if not words:
        return words

    normalized_words = []
    for word in words:
        normalized_word = dict(word)
        normalized_word["speaker"] = _normalize_speaker_label(word.get("speaker"))
        normalized_words.append(normalized_word)

    run_start = None
    for index, word in enumerate(normalized_words):
        if word["speaker"] == UNKNOWN_SPEAKER:
            if run_start is None:
                run_start = index
            continue

        if run_start is None:
            continue

        prev_index = run_start - 1
        prev_speaker = normalized_words[prev_index]["speaker"] if prev_index >= 0 else UNKNOWN_SPEAKER
        next_speaker = word["speaker"]

        if prev_speaker != UNKNOWN_SPEAKER and prev_speaker == next_speaker:
            for fill_index in range(run_start, index):
                normalized_words[fill_index]["speaker"] = prev_speaker

        run_start = None

    return normalized_words

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

def split_by_word_speakers(segments):
    """
    Rebuild segments from word-level speaker assignments.
    
    WhisperX assigns speakers per-word, but groups words into segments with
    a single majority-vote speaker label. This loses short interjections and
    bleeds speaker boundaries.
    
    This function splits segments wherever the word-level speaker changes,
    producing tight, per-speaker segments.
    """
    new_segments = []
    
    for seg in segments:
        words = seg.get("words", [])
        
        # If no word-level info, keep segment as-is
        if not words:
            normalized_seg = dict(seg)
            normalized_seg["speaker"] = _normalize_speaker_label(seg.get("speaker"))
            new_segments.append(normalized_seg)
            continue

        words = _fill_unlabeled_word_runs(words)
        
        # Group consecutive words by speaker
        current_speaker = None
        current_words = []
        
        for word in words:
            w_speaker = _normalize_speaker_label(word.get("speaker"))
            
            if w_speaker != current_speaker and current_words:
                # Flush previous group as a new segment
                new_segments.append(_words_to_segment(current_words, current_speaker))
                current_words = []
            
            current_speaker = w_speaker
            current_words.append(word)
        
        # Flush last group
        if current_words:
            new_segments.append(_words_to_segment(current_words, current_speaker))
    
    return new_segments


def _words_to_segment(words, speaker):
    """Build a segment dict from a list of word dicts."""
    texts = [w.get("word", "") for w in words]
    return {
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", words[-1].get("start", 0.0)),
        "text": " ".join(texts).strip(),
        "speaker": _normalize_speaker_label(speaker),
    }


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
                result = whisperx.assign_word_speakers(diarize_segments, result, fill_nearest=False)
            elif action == "transcribe" and "timeline" in inp:
                # User provided timeline from previous step
                provided_timeline = pd.DataFrame(inp["timeline"])
                result = whisperx.assign_word_speakers(provided_timeline, result, fill_nearest=False)

            # 5. Split segments at word-level speaker boundaries
            print("✂️ Splitting segments by word-level speaker assignments...")
            split_segments = split_by_word_speakers(result["segments"])
            
            # 6. Format Result for server.py compatibility
            final_segments = []
            for seg in split_segments:
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
