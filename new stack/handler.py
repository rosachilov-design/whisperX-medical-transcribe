import os
import time
import subprocess
import re
import math
from pathlib import Path
import json
import urllib.request
import runpod
import torch

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# --- Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Loading faster-whisper on {device}...")
model = WhisperModel("turbo", device=device, compute_type=compute_type)

HF_TOKEN = os.getenv("HF_TOKEN")
print("Loading pyannote diarization pipeline...")
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    if diarization_pipeline:
        diarization_pipeline.to(torch.device(device))
except Exception as e:
    print(f"Error loading pyannote: {e}")
    diarization_pipeline = None

# --- Helper functions ---

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"

def get_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return float(result.stdout.strip())

def clean_hallucinations(text: str) -> str:
    hallucination_patterns = [
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
    for pattern in hallucination_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def get_speaker_for_word(timeline, word_start, word_end):
    if not timeline:
        return "Unknown"
    best_speaker = None
    best_overlap = 0
    for entry in timeline:
        overlap_start = max(word_start, entry["start"])
        overlap_end = min(word_end, entry["end"])
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = entry["speaker"]
    if best_speaker:
        return best_speaker
    mid = (word_start + word_end) / 2
    nearest = min(timeline, key=lambda s: min(abs(s["start"] - mid), abs(s["end"] - mid)))
    return nearest["speaker"]


# --- Core Pipeline Actions ---

def do_diarize(file_path: Path):
    """Only run Pyannote diarization and return the timeline."""
    if not diarization_pipeline:
        return {"error": "Pipeline not loaded"}

    print(f"Running diarization on {file_path}...")
    wav_path = file_path.with_suffix('.converted.wav')
    cmd = ["ffmpeg", "-y", "-i", str(file_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    import soundfile as sf
    import numpy as np
    data, sample_rate = sf.read(str(wav_path), dtype='float32')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    waveform = torch.from_numpy(data)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    
    diarize_output = diarization_pipeline(audio_input, min_speakers=2)
    annotation = getattr(diarize_output, 'speaker_diarization', diarize_output)
    
    timeline = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        
    os.remove(wav_path)
    return {"timeline": timeline}


def do_transcribe(file_path: Path, timeline: list):
    """Only run faster-whisper context-aware chunk transcription using the provided timeline."""
    natural_chunks = []
    if timeline:
        current_chunk_turns = []
        chunk_start_time = timeline[0]["start"]
        for i, entry in enumerate(timeline):
            current_chunk_turns.append(entry)
            elapsed = entry["end"] - chunk_start_time
            is_last = (i == len(timeline) - 1)
            if elapsed >= 30 or is_last:
                natural_chunks.append({"start": chunk_start_time, "end": entry["end"], "turns": current_chunk_turns})
                if not is_last:
                    current_chunk_turns = []
                    chunk_start_time = timeline[i+1]["start"]
    
    if not natural_chunks:
        duration = get_duration(file_path)
        natural_chunks = [{"start": i*30, "end": min((i+1)*30, duration)} for i in range(math.ceil(duration/30))]

    speaker_map = {}
    speaker_counter = 0
    all_speaker_words = []

    for i, chunk in enumerate(natural_chunks):
        actual_start = chunk["start"]
        actual_end = chunk["end"]
        
        pad = 10.0
        start_time = max(0, actual_start - pad)
        end_time = actual_end + pad
        duration_s = end_time - start_time
        if duration_s <= 0:
            continue
            
        chunk_path = file_path.parent / f"chunk_{i}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration_s),
            "-i", str(file_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(chunk_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not chunk_path.exists(): continue
        
        previous_context = ""
        if all_speaker_words:
             previous_context = " ".join([sw["word"] for sw in all_speaker_words[-50:]])
             
        try:
             segments, _ = model.transcribe(
                 str(chunk_path), language="ru", word_timestamps=True,
                 condition_on_previous_text=True,
                 initial_prompt=previous_context if previous_context else "Это аудиозапись беседы или интервью."
             )
             segments = list(segments)
             for segment in segments:
                 words = getattr(segment, "words", [])
                 if not words:
                     text = re.sub(r'\[.*?\]', '', segment.text).strip()
                     if text:
                         abs_start = segment.start + start_time
                         abs_end = segment.end + start_time
                         midpoint = (abs_start + abs_end) / 2.0
                         if actual_start <= midpoint <= actual_end:
                             all_speaker_words.append({
                                 "word": text, "start": abs_start, "end": abs_end,
                                 "speaker_raw": get_speaker_for_word(timeline, abs_start, abs_end)
                             })
                     continue
                 for w in words:
                     word_text = getattr(w, "word", "").strip()
                     if word_text:
                         abs_start = w.start + start_time
                         abs_end = w.end + start_time
                         midpoint = (abs_start + abs_end) / 2.0
                         if actual_start <= midpoint <= actual_end:
                             all_speaker_words.append({
                                 "word": word_text, "start": abs_start, "end": abs_end,
                                 "speaker_raw": get_speaker_for_word(timeline, abs_start, abs_end)
                             })
        except Exception as e:
             print(f"Skipping chunk {i}: {e}")
             
        os.remove(chunk_path)

    # Align and Finalize
    final_segments = []
    if all_speaker_words:
        cur_raw = all_speaker_words[0]["speaker_raw"]
        cur_start = all_speaker_words[0]["start"]
        cur_words = [all_speaker_words[0]["word"]]
        
        for sw in all_speaker_words[1:]:
            if sw["speaker_raw"] == cur_raw:
                cur_words.append(sw["word"])
            else:
                if cur_raw not in speaker_map:
                    speaker_counter += 1
                    speaker_map[cur_raw] = f"Speaker {speaker_counter}"
                
                text = clean_hallucinations(" ".join(cur_words))
                if text:
                    final_segments.append({
                        "start": cur_start, "timestamp": format_timestamp(cur_start),
                        "text": text, "speaker": speaker_map[cur_raw]
                    })
                cur_raw = sw["speaker_raw"]
                cur_start = sw["start"]
                cur_words = [sw["word"]]
                
        if cur_words:
            if cur_raw not in speaker_map:
                speaker_counter += 1
                speaker_map[cur_raw] = f"Speaker {speaker_counter}"
            text = clean_hallucinations(" ".join(cur_words))
            if text:
                final_segments.append({
                    "start": cur_start, "timestamp": format_timestamp(cur_start),
                    "text": text, "speaker": speaker_map[cur_raw]
                })

    smoothed = []
    for seg in final_segments:
        if smoothed and seg["speaker"] == smoothed[-1]["speaker"]:
            smoothed[-1]["text"] += " " + seg["text"]
        else:
            smoothed.append(seg.copy())

    return {"result": smoothed}


# --- Serverless Handler Entrypoint ---

def handler(event):
    job_input = event.get('input', {})
    action = job_input.get('action', 'diarize')
    audio_url = job_input.get('audio')
    
    if not audio_url:
         return {"error": "Missing 'audio' input URL"}

    file_ext = audio_url.split("?")[0].split('.')[-1][-4:]
    if not file_ext.startswith("."): file_ext = ".m4a"
    local_path = Path("/tmp/downloaded_audio" + file_ext)

    print(f"Downloading audio from {audio_url}")
    urllib.request.urlretrieve(audio_url, str(local_path))
    
    try:
        if action == "diarize":
            result = do_diarize(local_path)
        elif action == "transcribe":
            timeline = job_input.get('timeline', [])
            result = do_transcribe(local_path, timeline)
        else:
            result = {"error": f"Unknown action: {action}"}
    finally:
         if local_path.exists():
              os.remove(local_path)

    return result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
