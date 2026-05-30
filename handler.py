import os
import runpod
import whisperx
import torch
import gc
import json
import re
import requests
import tempfile
import subprocess
import time
from collections import Counter
import pandas as pd

# ─── Config & Init ───
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this RunPod image. "
        "Start the worker on a GPU pod with a compatible NVIDIA driver."
    )

DEVICE = "cuda"
BATCH_SIZE = 16 
COMPUTE_TYPE = "float32" # Use full precision to avoid glitches/cutoffs
MODEL_DIR = "/app/models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
UNKNOWN_SPEAKER = "Unknown"
ZOOM_DIARIZATION_MODE = os.environ.get("ZOOM_DIARIZATION_MODE", "robust").strip().lower()
OCR_MAPPING_CONFIDENCE_THRESHOLD = 0.55
OCR_MAPPING_MIN_OVERLAP_SECONDS = 2.0
OCR_SAMPLE_FPS = 2
OCR_MIN_CONFIDENCE = 0.10
OCR_FUZZY_THRESHOLD = 75
OCR_AUTO_CONFIDENCE = 0.10
OCR_AUTO_CLUSTER_THRESHOLD = 85
OCR_AUTO_MIN_CLUSTER_COUNT = 3
OCR_FILL_GAPS_SECONDS = 4.0
OCR_MIN_SAME_NAME_FRAMES = 2
OCR_MIN_SEGMENT_SECONDS = 0.75
STAGE_PROGRESS_LOG_SECONDS = float(os.environ.get("STAGE_PROGRESS_LOG_SECONDS", "30"))
PROGRESS_UPLOAD_SECONDS = float(os.environ.get("PROGRESS_UPLOAD_SECONDS", "1"))
OCR_REGIONS = [
    {
        "name": "bottom_left_name_bar",
        "x1": 0.00,
        "y1": 0.82,
        "x2": 0.45,
        "y2": 1.0,
        "priority": 1,
    },
    {
        "name": "top_right_screen_share_name_bar",
        "x1": 0.817,
        "y1": 0.158,
        "x2": 0.994,
        "y2": 0.202,
        "priority": 2,
    },
    {
        "name": "right_side_thumbnail_column",
        "x1": 0.78,
        "y1": 0.12,
        "x2": 1.0,
        "y2": 0.42,
        "priority": 2,
    },
]
SHORT_ACKNOWLEDGEMENT_MAX_DURATION = 1.35
SHORT_ACKNOWLEDGEMENT_MAX_TOKENS = 4
SHORT_ACKNOWLEDGEMENT_CONTEXT_GAP = 1.0
SPEAKER_SWITCH_MIN_WORDS = 2
SPEAKER_SWITCH_MIN_DURATION = 0.45
FRAGMENT_MAX_WORDS = 3
FRAGMENT_MAX_DURATION = 1.2
FRAGMENT_CONTEXT_GAP = 0.75
SENTENCE_END_PUNCTUATION = ".?!…"
CONTINUATION_TOKENS = {
    "а",
    "бы",
    "в",
    "вот",
    "для",
    "же",
    "и",
    "из",
    "или",
    "как",
    "когда",
    "ли",
    "на",
    "не",
    "но",
    "о",
    "по",
    "под",
    "при",
    "про",
    "с",
    "со",
    "то",
    "у",
    "что",
    "это",
}
SHORT_ACKNOWLEDGEMENT_PHRASES = {
    "ага",
    "да",
    "да да",
    "мгм",
    "ну да",
    "ой да",
    "отлично",
    "понятно",
    "так",
    "так хорошо",
    "угу",
    "хорошо",
}
SHORT_ACKNOWLEDGEMENT_TOKENS = {
    "ага",
    "да",
    "мгм",
    "ну",
    "ой",
    "отлично",
    "понятно",
    "так",
    "угу",
    "хорошо",
}

# Global cache for models
MODELS = {
    "whisper": None,
    "align": {},
    "diarize": None,
    "ocr": None
}

def _elapsed_seconds(started_at):
    return f"{time.monotonic() - started_at:.1f}s"


def _stage_log(message):
    print(f"[stage] {message}", flush=True)


def _audio_duration_seconds(audio, sample_rate=16000):
    try:
        return max(0.0, float(len(audio)) / float(sample_rate))
    except Exception:
        return 0.0


def _format_progress_time(seconds):
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _build_s3_progress_writer(s3_creds, progress_key):
    if not s3_creds or not progress_key:
        return None

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_creds["endpoint"],
        region_name=s3_creds["region"],
        aws_access_key_id=s3_creds["access_key"],
        aws_secret_access_key=s3_creds["secret_key"],
        config=Config(signature_version="s3v4"),
    )
    bucket = s3_creds["bucket"]
    last_uploaded_at = [0.0]

    def write_progress(payload, force=False):
        now = time.monotonic()
        if not force and now - last_uploaded_at[0] < PROGRESS_UPLOAD_SECONDS:
            return
        last_uploaded_at[0] = now
        s3.put_object(
            Bucket=bucket,
            Key=progress_key,
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    return write_progress


def get_whisper():
    if MODELS["whisper"] is None:
        started_at = time.monotonic()
        _stage_log("Loading Whisper model (large-v3, float32, high sensitivity VAD)...")
        vad_options = {"vad_onset": 0.450, "vad_offset": 0.363}
        MODELS["whisper"] = whisperx.load_model(
            "large-v3", 
            DEVICE, 
            compute_type=COMPUTE_TYPE, 
            download_root=MODEL_DIR,
            vad_options=vad_options
        )
        _stage_log(f"Whisper model ready in {_elapsed_seconds(started_at)}.")
    return MODELS["whisper"]

def get_align(lang):
    if lang not in MODELS["align"]:
        started_at = time.monotonic()
        _stage_log(f"Loading alignment model ({lang})...")
        MODELS["align"][lang] = whisperx.load_align_model(language_code=lang, device=DEVICE, model_dir=MODEL_DIR)
        _stage_log(f"Alignment model ({lang}) ready in {_elapsed_seconds(started_at)}.")
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
        started_at = time.monotonic()
        _stage_log(f"Loading diarization pipeline ({DIARIZATION_MODEL})...")
        MODELS["diarize"] = _build_diarization_pipeline(DIARIZATION_MODEL)
        _stage_log(f"Diarization pipeline loaded in {_elapsed_seconds(started_at)}.")
        
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
                clustering["threshold"] = 0.45
                changed.append("clustering.threshold=0.45")

            segmentation = params.get("segmentation")
            if isinstance(segmentation, dict):
                if "min_duration_off" in segmentation:
                    segmentation["min_duration_off"] = 0.15
                    changed.append("min_duration_off=0.15")
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


def get_ocr():
    if MODELS["ocr"] is None:
        started_at = time.monotonic()
        _stage_log("Loading PaddleOCR model...")
        from paddle_ocr_factory import create_zoom_paddle_ocr

        MODELS["ocr"] = create_zoom_paddle_ocr()
        _stage_log(f"PaddleOCR model ready in {_elapsed_seconds(started_at)}.")
    return MODELS["ocr"]


def is_mp4_path(path):
    return str(path or "").lower().split("?")[0].endswith(".mp4")


def extract_audio_for_transcription(video_path):
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path


def _clean_ocr_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;|/\\_-—–[](){}")
    return text


def _normalize_known_speakers(value):
    if not value:
        return []

    raw_items = value.split(";") if isinstance(value, str) else value
    normalized = []
    seen = set()
    for item in raw_items:
        name = _clean_ocr_text(item)
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _is_name_like_text(text, min_confidence, confidence):
    text = _clean_ocr_text(text)
    if confidence < min_confidence:
        return False
    if not 4 <= len(text) <= 40:
        return False
    if len(text.split()) > 4:
        return False
    if not re.search(r"[A-Za-zА-Яа-яЁё]", text):
        return False
    if re.search(r"\d", text):
        return False

    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
    if letters / max(len(text), 1) < 0.50:
        return False

    lowered = text.lower()
    stop_words = (
        "zoom",
        "recording",
        "screen",
        "share",
        "sharing",
        "подключение",
        "демонстрация",
        "запись",
        "экран",
    )
    return not any(word in lowered for word in stop_words)


def _parse_paddle_result(result):
    texts = []
    confidences = []

    # Handle PaddleOCR v5+ dict format (may be wrapped in a list)
    result_dict = result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) else (result if isinstance(result, dict) else None)
    if result_dict is not None:
        rec_texts = result_dict.get("rec_texts") or []
        rec_scores = result_dict.get("rec_scores") or []
        for t, s in zip(rec_texts, rec_scores):
            if isinstance(t, str):
                texts.append(t)
                try:
                    confidences.append(float(s))
                except Exception:
                    pass
        text = _clean_ocr_text(" ".join(texts))
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, confidence

    def visit(node):
        if not node:
            return
        if isinstance(node, tuple) and len(node) >= 2 and isinstance(node[0], str):
            texts.append(node[0])
            try:
                confidences.append(float(node[1]))
            except Exception:
                pass
            return
        if isinstance(node, list):
            if (
                len(node) >= 2
                and isinstance(node[1], tuple)
                and len(node[1]) >= 2
                and isinstance(node[1][0], str)
            ):
                texts.append(node[1][0])
                try:
                    confidences.append(float(node[1][1]))
                except Exception:
                    pass
                return
            for child in node:
                visit(child)

    visit(result)
    text = _clean_ocr_text(" ".join(texts))
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def _crop_region(frame, region):
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(w * region["x1"])))
    y1 = max(0, min(h, int(h * region["y1"])))
    x2 = max(0, min(w, int(w * region["x2"])))
    y2 = max(0, min(h, int(h * region["y2"])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    import cv2

    return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _read_ocr_candidate(ocr, frame, region):
    crop = _crop_region(frame, region)
    if crop.size == 0:
        return {
            "region": region["name"],
            "raw_text": None,
            "confidence": 0.0,
            "priority": region.get("priority", 0),
        }

    try:
        result = ocr.ocr(crop, cls=False)
    except TypeError:
        result = ocr.ocr(crop)
    raw_text, confidence = _parse_paddle_result(result)
    return {
        "region": region["name"],
        "raw_text": raw_text or None,
        "confidence": round(confidence, 4),
        "priority": region.get("priority", 0),
    }


def _cluster_discovered_speakers(raw_candidates):
    try:
        from rapidfuzz import fuzz
    except Exception as exc:
        raise RuntimeError("rapidfuzz is required for OCR speaker discovery.") from exc

    clusters = []
    for text in raw_candidates:
        clean = _clean_ocr_text(text)
        if not clean:
            continue

        best_cluster = None
        best_score = 0
        for cluster in clusters:
            score = fuzz.WRatio(clean, cluster["canonical"])
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster and best_score >= OCR_AUTO_CLUSTER_THRESHOLD:
            best_cluster["items"].append(clean)
            counts = Counter(best_cluster["items"])
            best_cluster["canonical"] = counts.most_common(1)[0][0]
        else:
            clusters.append({"canonical": clean, "items": [clean]})

    discovered = []
    for cluster in clusters:
        if len(cluster["items"]) < OCR_AUTO_MIN_CLUSTER_COUNT:
            continue
        counts = Counter(cluster["items"])
        canonical, count = counts.most_common(1)[0]
        if not _is_name_like_text(canonical, 0.0, 1.0):
            continue
        discovered.append({
            "speaker": canonical,
            "raw_examples": [item for item, _ in counts.most_common(5)],
            "count": len(cluster["items"]),
        })

    # Merge highly similar discovered clusters (e.g. "Татьяна" and "атьяна" if both existed)
    try:
        from rapidfuzz import fuzz
        merged = []
        for item in discovered:
            clean = item["speaker"]
            best = None
            best_score = 0
            for existing in merged:
                score = fuzz.WRatio(clean, existing["speaker"])
                if score > best_score:
                    best_score = score
                    best = existing
            if best and best_score >= 90:
                best["count"] += item["count"]
                best["raw_examples"] = list(dict.fromkeys(best["raw_examples"] + item["raw_examples"]))[:5]
                if len(item["speaker"]) > len(best["speaker"]):
                    best["speaker"] = item["speaker"]
            else:
                merged.append(dict(item))
        discovered = merged
    except Exception:
        pass

    return sorted(discovered, key=lambda item: item["count"], reverse=True)


def _match_known_speaker(candidate, known_speakers):
    if not candidate.get("raw_text") or candidate.get("confidence", 0.0) < OCR_MIN_CONFIDENCE:
        return None
    try:
        from rapidfuzz import process, fuzz
    except Exception as exc:
        raise RuntimeError("rapidfuzz is required for OCR speaker matching.") from exc

    match = process.extractOne(candidate["raw_text"], known_speakers, scorer=fuzz.WRatio)
    if not match:
        return None
    name, fuzzy_score, _ = match
    if fuzzy_score < OCR_FUZZY_THRESHOLD:
        return None
    combined_score = float(candidate["confidence"]) * (float(fuzzy_score) / 100.0)
    return {
        "speaker": name,
        "score": round(combined_score, 4),
        "fuzzy_score": round(float(fuzzy_score), 2),
    }


def _match_discovered_speaker(candidate, discovered_speakers):
    if not _is_name_like_text(candidate.get("raw_text"), OCR_AUTO_CONFIDENCE, candidate.get("confidence", 0.0)):
        return None
    try:
        from rapidfuzz import process, fuzz
    except Exception as exc:
        raise RuntimeError("rapidfuzz is required for OCR speaker discovery matching.") from exc

    names = [speaker["speaker"] for speaker in discovered_speakers]
    match = process.extractOne(candidate["raw_text"], names, scorer=fuzz.WRatio)
    if not match:
        return None
    name, fuzzy_score, _ = match
    if fuzzy_score < OCR_AUTO_CLUSTER_THRESHOLD:
        return None
    combined_score = float(candidate["confidence"]) * (float(fuzzy_score) / 100.0)
    return {
        "speaker": name,
        "score": round(combined_score, 4),
        "fuzzy_score": round(float(fuzzy_score), 2),
    }


def _select_ocr_candidate(candidates, known_speakers, discovered_speakers):
    selected = None
    for candidate in candidates:
        match = (
            _match_known_speaker(candidate, known_speakers)
            if known_speakers
            else _match_discovered_speaker(candidate, discovered_speakers)
        )
        if not match:
            continue

        enriched = dict(candidate)
        enriched.update(match)
        if selected is None:
            selected = enriched
            continue

        score_delta = enriched["score"] - selected["score"]
        if score_delta > 0.05 or (
            abs(score_delta) <= 0.05 and enriched.get("priority", 0) > selected.get("priority", 0)
        ):
            selected = enriched

    return selected


def _fill_internal_unknown_gaps(samples):
    filled = [dict(sample) for sample in samples]
    index = 0
    while index < len(filled):
        if filled[index]["speaker"] != UNKNOWN_SPEAKER:
            index += 1
            continue

        gap_start = index
        while index < len(filled) and filled[index]["speaker"] == UNKNOWN_SPEAKER:
            index += 1
        gap_end = index - 1

        prev_speaker = filled[gap_start - 1]["speaker"] if gap_start > 0 else UNKNOWN_SPEAKER
        next_speaker = filled[index]["speaker"] if index < len(filled) else UNKNOWN_SPEAKER
        duration = filled[gap_end]["t"] - filled[gap_start]["t"]

        if (
            prev_speaker != UNKNOWN_SPEAKER
            and prev_speaker == next_speaker
            and duration <= OCR_FILL_GAPS_SECONDS
        ):
            for fill_index in range(gap_start, gap_end + 1):
                filled[fill_index]["speaker"] = prev_speaker

    return filled


def _confirm_ocr_switches(samples):
    confirmed = []
    current = UNKNOWN_SPEAKER
    candidate = None
    candidate_count = 0

    for sample in samples:
        name = sample["speaker"]
        output = dict(sample)

        if name == UNKNOWN_SPEAKER:
            candidate = None
            candidate_count = 0
            output["speaker"] = UNKNOWN_SPEAKER
            confirmed.append(output)
            continue

        if name == current:
            candidate = None
            candidate_count = 0
            output["speaker"] = current
            confirmed.append(output)
            continue

        if name == candidate:
            candidate_count += 1
        else:
            candidate = name
            candidate_count = 1

        if candidate_count >= OCR_MIN_SAME_NAME_FRAMES:
            current = candidate
            candidate = None
            candidate_count = 0

        output["speaker"] = current
        confirmed.append(output)

    return confirmed


def _samples_to_intervals(samples, duration):
    if not samples:
        return [{"start": 0.0, "end": round(duration, 3), "speaker": UNKNOWN_SPEAKER, "source": "zoom_ocr"}]

    intervals = []
    current_speaker = samples[0]["speaker"]
    current_start = samples[0]["t"]

    for sample in samples[1:]:
        if sample["speaker"] == current_speaker:
            continue
        intervals.append({
            "start": round(current_start, 3),
            "end": round(sample["t"], 3),
            "speaker": current_speaker,
            "source": "zoom_ocr",
        })
        current_start = sample["t"]
        current_speaker = sample["speaker"]

    intervals.append({
        "start": round(current_start, 3),
        "end": round(duration, 3),
        "speaker": current_speaker,
        "source": "zoom_ocr",
    })

    return _cleanup_rare_speaker_blips(_merge_short_ocr_intervals(_merge_adjacent_ocr_intervals(intervals)))


def _merge_adjacent_ocr_intervals(intervals):
    merged = []
    for interval in intervals:
        if interval["end"] <= interval["start"]:
            continue
        if merged and merged[-1]["speaker"] == interval["speaker"]:
            merged[-1]["end"] = interval["end"]
        else:
            merged.append(dict(interval))
    return merged


def _merge_short_ocr_intervals(intervals):
    intervals = [dict(interval) for interval in intervals]
    index = 0
    while index < len(intervals):
        interval = intervals[index]
        duration = interval["end"] - interval["start"]
        if duration >= OCR_MIN_SEGMENT_SECONDS or len(intervals) == 1:
            index += 1
            continue

        if index > 0:
            intervals[index - 1]["end"] = interval["end"]
            intervals.pop(index)
            continue

        intervals[index + 1]["start"] = interval["start"]
        intervals.pop(index)

    return _merge_adjacent_ocr_intervals(intervals)


def _cleanup_rare_speaker_blips(intervals):
    """
    Merge very short segments from speakers with very little total time.
    These are typically OCR misreads like 'puna' instead of 'Ирина'.
    """
    if not intervals:
        return intervals

    # Calculate total duration per speaker
    speaker_durations = {}
    for item in intervals:
        spk = item["speaker"]
        dur = max(0.0, item["end"] - item["start"])
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

    cleaned = [dict(interval) for interval in intervals]
    index = 0
    while index < len(cleaned):
        item = cleaned[index]
        duration = max(0.0, item["end"] - item["start"])
        spk = item["speaker"]
        total_spk_dur = speaker_durations.get(spk, 0.0)

        # Blip: < 2.0s AND speaker has < 5s total (rare/garbage) AND not Unknown
        if spk != UNKNOWN_SPEAKER and duration < 2.0 and total_spk_dur < 5.0:
            prev = cleaned[index - 1] if index > 0 else None
            next_item = cleaned[index + 1] if index + 1 < len(cleaned) else None

            # Prefer merging into the neighbor with longer total duration
            prev_dur = speaker_durations.get(prev["speaker"], 0.0) if prev else 0.0
            next_dur = speaker_durations.get(next_item["speaker"], 0.0) if next_item else 0.0

            if prev and next_item and prev["speaker"] == next_item["speaker"]:
                # Both neighbors agree — merge into them
                cleaned[index - 1]["end"] = item["end"]
                cleaned.pop(index)
                speaker_durations[prev["speaker"]] = speaker_durations.get(prev["speaker"], 0.0) + duration
                speaker_durations[spk] = max(0.0, speaker_durations.get(spk, 0.0) - duration)
                continue
            elif next_item and (not prev or next_dur >= prev_dur):
                cleaned[index + 1]["start"] = item["start"]
                cleaned.pop(index)
                speaker_durations[next_item["speaker"]] = speaker_durations.get(next_item["speaker"], 0.0) + duration
                speaker_durations[spk] = max(0.0, speaker_durations.get(spk, 0.0) - duration)
                continue
            elif prev:
                cleaned[index - 1]["end"] = item["end"]
                cleaned.pop(index)
                speaker_durations[prev["speaker"]] = speaker_durations.get(prev["speaker"], 0.0) + duration
                speaker_durations[spk] = max(0.0, speaker_durations.get(spk, 0.0) - duration)
                continue

        index += 1

    return _merge_adjacent_ocr_intervals(cleaned)


def _summarize_low_confidence_intervals(timeline):
    return [
        {"start": item["start"], "end": item["end"], "speaker": item["speaker"]}
        for item in timeline
        if item["speaker"] == UNKNOWN_SPEAKER
    ][:25]


def build_zoom_ocr_timeline(video_path, known_speakers=None, sample_fps=OCR_SAMPLE_FPS, regions=None):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python-headless is required for Zoom .mp4 OCR diarization.") from exc

    regions = regions or OCR_REGIONS
    known_speakers = _normalize_known_speakers(known_speakers)
    ocr = get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / native_fps) if native_fps else 0.0
    if duration <= 0:
        duration = 0.0

    raw_samples = []
    discovery_candidates = []
    step = 1.0 / float(sample_fps)
    total_samples = max(1, int(duration * sample_fps) + 1) if duration > 0 else 1
    ocr_started_at = time.monotonic()
    next_progress_log_at = ocr_started_at + STAGE_PROGRESS_LOG_SECONDS
    samples_read = 0
    _stage_log(
        "Zoom OCR scan started: "
        f"duration={duration:.1f}s, sample_fps={sample_fps}, regions={len(regions)}, "
        f"estimated_ocr_calls={total_samples * len(regions)}."
    )
    t = 0.0
    while t <= duration + 0.001:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        samples_read += 1

        candidates = [_read_ocr_candidate(ocr, frame, region) for region in regions]
        raw_samples.append({"t": round(t, 3), "candidates": candidates})

        if not known_speakers:
            for candidate in candidates:
                if _is_name_like_text(candidate.get("raw_text"), OCR_AUTO_CONFIDENCE, candidate.get("confidence", 0.0)):
                    discovery_candidates.append(candidate["raw_text"])

        now = time.monotonic()
        if now >= next_progress_log_at:
            pct = min(100.0, samples_read * 100.0 / total_samples)
            _stage_log(
                "Zoom OCR progress: "
                f"{samples_read}/{total_samples} samples ({pct:.1f}%), "
                f"video_t={t:.1f}s/{duration:.1f}s, elapsed={_elapsed_seconds(ocr_started_at)}."
            )
            next_progress_log_at = now + STAGE_PROGRESS_LOG_SECONDS
        t += step

    cap.release()
    _stage_log(
        "Zoom OCR frame scan complete: "
        f"{samples_read}/{total_samples} samples, elapsed={_elapsed_seconds(ocr_started_at)}."
    )

    discovered_speakers = [] if known_speakers else _cluster_discovered_speakers(discovery_candidates)

    selected_samples = []
    for sample in raw_samples:
        selected = _select_ocr_candidate(sample["candidates"], known_speakers, discovered_speakers)
        selected_samples.append({
            "t": sample["t"],
            "speaker": selected["speaker"] if selected else UNKNOWN_SPEAKER,
            "raw_text": selected.get("raw_text") if selected else None,
            "confidence": selected.get("confidence", 0.0) if selected else 0.0,
            "region": selected.get("region") if selected else None,
        })

    smoothed_samples = _confirm_ocr_switches(_fill_internal_unknown_gaps(selected_samples))
    timeline = _samples_to_intervals(smoothed_samples, duration)
    total_duration = sum(max(0.0, item["end"] - item["start"]) for item in timeline)
    unknown_duration = sum(
        max(0.0, item["end"] - item["start"])
        for item in timeline
        if item["speaker"] == UNKNOWN_SPEAKER
    )

    summary = {
        "source": "zoom_ocr",
        "sample_fps": sample_fps,
        "regions": regions,
        "known_speakers": known_speakers,
        "discovered_speakers": discovered_speakers,
        "unknown_rate": round(unknown_duration / total_duration, 4) if total_duration else 1.0,
        "low_confidence_intervals": _summarize_low_confidence_intervals(timeline),
    }
    _stage_log(
        "Zoom OCR timeline ready: "
        f"intervals={len(timeline)}, unknown_rate={summary['unknown_rate']}, "
        f"elapsed={_elapsed_seconds(ocr_started_at)}."
    )
    return timeline, summary


def _format_diarization_timeline(diarize_segments, source="pyannote"):
    timeline = []
    for _, row in diarize_segments.iterrows():
        timeline.append({
            "start": round(float(row["start"]), 3),
            "end": round(float(row["end"]), 3),
            "speaker": str(row["speaker"]),
            "source": source,
        })
    return timeline


def _run_pyannote_diarization(audio, min_speakers, max_speakers, num_speakers=None, progress_writer=None):
    pipe = get_diarize()
    exact = int(num_speakers) if num_speakers else None
    min_value = int(min_speakers) if min_speakers else None
    max_value = int(max_speakers) if max_speakers else None
    duration = _audio_duration_seconds(audio)
    started_at = time.monotonic()
    _stage_log(f"Pyannote diarization started (min={min_value}, max={max_value}, num={exact})...")

    kwargs = {}
    if min_value:
        kwargs["min_speakers"] = min_value
    if max_value:
        kwargs["max_speakers"] = max_value
    if exact:
        kwargs["num_speakers"] = exact

    def report_progress(current, total, unit="seconds", label="Pyannote diarization"):
        if not progress_writer:
            return
        current = max(0.0, float(current or 0.0))
        total = max(current, float(total or 0.0))
        pct = min(99, int(current * 100 / total)) if total else 0
        if unit == "seconds":
            message = f"{label}: {_format_progress_time(current)} / {_format_progress_time(total)}"
        else:
            message = f"{label}: {int(current)}/{int(total)} {unit}"
        progress_writer({
            "stage": "pyannote",
            "current": current,
            "total": total,
            "unit": unit,
            "progress": pct,
            "message": message,
        })

    def hook(step_name, step_artifact=None, file=None, **hook_kwargs):
        if step_artifact is not None and hasattr(step_artifact, "end") and duration > 0:
            report_progress(step_artifact.end, duration)
            return
        completed = hook_kwargs.get("completed") or hook_kwargs.get("current")
        total = hook_kwargs.get("total")
        if completed is not None and total:
            report_progress(completed, total, unit="steps", label=f"Pyannote {step_name}")

    if progress_writer:
        progress_writer({
            "stage": "pyannote",
            "current": 0,
            "total": duration,
            "unit": "seconds",
            "progress": 0,
            "message": f"Pyannote diarization: 00:00 / {_format_progress_time(duration)}",
        }, force=True)

    try:
        segments = pipe(audio, hook=hook, **kwargs)
    except TypeError as exc:
        if "hook" not in str(exc):
            raise
        _stage_log("Pyannote progress hook is not supported by this pipeline; continuing without detailed progress.")
        segments = pipe(audio, **kwargs)

    if progress_writer:
        progress_writer({
            "stage": "pyannote",
            "current": duration,
            "total": duration,
            "unit": "seconds",
            "progress": 99,
            "message": f"Pyannote diarization: {_format_progress_time(duration)} / {_format_progress_time(duration)}",
        }, force=True)

    try:
        segment_count = len(segments)
    except Exception:
        segment_count = "unknown"
    _stage_log(f"Pyannote diarization complete: segments={segment_count}, elapsed={_elapsed_seconds(started_at)}.")
    return segments


def _overlap_seconds(left, right):
    return max(
        0.0,
        min(float(left["end"]), float(right["end"])) - max(float(left["start"]), float(right["start"])),
    )


def map_pyannote_speakers_to_ocr_names(pyannote_timeline, ocr_timeline):
    votes = {}
    totals = {}
    pyannote_speakers = {
        turn.get("speaker", UNKNOWN_SPEAKER)
        for turn in pyannote_timeline
        if turn.get("speaker") and turn.get("speaker") != UNKNOWN_SPEAKER
    }

    for pyannote_turn in pyannote_timeline:
        pyannote_speaker = pyannote_turn.get("speaker", UNKNOWN_SPEAKER)
        if not pyannote_speaker or pyannote_speaker == UNKNOWN_SPEAKER:
            continue

        votes.setdefault(pyannote_speaker, {})
        for ocr_turn in ocr_timeline:
            ocr_speaker = ocr_turn.get("speaker", UNKNOWN_SPEAKER)
            if not ocr_speaker or ocr_speaker == UNKNOWN_SPEAKER:
                continue

            overlap = _overlap_seconds(pyannote_turn, ocr_turn)
            if overlap <= 0:
                continue

            votes[pyannote_speaker][ocr_speaker] = votes[pyannote_speaker].get(ocr_speaker, 0.0) + overlap
            totals[pyannote_speaker] = totals.get(pyannote_speaker, 0.0) + overlap

    mapping = {}
    details = {}
    for pyannote_speaker in sorted(pyannote_speakers):
        name_votes = votes.get(pyannote_speaker, {})
        if not name_votes:
            details[pyannote_speaker] = {
                "name": UNKNOWN_SPEAKER,
                "accepted": False,
                "confidence": 0.0,
                "overlap_seconds": 0.0,
                "votes": {},
            }
            continue

        best_name, best_seconds = max(name_votes.items(), key=lambda item: item[1])
        total = totals.get(pyannote_speaker, 0.0)
        confidence = best_seconds / total if total > 0 else 0.0
        accepted = (
            best_seconds >= OCR_MAPPING_MIN_OVERLAP_SECONDS
            and confidence >= OCR_MAPPING_CONFIDENCE_THRESHOLD
        )

        if accepted:
            mapping[pyannote_speaker] = best_name

        details[pyannote_speaker] = {
            "name": best_name if accepted else UNKNOWN_SPEAKER,
            "accepted": accepted,
            "confidence": round(confidence, 4),
            "overlap_seconds": round(best_seconds, 3),
            "votes": {
                name: round(seconds, 3)
                for name, seconds in sorted(name_votes.items(), key=lambda item: item[1], reverse=True)
            },
        }

    return mapping, details


def apply_speaker_mapping_to_timeline(pyannote_timeline, speaker_mapping, fallback_speaker=UNKNOWN_SPEAKER):
    named_timeline = []
    for turn in pyannote_timeline:
        raw_speaker = turn.get("speaker", UNKNOWN_SPEAKER)
        mapped = speaker_mapping.get(raw_speaker, fallback_speaker)
        named_timeline.append({
            "start": turn["start"],
            "end": turn["end"],
            "speaker": mapped,
            "source": "pyannote_ocr_map" if raw_speaker in speaker_mapping else "pyannote_ocr_unmapped",
            "speaker_raw": raw_speaker,
        })
    return named_timeline


def _normalize_speaker_label(label):
    if label is None:
        return UNKNOWN_SPEAKER

    if isinstance(label, float) and pd.isna(label):
        return UNKNOWN_SPEAKER

    text = str(label).strip()
    if not text or text.lower() == "nan":
        return UNKNOWN_SPEAKER

    return text


def _normalize_short_acknowledgement_text(text):
    normalized = str(text or "").lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _segment_duration(segment):
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    return max(0.0, end - start)


def _segment_gap(left, right):
    if not left or not right:
        return float("inf")

    left_end = float(left.get("end", left.get("start", 0.0)) or 0.0)
    right_start = float(right.get("start", 0.0) or 0.0)
    return max(0.0, right_start - left_end)


def _segment_token_count(segment):
    words = segment.get("words") or []
    if words:
        return len(words)

    text = str(segment.get("text", "") or "").strip()
    return len(text.split()) if text else 0


def _normalize_word(word):
    normalized_word = dict(word)
    normalized_word["speaker"] = _normalize_speaker_label(word.get("speaker"))
    return normalized_word


def _merge_segment_words(left_words, right_words):
    merged = []
    for word in (left_words or []) + (right_words or []):
        merged.append(_normalize_word(word))
    return merged


def _extract_tokens(text):
    normalized = _normalize_short_acknowledgement_text(text)
    return normalized.split() if normalized else []


def _last_token(text):
    tokens = _extract_tokens(text)
    return tokens[-1] if tokens else ""


def _starts_with_lowercase(text):
    stripped = str(text or "").lstrip(' "\'([{-')
    return bool(stripped) and stripped[0].islower()


def _ends_with_sentence_punctuation(text):
    stripped = str(text or "").rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_END_PUNCTUATION


def _is_complete_short_acknowledgement_run(segment):
    return _is_short_acknowledgement_segment(segment) and _ends_with_sentence_punctuation(segment.get("text", ""))


def _is_fragment_candidate(segment):
    if not segment or _is_short_acknowledgement_segment(segment):
        return False

    return (
        _segment_token_count(segment) <= FRAGMENT_MAX_WORDS
        and _segment_duration(segment) <= FRAGMENT_MAX_DURATION
    )


def _build_speaker_runs(words):
    runs = []
    current_speaker = None
    current_words = []

    for word in words:
        normalized_word = _normalize_word(word)
        word_speaker = normalized_word["speaker"]

        if word_speaker != current_speaker and current_words:
            runs.append(_words_to_segment(current_words, current_speaker))
            current_words = []

        current_speaker = word_speaker
        current_words.append(normalized_word)

    if current_words:
        runs.append(_words_to_segment(current_words, current_speaker))

    return runs


def _should_start_new_turn(current_segment, candidate_segment):
    candidate_speaker = _normalize_speaker_label(candidate_segment.get("speaker"))
    current_speaker = _normalize_speaker_label(current_segment.get("speaker"))
    if candidate_speaker == current_speaker:
        return False

    if candidate_speaker == UNKNOWN_SPEAKER:
        return True

    if _is_complete_short_acknowledgement_run(candidate_segment):
        return True

    if _segment_token_count(candidate_segment) >= SPEAKER_SWITCH_MIN_WORDS:
        return True

    if _segment_duration(candidate_segment) >= SPEAKER_SWITCH_MIN_DURATION:
        return True

    if _ends_with_sentence_punctuation(current_segment.get("text", "")):
        return True

    return False


def _merge_segments(primary_segment, secondary_segment, speaker=None):
    speaker_to_use = _normalize_speaker_label(
        speaker if speaker is not None else primary_segment.get("speaker")
    )
    merged_words = _merge_segment_words(primary_segment.get("words"), secondary_segment.get("words"))
    if merged_words:
        return _words_to_segment(merged_words, speaker_to_use)

    merged_text = " ".join(
        part.strip() for part in [primary_segment.get("text", ""), secondary_segment.get("text", "")] if part and str(part).strip()
    ).strip()
    return {
        "start": primary_segment.get("start", secondary_segment.get("start", 0.0)),
        "end": secondary_segment.get("end", primary_segment.get("end", 0.0)),
        "text": merged_text,
        "speaker": speaker_to_use,
        "words": merged_words,
    }


def merge_adjacent_same_speaker_segments(segments):
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for segment in segments[1:]:
        normalized_segment = dict(segment)
        normalized_segment["speaker"] = _normalize_speaker_label(segment.get("speaker"))
        previous = merged[-1]
        if normalized_segment["speaker"] == previous["speaker"]:
            merged[-1] = _merge_segments(previous, normalized_segment, speaker=previous["speaker"])
        else:
            merged.append(normalized_segment)

    return merged


def _is_short_acknowledgement_segment(segment):
    text = _normalize_short_acknowledgement_text(segment.get("text", ""))
    if not text:
        return False

    tokens = text.split()
    if len(tokens) > SHORT_ACKNOWLEDGEMENT_MAX_TOKENS:
        return False

    if _segment_duration(segment) > SHORT_ACKNOWLEDGEMENT_MAX_DURATION:
        return False

    if text in SHORT_ACKNOWLEDGEMENT_PHRASES:
        return True

    return all(token in SHORT_ACKNOWLEDGEMENT_TOKENS for token in tokens)


def _fill_unlabeled_word_runs(words):
    """
    Conservatively fill runs of unlabeled words only when both surrounding
    labeled words agree on the speaker. Everything else stays Unknown.
    """
    if not words:
        return words

    normalized_words = [_normalize_word(word) for word in words]

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
    
    This function honors word-level speaker changes, but avoids opening a new
    turn for every microscopic flip. Tiny flips stay attached unless they are
    substantial enough, a complete acknowledgement, or clearly begin after a
    sentence boundary.
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
        runs = _build_speaker_runs(words)
        if not runs:
            continue

        current_segment = runs[0]
        for run in runs[1:]:
            if _should_start_new_turn(current_segment, run):
                new_segments.append(current_segment)
                current_segment = run
            else:
                current_segment = _merge_segments(
                    current_segment,
                    run,
                    speaker=current_segment.get("speaker"),
                )

        new_segments.append(current_segment)
    
    return new_segments


def repair_short_acknowledgement_speakers(segments):
    """
    Recover short acknowledgement segments that stayed unlabeled because the
    unlabeled word run happened at an original Whisper segment edge.

    We only relabel conservative cases: the short segment must be a simple
    acknowledgement ("да", "угу", "хорошо", etc.) and both neighboring
    segments must agree on the speaker within a tight timing window.
    """
    if not segments:
        return segments

    repaired = []
    for segment in segments:
        normalized_segment = dict(segment)
        normalized_segment["speaker"] = _normalize_speaker_label(segment.get("speaker"))
        repaired.append(normalized_segment)

    for index, segment in enumerate(repaired):
        if segment["speaker"] != UNKNOWN_SPEAKER:
            continue

        if not _is_short_acknowledgement_segment(segment):
            continue

        prev_segment = repaired[index - 1] if index > 0 else None
        next_segment = repaired[index + 1] if index + 1 < len(repaired) else None
        prev_speaker = _normalize_speaker_label(prev_segment.get("speaker")) if prev_segment else UNKNOWN_SPEAKER
        next_speaker = _normalize_speaker_label(next_segment.get("speaker")) if next_segment else UNKNOWN_SPEAKER

        if prev_speaker == UNKNOWN_SPEAKER or prev_speaker != next_speaker:
            continue

        if _segment_gap(prev_segment, segment) > SHORT_ACKNOWLEDGEMENT_CONTEXT_GAP:
            continue

        if _segment_gap(segment, next_segment) > SHORT_ACKNOWLEDGEMENT_CONTEXT_GAP:
            continue

        segment["speaker"] = prev_speaker

    return repaired


def repair_fragmented_turns(segments):
    """
    Reassign tiny broken fragments to the neighboring turn when they look like
    sentence continuation rather than a real interruption.
    """
    if not segments:
        return segments

    repaired = []
    for segment in segments:
        normalized_segment = dict(segment)
        normalized_segment["speaker"] = _normalize_speaker_label(segment.get("speaker"))
        repaired.append(normalized_segment)

    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(repaired) - 1:
            current = repaired[index]
            next_segment = repaired[index + 1]
            if current["speaker"] == next_segment["speaker"]:
                index += 1
                continue

            if _segment_gap(current, next_segment) > FRAGMENT_CONTEXT_GAP:
                index += 1
                continue

            current_fragment = _is_fragment_candidate(current)
            next_fragment = _is_fragment_candidate(next_segment)

            if current_fragment and not _is_complete_short_acknowledgement_run(current):
                if (
                    _starts_with_lowercase(next_segment.get("text", ""))
                    or _last_token(current.get("text", "")) in CONTINUATION_TOKENS
                    or not _ends_with_sentence_punctuation(current.get("text", ""))
                ):
                    repaired[index + 1] = _merge_segments(
                        current,
                        next_segment,
                        speaker=next_segment["speaker"],
                    )
                    del repaired[index]
                    changed = True
                    if index:
                        index -= 1
                    continue

            if next_fragment and not _is_complete_short_acknowledgement_run(next_segment):
                if (
                    _starts_with_lowercase(next_segment.get("text", ""))
                    or _last_token(current.get("text", "")) in CONTINUATION_TOKENS
                    or not _ends_with_sentence_punctuation(current.get("text", ""))
                ):
                    repaired[index] = _merge_segments(
                        current,
                        next_segment,
                        speaker=current["speaker"],
                    )
                    del repaired[index + 1]
                    changed = True
                    continue

            index += 1

    return merge_adjacent_same_speaker_segments(repaired)


def _words_to_segment(words, speaker):
    """Build a segment dict from a list of word dicts."""
    normalized_words = [_normalize_word(word) for word in words]
    texts = [w.get("word", "") for w in normalized_words]
    return {
        "start": normalized_words[0].get("start", 0.0),
        "end": normalized_words[-1].get("end", normalized_words[-1].get("start", 0.0)),
        "text": " ".join(texts).strip(),
        "speaker": _normalize_speaker_label(speaker),
        "words": normalized_words,
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
    job_started_at = time.monotonic()
    inp = job["input"]
    action = inp.get("action", "full") # default to full if not specified
    audio_url = inp.get("audio") or inp.get("audio_url")
    s3_creds = inp.get("s3_creds")
    progress_s3_key = inp.get("progress_s3_key")
    language = inp.get("language", "ru")
    known_speakers = inp.get("known_speakers") or []
    zoom_diarization_mode = (inp.get("diarization_mode") or ZOOM_DIARIZATION_MODE or "robust").strip().lower()
    if zoom_diarization_mode in {"pyannote_ocr", "pyannote+ocr"}:
        zoom_diarization_mode = "robust"
    if zoom_diarization_mode in {"ocr", "ocr_only"}:
        zoom_diarization_mode = "ocr_only"
    # Default to exactly 2 speakers for medical interviews (interviewer + respondent)
    # Setting num_speakers=2 forces pyannote to find the 2 most distinct voice clusters
    min_speakers = inp.get("min_speakers") or 2
    max_speakers = inp.get("max_speakers") or 2
    requested_num_speakers = inp.get("num_speakers")
    num_speakers = requested_num_speakers or 2
    
    if not audio_url:
        return {"error": "Missing audio URL"}

    local_path = None
    audio_path_for_transcription = None
    try:
        download_started_at = time.monotonic()
        _stage_log(f"Downloading input for action={action}...")
        local_path = download_file(audio_url, s3_creds)
        _stage_log(f"Input downloaded in {_elapsed_seconds(download_started_at)}.")
        is_video_input = is_mp4_path(local_path) or is_mp4_path(audio_url)
        if is_video_input:
            extract_started_at = time.monotonic()
            _stage_log("Extracting audio from video for transcription/diarization...")
            audio_path_for_transcription = extract_audio_for_transcription(local_path)
            _stage_log(f"Audio extraction complete in {_elapsed_seconds(extract_started_at)}.")
        else:
            audio_path_for_transcription = local_path
        load_audio_started_at = time.monotonic()
        _stage_log("Loading audio waveform...")
        audio = whisperx.load_audio(audio_path_for_transcription)
        _stage_log(f"Audio waveform loaded in {_elapsed_seconds(load_audio_started_at)}.")
        progress_writer = _build_s3_progress_writer(s3_creds, progress_s3_key)
        
        response = {}

        # 1a. Zoom video diarization.
        if action in ["diarize", "full"] and is_video_input:
            _stage_log(f"Running Zoom video diarization mode={zoom_diarization_mode}...")
            timeline, ocr_summary = build_zoom_ocr_timeline(local_path, known_speakers=known_speakers)
            response["ocr_timeline"] = timeline
            response["ocr_diarization"] = ocr_summary

            if zoom_diarization_mode == "robust":
                pyannote_segments = _run_pyannote_diarization(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=requested_num_speakers,
                    progress_writer=progress_writer,
                )
                pyannote_timeline = _format_diarization_timeline(pyannote_segments, source="pyannote")
                speaker_mapping, mapping_details = map_pyannote_speakers_to_ocr_names(pyannote_timeline, timeline)
                named_timeline = apply_speaker_mapping_to_timeline(pyannote_timeline, speaker_mapping)
                diarize_segments = pd.DataFrame([
                    {"start": item["start"], "end": item["end"], "speaker": item["speaker"]}
                    for item in named_timeline
                ])
                response["timeline"] = named_timeline
                response["pyannote_timeline"] = pyannote_timeline
                response["speaker_name_mapping"] = mapping_details
                response["ocr_diarization"]["mode"] = "robust"
                response["ocr_diarization"]["mapping_confidence_threshold"] = OCR_MAPPING_CONFIDENCE_THRESHOLD
                response["ocr_diarization"]["mapping_min_overlap_seconds"] = OCR_MAPPING_MIN_OVERLAP_SECONDS
            else:
                diarize_segments = pd.DataFrame([
                    {"start": item["start"], "end": item["end"], "speaker": item["speaker"]}
                    for item in timeline
                ])
                response["timeline"] = timeline
                response["ocr_diarization"]["mode"] = "ocr_only"

            if action == "diarize":
                return response

        # 1. Diarization (if requested or full)
        if action in ["diarize", "full"] and not is_video_input:
            diarize_segments = _run_pyannote_diarization(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
                progress_writer=progress_writer,
            )
            timeline = _format_diarization_timeline(diarize_segments, source="pyannote")
            response["timeline"] = timeline
            
            if action == "diarize":
                return response

        # 2. Transcription (if requested or full)
        if action in ["transcribe", "full"]:
            if progress_writer:
                progress_writer({
                    "stage": "transcription",
                    "current": None,
                    "total": None,
                    "unit": "stage",
                    "progress": 99,
                    "message": "Transcribing and aligning...",
                }, force=True)
            model = get_whisper()
            transcribe_started_at = time.monotonic()
            _stage_log("Transcription started...")
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
            _stage_log(f"Transcription complete in {_elapsed_seconds(transcribe_started_at)}.")
            
            # 3. Alignment
            align_started_at = time.monotonic()
            _stage_log("Alignment started...")
            model_a, metadata = get_align(language)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            _stage_log(f"Alignment complete in {_elapsed_seconds(align_started_at)}.")
            
            # 4. Assign Speakers (if we have diarization info)
            if action == "full":
                # We already have diarize_segments from step 1
                result = whisperx.assign_word_speakers(diarize_segments, result, fill_nearest=False)
            elif action == "transcribe" and "timeline" in inp:
                # User provided timeline from previous step
                provided_timeline = pd.DataFrame(inp["timeline"])
                result = whisperx.assign_word_speakers(provided_timeline, result, fill_nearest=False)

            # 5. Split segments at word-level speaker boundaries
            _stage_log("Splitting segments by word-level speaker assignments...")
            split_segments = split_by_word_speakers(result["segments"])
            refined_segments = repair_short_acknowledgement_speakers(split_segments)
            refined_segments = repair_fragmented_turns(refined_segments)
            
            # 6. Format Result for server.py compatibility
            final_segments = []
            for seg in refined_segments:
                text = clean_hallucinations(seg["text"])
                if text:
                    final_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": text,
                        "speaker": seg.get("speaker", "Unknown")
                    })
            
            response["result"] = final_segments

        _stage_log(f"Job complete in {_elapsed_seconds(job_started_at)}.")
        return response

    except Exception as e:
        _stage_log(f"Job failed after {_elapsed_seconds(job_started_at)}: {e}")
        return {"error": str(e)}
    finally:
        if audio_path_for_transcription and audio_path_for_transcription != local_path and os.path.exists(audio_path_for_transcription):
            os.remove(audio_path_for_transcription)
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
