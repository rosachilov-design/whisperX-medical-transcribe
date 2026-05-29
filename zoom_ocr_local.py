import re
from collections import Counter
from pathlib import Path


UNKNOWN_SPEAKER = "Unknown"
OCR_SAMPLE_FPS = 2
OCR_MIN_CONFIDENCE = 0.10
OCR_FUZZY_THRESHOLD = 75
OCR_AUTO_CONFIDENCE = 0.10
OCR_AUTO_CLUSTER_THRESHOLD = 85
OCR_AUTO_MIN_CLUSTER_COUNT = 3
OCR_FILL_GAPS_SECONDS = 4.0
OCR_MIN_SAME_NAME_FRAMES = 2
OCR_MIN_SEGMENT_SECONDS = 0.75
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


_OCR_MODEL = None
_HUNYUAN_MODEL = None
_HUNYUAN_PROCESSOR = None


def get_ocr():
    global _OCR_MODEL
    if _OCR_MODEL is None:
        from paddle_ocr_factory import create_zoom_paddle_ocr

        _OCR_MODEL = create_zoom_paddle_ocr()
    return _OCR_MODEL


def get_hunyuan_ocr():
    global _HUNYUAN_MODEL, _HUNYUAN_PROCESSOR
    if _HUNYUAN_MODEL is None or _HUNYUAN_PROCESSOR is None:
        try:
            import torch
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "HunyuanOCR local mode requires torch and a HunyuanOCR-capable transformers build. "
                "Install the Transformers commit from the tencent/HunyuanOCR model card."
            ) from exc

        model_name = "tencent/HunyuanOCR"
        _HUNYUAN_PROCESSOR = AutoProcessor.from_pretrained(model_name, use_fast=False)
        kwargs = {
            "attn_implementation": "eager",
            "device_map": "auto",
        }
        if torch.cuda.is_available():
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["dtype"] = torch.float32

        _HUNYUAN_MODEL = HunYuanVLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        _HUNYUAN_MODEL.eval()
    return _HUNYUAN_PROCESSOR, _HUNYUAN_MODEL


def _clean_ocr_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;|/\\_-—–[](){}")


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
    import cv2

    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(w * region["x1"])))
    y1 = max(0, min(h, int(h * region["y1"])))
    x2 = max(0, min(w, int(w * region["x2"])))
    y2 = max(0, min(h, int(h * region["y2"])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
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


def _read_hunyuan_candidate(hunyuan, frame, region):
    crop = _crop_region(frame, region)
    if crop.size == 0:
        return {
            "region": region["name"],
            "raw_text": None,
            "confidence": 0.0,
            "priority": region.get("priority", 0),
        }

    try:
        import cv2
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("HunyuanOCR local mode requires cv2, torch, and pillow.") from exc

    processor, model = hunyuan
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    prompt = (
        "Read only the Zoom participant name visible in this cropped name label. "
        "Return exactly one JSON object: {\"name\": \"...\"}. "
        "If no participant name is visible, return {\"name\": null}. "
        "Do not infer names from faces, slides, or context."
    )
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    inputs = processor(text=texts, images=image, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    raw_text = _extract_hunyuan_name(output)
    return {
        "region": region["name"],
        "raw_text": raw_text or None,
        "confidence": 0.99 if raw_text else 0.0,
        "priority": region.get("priority", 0),
    }


def _extract_hunyuan_name(output):
    import json

    text = str(output or "").strip()
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            name = data.get("name")
            if name is None:
                return None
            return _clean_ocr_text(name)
        except Exception:
            pass

    text = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^name\s*[:=]\s*", "", text, flags=re.IGNORECASE).strip()
    if text.lower() in {"null", "none", "no name", "unknown"}:
        return None
    return _clean_ocr_text(text)


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
        canonical, _ = counts.most_common(1)[0]
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
    return {
        "speaker": name,
        "score": round(float(candidate["confidence"]) * (float(fuzzy_score) / 100.0), 4),
        "fuzzy_score": round(float(fuzzy_score), 2),
    }


def _match_discovered_speaker(candidate, discovered_speakers):
    if not _is_name_like_text(candidate.get("raw_text"), OCR_AUTO_CONFIDENCE, candidate.get("confidence", 0.0)):
        return None
    try:
        from rapidfuzz import process, fuzz
    except Exception as exc:
        raise RuntimeError("rapidfuzz is required for OCR speaker matching.") from exc

    names = [speaker["speaker"] for speaker in discovered_speakers]
    match = process.extractOne(candidate["raw_text"], names, scorer=fuzz.WRatio)
    if not match:
        return None
    name, fuzzy_score, _ = match
    if fuzzy_score < OCR_AUTO_CLUSTER_THRESHOLD:
        return None
    return {
        "speaker": name,
        "score": round(float(candidate["confidence"]) * (float(fuzzy_score) / 100.0), 4),
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

        if prev_speaker != UNKNOWN_SPEAKER and prev_speaker == next_speaker and duration <= OCR_FILL_GAPS_SECONDS:
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


def _summarize_low_confidence_intervals(timeline):
    return [
        {"start": item["start"], "end": item["end"], "speaker": item["speaker"]}
        for item in timeline
        if item["speaker"] == UNKNOWN_SPEAKER
    ][:25]


def build_zoom_ocr_timeline(
    video_path,
    known_speakers=None,
    sample_fps=OCR_SAMPLE_FPS,
    regions=None,
    on_progress=None,
    ocr_engine="paddle",
):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python-headless is required for local Zoom OCR.") from exc

    video_path = Path(video_path)
    regions = regions or OCR_REGIONS
    known_speakers = [_clean_ocr_text(name) for name in (known_speakers or []) if _clean_ocr_text(name)]
    ocr_engine = (ocr_engine or "paddle").lower()
    if ocr_engine == "hunyuan":
        ocr = get_hunyuan_ocr()
        read_candidate = _read_hunyuan_candidate
    elif ocr_engine == "paddle":
        ocr = get_ocr()
        read_candidate = _read_ocr_candidate
    else:
        raise ValueError(f"Unsupported local OCR engine: {ocr_engine}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / native_fps) if native_fps else 0.0
    step = 1.0 / float(sample_fps)
    total_frames = max(1, int(duration * sample_fps) + 1) if duration > 0 else 1

    raw_samples = []
    discovery_candidates = []
    t = 0.0
    frame_index = 0
    while t <= duration + 0.001:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if on_progress is not None:
            on_progress(frame_index, max(total_frames, frame_index))

        candidates = [read_candidate(ocr, frame, region) for region in regions]
        raw_samples.append({"t": round(t, 3), "candidates": candidates})

        if not known_speakers:
            for candidate in candidates:
                if _is_name_like_text(candidate.get("raw_text"), OCR_AUTO_CONFIDENCE, candidate.get("confidence", 0.0)):
                    discovery_candidates.append(candidate["raw_text"])
        t += step

    cap.release()
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

    return timeline, {
        "source": "zoom_ocr_local",
        "ocr_engine": ocr_engine,
        "sample_fps": sample_fps,
        "regions": regions,
        "known_speakers": known_speakers,
        "discovered_speakers": discovered_speakers,
        "unknown_rate": round(unknown_duration / total_duration, 4) if total_duration else 1.0,
        "low_confidence_intervals": _summarize_low_confidence_intervals(timeline),
    }
