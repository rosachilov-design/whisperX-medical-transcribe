# Plan: Integrating pyannote.audio Speaker Diarization

## What pyannote does

`pyannote.audio` analyzes the audio and produces a timeline of **who speaks when**:

```
Speaker A: 00:00 → 00:45
Speaker B: 00:46 → 01:20
Speaker A: 01:21 → 02:10
Speaker C: 02:11 → 03:05
```

We then match each Whisper text segment to this timeline by timestamp to assign the correct speaker.

---

## Prerequisites

### 1. Install dependencies
```bash
pip install pyannote.audio
```
This pulls in `torch`, `torchaudio`, `speechbrain`, and other deps (most already installed).

### 2. Get a HuggingFace token
- Go to https://huggingface.co/pyannote/speaker-diarization-3.1
- Accept the model terms (one-time, free)
- Generate an access token at https://huggingface.co/settings/tokens
- Store it (we'll use an env var `HF_TOKEN` or a `.env` file)

### 3. First run downloads ~70MB of model weights (cached after that)

---

## Architecture Change

### Current flow (broken):
```
Audio → Whisper (30s chunks) → fake speaker ID per segment → merge → output
```

### New flow:
```
Audio → [parallel]
         ├─ pyannote diarization (full file) → speaker timeline
         └─ Whisper transcription (30s chunks) → text segments
       → match text segments to speaker timeline → merge → output
```

---

## Step-by-Step Implementation

### Step 1: Load pyannote pipeline at startup (alongside Whisper)

In `server.py`, at the top level alongside the Whisper model loading:

```python
from pyannote.audio import Pipeline as DiarizationPipeline

HF_TOKEN = os.getenv("HF_TOKEN", "")

print("Loading diarization pipeline...")
diarization_pipeline = DiarizationPipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
diarization_pipeline.to(torch.device(device))
print("Diarization pipeline loaded.")
```

**Note:** Both Whisper and pyannote share the same GPU. They run sequentially (not in parallel), which is fine — diarization is fast (~30s for 1hr of audio on GPU).

### Step 2: Add a `run_diarization()` function

```python
def run_diarization(file_path: Path) -> list:
    """
    Returns a list of diarization segments:
    [
        {"start": 0.0, "end": 12.5, "speaker": "SPEAKER_00"},
        {"start": 12.8, "end": 45.2, "speaker": "SPEAKER_01"},
        ...
    ]
    """
    # Convert to WAV 16kHz mono first (pyannote works best with this)
    wav_path = CACHE_DIR / f"{file_path.stem}_diarize.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(file_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run diarization
    diarization = diarization_pipeline(str(wav_path))

    # Parse results into a simple list
    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timeline.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Cleanup
    if wav_path.exists():
        os.remove(wav_path)

    return timeline
```

### Step 3: Add a `get_speaker_at_time()` lookup function

```python
def get_speaker_at_time(timeline: list, time_sec: float) -> str:
    """
    Given the diarization timeline and a timestamp (seconds),
    return the speaker label active at that time.
    Falls back to "Unknown" if no speaker is found.
    """
    for seg in timeline:
        if seg["start"] <= time_sec <= seg["end"]:
            return seg["speaker"]
    # If between segments, find the nearest one
    if timeline:
        nearest = min(timeline, key=lambda s: min(abs(s["start"] - time_sec), abs(s["end"] - time_sec)))
        return nearest["speaker"]
    return "Unknown"
```

### Step 4: Update `run_live_transcription()` to use real diarization

Change the flow:

1. **Phase 1 — Diarization** (new): Run pyannote on the full file. Update status to `"diarizing"` so the frontend can show progress like "Identifying speakers..."
2. **Phase 2 — Transcription** (existing): Run Whisper chunk-by-chunk as before, BUT instead of the fake `speaker_id = (len(all_segments) // 5) % 3 + 1`, do:

```python
# Replace the mock speaker heuristic with:
speaker_label = get_speaker_at_time(timeline, seg_start)
```

3. **Rename speakers** to friendly names: pyannote outputs `SPEAKER_00`, `SPEAKER_01`, etc. We can map these to `Speaker 1`, `Speaker 2`, etc:

```python
speaker_map = {}  # built as we encounter new pyannote labels
speaker_counter = 0

# Inside the loop:
raw_label = get_speaker_at_time(timeline, seg_start)
if raw_label not in speaker_map:
    speaker_counter += 1
    speaker_map[raw_label] = f"Speaker {speaker_counter}"
speaker = speaker_map[raw_label]
```

### Step 5: Update frontend status display

The frontend currently shows "Transcribing Live..." as the status. We should add awareness of the new diarization phase:

- `"diarizing"` → Show *"Identifying speakers..."* in the status bar
- `"transcribing"` → Show *"Transcribing..."* as before

Small change in `app.js` `updateUI()`:
```javascript
if (data.status === 'diarizing') {
    statusText.textContent = 'Identifying speakers...';
} else if (data.status === 'transcribing') {
    statusText.textContent = 'Transcribing Live...';
}
```

### Step 6: Handle the HF token in the app

Options (pick one):
- **Environment variable**: `set HF_TOKEN=hf_xxxxx` before running the server
- **`.env` file**: Add `python-dotenv`, create `.env` with `HF_TOKEN=hf_xxxxx`
- **Config in `run_app.bat`**: Set it there before launching python

Recommended: `.env` file for simplicity.

---

## Updated `run_live_transcription()` (pseudo-code)

```python
def run_live_transcription(file_path: Path, task_id: str):
    try:
        # === Phase 1: Diarization ===
        transcriptions[task_id]["status"] = "diarizing"
        transcriptions[task_id]["progress"] = 0
        timeline = run_diarization(file_path)
        transcriptions[task_id]["progress"] = 10  # diarization is ~10% of work

        # Build speaker map
        speaker_map = {}
        speaker_counter = 0

        # === Phase 2: Transcription ===
        transcriptions[task_id]["status"] = "transcribing"
        duration = get_duration(file_path)
        chunk_size = 30
        num_chunks = math.ceil(duration / chunk_size)
        all_segments = []

        for i in range(num_chunks):
            start_time = i * chunk_size
            # ... extract chunk with ffmpeg (same as before) ...
            # ... transcribe with Whisper (same as before) ...

            for segment in result["segments"]:
                seg_start = segment["start"] + start_time
                text = ...  # same cleanup as before

                # NEW: real speaker assignment
                raw_label = get_speaker_at_time(timeline, seg_start)
                if raw_label not in speaker_map:
                    speaker_counter += 1
                    speaker_map[raw_label] = f"Speaker {speaker_counter}"
                speaker = speaker_map[raw_label]

                new_seg = {
                    "start": seg_start,
                    "timestamp": format_timestamp(seg_start),
                    "text": text,
                    "speaker": speaker
                }
                all_segments.append(new_seg)

            # Merge and update live result
            transcriptions[task_id]["result"] = merge_speaker_segments(all_segments)
            # Progress: 10% (diarization) + 90% * (chunks done / total)
            transcriptions[task_id]["progress"] = 10 + int(((i + 1) / num_chunks) * 90)

        # ... save MD + DOCX (same as before) ...
```

---

## Performance Estimate

| Step | Time (1hr audio, GPU) | Time (1hr audio, CPU) |
|---|---|---|
| Diarization | ~30 seconds | ~3-5 minutes |
| Whisper transcription | ~3-5 minutes | ~20-40 minutes |
| **Total** | **~4-6 minutes** | **~25-45 minutes** |

Diarization adds minimal overhead on GPU.

---

## Files to change

| File | Changes |
|---|---|
| `server.py` | Add pyannote import + pipeline load. Add `run_diarization()` and `get_speaker_at_time()`. Replace mock speaker heuristic. Add `"diarizing"` status phase. |
| `app.js` | Handle `"diarizing"` status in `updateUI()`. |
| `.env` (new) | Store `HF_TOKEN=hf_xxxxx`. |
| `run_app.bat` | Optionally load `.env` or set `HF_TOKEN`. |
| `requirements.txt` | Add `pyannote.audio`. |

---

## Bulk Speaker Renaming (bonus, from future-plans.md)

Once pyannote assigns real speaker clusters, the existing "speaker pill" UI already lets users click to rename individual segments. But we should also add **bulk rename** — when you rename "Speaker 1" → "Анна Григорьевна", it updates ALL segments with that speaker, not just one. This is a small additional change to `update_speaker` endpoint.
