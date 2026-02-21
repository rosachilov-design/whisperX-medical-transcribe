# Implementation Plan: Speaker Line Merging + DOCX Export

## Problem 1: Segments break mid-speaker

**Root cause:** Whisper returns one segment per short phrase/sentence. The server stores each as a separate item in `all_segments` and the frontend renders each as its own block with speaker header + timestamp. So you get:

```
Speaker 1  10:13
oh
Speaker 1  10:14
hey
Speaker 1  10:15
there
```

**Goal:** Merge consecutive segments from the same speaker into a single paragraph:

```
Speaker 1  10:13
oh hey there
```

---

## Plan — Step by Step

### Step 1: Add a post-processing merge function in `server.py`

After all chunks are transcribed (the `for i in range(num_chunks)` loop finishes), add a **`merge_speaker_segments()`** function that:

1. Iterates through `all_segments` (the raw per-phrase list).
2. Groups consecutive segments that have the **same `speaker`** value.
3. For each group, produces a **single merged segment** with:
   - `start` = the `start` of the **first** segment in the group
   - `timestamp` = the `timestamp` of the **first** segment in the group
   - `text` = all texts **joined with a space** (e.g., `"oh" + "hey" + "there"` → `"oh hey there"`)
   - `speaker` = the shared speaker name
4. Returns the merged list.

This merged list replaces `all_segments` for everything downstream (MD/DOCX generation and the result sent to the frontend).

### Step 2: Update the live-streaming result to also reflect merging

Currently `transcriptions[task_id]["result"]` is appended to on every segment. The frontend polls this and renders incrementally. We need to change this so:

- During transcription, keep appending raw segments to `all_segments` as before (used for the Whisper `initial_prompt` context).
- After each chunk, **re-merge** and replace `transcriptions[task_id]["result"]` with the merged list. The frontend already replaces all content when `lastSegmentCount` changes.
- **Frontend change:** Instead of only appending new segments, when the *existing* merged segments change (last segment text grows), re-render the last segment's text. This requires a small tweak to `updateUI()` in `app.js`.

### Step 3: Update `.md` generation to use merged segments

The `regenerate_md()` and the inline MD generation at line 110–116 should iterate over the **merged** segments instead of raw ones. Each merged segment becomes one block:

```md
**[10:13] Speaker 1:** oh hey there
```

### Step 4: Add `.docx` export

- **Add `python-docx` dependency.**
- Create a `generate_docx()` function in `server.py` that:
  1. Creates a `Document()`.
  2. Adds a title paragraph: `"Transcription: {filename}"`.
  3. For each merged segment, adds a paragraph with:
     - Bold run: `[timestamp] Speaker Name:`
     - Normal run: ` the merged text`
  4. Saves to `file_path.with_suffix(".docx")`.
- Call `generate_docx()` alongside the `.md` generation at the end of `run_live_transcription` and in `regenerate_md`.
- Store `docx_path` in `transcriptions[task_id]` alongside `md_path`.

### Step 5: Add a `/download/{filename}` route update for `.docx`

The existing `/download/{filename}` endpoint already serves any file by name. Just ensure the media type logic handles `.docx` (use `application/vnd.openxmlformats-officedocument.wordprocessingml.document`).

### Step 6: Update the frontend for two download buttons

In `index.html`, change the footer to show two buttons:
- **Download .md** (existing)
- **Download .docx** (new)

In `app.js`, update `updateUI` to populate both filenames from the server response (`md_path` and `docx_path`), and wire the new button's `onclick` to download the `.docx` file.

---

## Summary of files to change

| File | Changes |
|---|---|
| `server.py` | Add `merge_speaker_segments()`, call it after transcription and after speaker rename. Add `generate_docx()`. Update MD generation to use merged data. Store `docx_path`. |
| `app.js` | Update `updateUI()` to handle segment merging (re-render last segment when text grows). Add `.docx` download button handler. |
| `index.html` | Add a second download button for `.docx`. |
| `requirements.txt` / install | Add `python-docx` dependency. |
