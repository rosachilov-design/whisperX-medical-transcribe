"""
Transcriber Pro — Local Review Dashboard
Lightweight local server for reviewing cloud-transcribed results.
No GPU needed. Loads .json state files and pairs them with local audio.

Also supports S3 upload for sending files to RunPod.
"""

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import os
import sys
import io
import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from pathlib import PurePosixPath
import requests as http_requests
import paramiko
from scp import SCPClient
import tarfile
import shutil

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv
from preprocess_for_transcription import build_filter_chain

# Load .env credentials
load_dotenv()

# Fix UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directories ───
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── S3 Config ───
S3_BUCKET = "ez2d4o9xmt"
S3_ENDPOINT = "https://s3api-us-wa-1.runpod.io"
S3_REGION = "us-wa-1"
S3_MULTIPART_THRESHOLD = int(os.getenv("S3_MULTIPART_THRESHOLD", str(64 * 1024 * 1024)))
S3_MULTIPART_CHUNKSIZE = int(os.getenv("S3_MULTIPART_CHUNKSIZE", str(64 * 1024 * 1024)))
S3_MAX_CONCURRENCY = int(os.getenv("S3_MAX_CONCURRENCY", "4"))
RUNPOD_ESTIMATED_SECONDS = int(os.getenv("RUNPOD_ESTIMATED_SECONDS", str(30 * 60)))
SUPPORTED_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav"}

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    region_name=S3_REGION,
    aws_access_key_id=os.getenv("RUNPOD_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("RUNPOD_SECRET_KEY"),
    config=Config(signature_version="s3v4"),
)


def build_s3_transfer_config():
    return TransferConfig(
        multipart_threshold=S3_MULTIPART_THRESHOLD,
        multipart_chunksize=S3_MULTIPART_CHUNKSIZE,
        max_concurrency=S3_MAX_CONCURRENCY,
        use_threads=True,
    )

# ─── RunPod API & SSH Config ───
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "") # New Serverless Endpoint
RUNPOD_GQL = "https://api.runpod.io/graphql"
HF_TOKEN = os.getenv("HF_TOKEN", "") # HuggingFace Token for Diarization

# SSH Config (User-provided via UI)
pod_config = {
    "ip": os.getenv("POD_IP", ""),
    "ssh_port": int(os.getenv("POD_SSH_PORT", "22")),
    "key_path": os.getenv("POD_KEY_PATH", "runpod")
}

# ─── State ───
transcriptions = {}

def download_results_from_s3():
    """Check S3 for finished JSON results and pull them to local uploads."""
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        if 'Contents' not in response:
            return

        result_exts = {".json"}
        found_new = False
        for obj in response['Contents']:
            s3_key = obj['Key']
            # Strip the 'transcriber/uploads/' prefix if present
            base_name = s3_key.split('/')[-1] if '/' in s3_key else s3_key
            ext = Path(base_name).suffix.lower()

            if ext in result_exts:
                local_path = UPLOAD_DIR / base_name
                if not local_path.exists():
                    if not found_new:
                        print("☁️ New results found on cloud!")
                        found_new = True
                    print(f"  📥 Downloading: {base_name}")
                    s3.download_file(S3_BUCKET, s3_key, str(local_path))

                    if ext == ".json":
                        try:
                            with open(local_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                task_id = data.get("filename")
                                if task_id:
                                    transcriptions[task_id] = data
                        except:
                            pass
    except Exception as e:
        print(f"⚠️ Cloud sync check failed: {e}")

def load_existing_tasks():
    """Load previously completed transcriptions from JSON files on disk."""
    print("📂 Scanning local uploads...")
    for json_file in UPLOAD_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                task_id = data.get("filename")
                if task_id:
                    transcriptions[task_id] = data
                    print(f"  ✅ Loaded: {task_id}")
        except Exception as e:
            print(f"  ❌ Failed to load {json_file.name}: {e}")

# Initial load from disk
load_existing_tasks()

# Start a background thread to check cloud every 30 seconds
def cloud_watchdog():
    # Initial sync on first run (non-blocking)
    download_results_from_s3()
    while True:
        time.sleep(30)
        download_results_from_s3()

threading.Thread(target=cloud_watchdog, daemon=True).start()


# ─── Helpers ───

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"


def format_transcription_segments(result_segments):
    formatted_segments = []
    for seg in result_segments or []:
        start = seg["start"]
        formatted_segments.append({
            "start": start,
            "end": seg.get("end", start + 2),
            "timestamp": format_timestamp(start),
            "speaker": seg.get("speaker", "Unknown"),
            "text": seg["text"]
        })
    return formatted_segments


def is_video_filename(filename):
    return Path(filename or "").suffix.lower() == ".mp4"


def is_supported_audio_filename(filename):
    return Path(filename or "").suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def normalize_known_speakers(value):
    if not value:
        return []
    if isinstance(value, str):
        raw_items = value.split(";")
    else:
        raw_items = value
    normalized = []
    seen = set()
    for item in raw_items:
        name = re.sub(r"\s+", " ", str(item or "")).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def get_json_result_name(task):
    filename = task.get("filename")
    if not filename:
        return None
    return Path(filename).with_suffix(".json").name


def get_json_name_for_filename(filename: str) -> str:
    return Path(filename).with_suffix(".json").name


def serialize_task(task):
    serialized = dict(task)
    json_name = get_json_result_name(task)
    if json_name:
        serialized["json_path"] = json_name
    return serialized


def find_remote_json_keys(filename: str, task: dict | None = None):
    """Find likely JSON result objects for a task in the RunPod S3 bucket."""
    json_names = {get_json_name_for_filename(filename)}
    if task and task.get("s3_key"):
        json_names.add(Path(task["s3_key"]).with_suffix(".json").name)

    matches = []
    continuation_token = None
    while True:
        params = {"Bucket": S3_BUCKET}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**params)
        for entry in response.get("Contents", []):
            key = entry.get("Key", "")
            if key and Path(key).name in json_names:
                matches.append(key)

        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

    return matches


def delete_s3_keys(keys):
    """Delete S3 objects one-by-one for RunPod S3 compatibility."""
    deleted = 0
    for key in keys:
        if not key:
            continue
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        deleted += 1
    return deleted


def delete_local_upload_files():
    """Delete local server-side cached uploads/results."""
    deleted = 0
    for item in UPLOAD_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        deleted += 1
    transcriptions.clear()
    return deleted


def clean_hallucinations(text: str) -> str:
    """Remove common Russian Whisper hallucinations."""
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


def persist_task_json(task_id):
    """Persist the current transcription state as JSON."""
    task = transcriptions[task_id]
    state_file = UPLOAD_DIR / get_json_result_name(task)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)


def improve_audio_for_transcription(input_path: Path, output_path: Path):
    """Create a cleaned M4A optimized for speech transcription."""
    filter_chain = build_filter_chain("mild", denoise=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        filter_chain,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown ffmpeg error"
        raise RuntimeError(error_text)


def run_local_ocr_task(task_id: str, known_speakers=None, ocr_engine="paddle"):
    task = transcriptions.get(task_id)
    if not task:
        return

    try:
        filename = task.get("filename", task_id)
        if not is_video_filename(filename):
            raise ValueError("Local OCR test is only available for .mp4 files.")

        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {filename}")

        from paddle_ocr_factory import get_ocr_device

        task["status"] = "local_ocr_processing"
        task["progress"] = 10
        task["ocr_frame"] = 0
        task["ocr_frames_total"] = 0
        task["ocr_device"] = get_ocr_device()
        persist_task_json(task_id)

        from zoom_ocr_local import build_zoom_ocr_timeline

        last_persist_at = [0.0]
        last_terminal_line = [None]
        device_label = f"{task['ocr_device']} / {ocr_engine}"

        print(f"Local OCR starting: {task_id} on {device_label}", flush=True)

        def on_ocr_progress(current, total):
            task["ocr_frame"] = current
            task["ocr_frames_total"] = total
            progress_pct = min(95, 10 + int(85 * current / max(total, 1)))
            task["progress"] = progress_pct
            now = time.time()
            if now - last_persist_at[0] >= 0.4 or current >= total:
                last_persist_at[0] = now
                persist_task_json(task_id)

            line = (
                f"Local OCR [{task_id}] frame {current}/{total} "
                f"[{device_label}] {progress_pct}%"
            )
            if line != last_terminal_line[0]:
                last_terminal_line[0] = line
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()

        timeline, ocr_summary = build_zoom_ocr_timeline(
            file_path,
            known_speakers=normalize_known_speakers(known_speakers),
            on_progress=on_ocr_progress,
            ocr_engine=ocr_engine,
        )

        task["timeline"] = timeline
        task["ocr_diarization"] = ocr_summary
        task["status"] = "diarization_complete"
        task["progress"] = 100
        task.pop("ocr_frame", None)
        task.pop("ocr_frames_total", None)
        persist_task_json(task_id)
        if last_terminal_line[0] is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        print(f"Local OCR diarization complete: {task_id}", flush=True)
    except Exception as e:
        task["status"] = "error"
        task["error"] = f"Local OCR failed: {e}"
        task["progress"] = 0
        task.pop("ocr_frame", None)
        task.pop("ocr_frames_total", None)
        persist_task_json(task_id)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"Local OCR failed for {task_id}: {e}", flush=True)


def list_s3_bucket_files():
    """Return all actual objects currently present in the configured S3 bucket."""
    files = []
    continuation_token = None

    while True:
        params = {"Bucket": S3_BUCKET}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**params)
        for entry in response.get("Contents", []):
            key = entry.get("Key", "")
            if not key or key.endswith("/"):
                continue
            files.append({
                "key": key,
                "name": Path(key).name,
                "size": entry.get("Size", 0),
                "last_modified": entry.get("LastModified").isoformat() if entry.get("LastModified") else None,
            })

        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

    files.sort(key=lambda item: item["last_modified"] or "", reverse=True)
    return files


# ─── S3 Upload (Background Thread) ───

def upload_to_s3(file_path: Path, task_id: str):
    """Upload audio file to RunPod S3 bucket in background."""
    try:
        transcriptions[task_id]["status"] = "uploading"
        transcriptions[task_id]["progress"] = 5

        file_size = file_path.stat().st_size
        uploaded = 0

        def progress_callback(bytes_transferred):
            nonlocal uploaded
            uploaded += bytes_transferred
            pct = min(int((uploaded / file_size) * 90), 90)
            transcriptions[task_id]["progress"] = pct

        # Use a safe ASCII key for S3 to prevent URL encoding mismatch with Signature v4
        safe_key = f"{uuid.uuid4().hex}_{int(time.time())}{file_path.suffix}"

        s3.upload_file(
            str(file_path),
            S3_BUCKET,
            f"transcriber/uploads/{safe_key}",
            Callback=progress_callback,
            Config=build_s3_transfer_config(),
        )

        transcriptions[task_id]["status"] = "uploaded"
        transcriptions[task_id]["progress"] = 100
        transcriptions[task_id]["s3_key"] = safe_key
        print(f"☁️ Uploaded {file_path.name} to S3 as {safe_key}")

    except Exception as e:
        transcriptions[task_id]["status"] = "error"
        transcriptions[task_id]["error"] = f"S3 upload failed: {e}"
        print(f"❌ S3 upload failed: {e}")


def require_uploaded_s3_key(task: dict):
    """Return the full S3 object key for a task, or a 409 response if not ready."""
    safe_key = task.get("s3_key")
    if not safe_key:
        status = task.get("status") or "unknown"
        return None, JSONResponse(
            status_code=409,
            content={
                "error": (
                    f"File has not finished uploading to S3 yet (status: {status}). "
                    "Wait until the upload is complete, then start cloud processing again."
                )
            },
        )

    if safe_key.startswith("transcriber/uploads/"):
        return safe_key, None
    return f"transcriber/uploads/{safe_key}", None


def get_runpod_output_error(output):
    """Return a handler-level error from RunPod output, if one was returned."""
    if isinstance(output, dict) and output.get("error"):
        return str(output["error"])
    return None


def get_progress_s3_key(task: dict):
    safe_key = task.get("s3_key")
    if not safe_key:
        return None
    upload_name = PurePosixPath(safe_key).name
    progress_name = PurePosixPath(upload_name).with_suffix(".progress.json").name
    return f"transcriber/progress/{progress_name}"


def refresh_runpod_progress(task: dict):
    progress_key = task.get("progress_s3_key")
    if not progress_key:
        return False

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=progress_key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return False

    task["runpod_stage"] = payload.get("stage")
    task["runpod_progress_message"] = payload.get("message")
    task["pyannote_current"] = payload.get("current")
    task["pyannote_total"] = payload.get("total")
    task["pyannote_unit"] = payload.get("unit")
    if payload.get("progress") is not None:
        task["progress"] = max(int(task.get("progress") or 0), int(payload["progress"]))
    return True


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def mark_runpod_queued(task: dict, label: str):
    task["runpod_submitted_at"] = task.get("runpod_submitted_at") or time.time()
    elapsed = time.time() - task["runpod_submitted_at"]
    task["progress"] = max(int(task.get("progress") or 0), 20)
    task["runpod_stage"] = "queue"
    task["runpod_progress_message"] = f"{label}: queued for {format_elapsed(elapsed)}"
    task["runpod_elapsed_seconds"] = int(elapsed)


def mark_runpod_processing(task: dict, label: str, floor: int = 25, ceiling: int = 95):
    task["runpod_started_at"] = task.get("runpod_started_at") or time.time()
    elapsed = time.time() - task["runpod_started_at"]
    estimated = max(RUNPOD_ESTIMATED_SECONDS, 1)
    estimated_progress = floor + int((ceiling - floor) * min(elapsed / estimated, 1.0))
    task["progress"] = max(int(task.get("progress") or 0), estimated_progress)
    task["runpod_stage"] = "processing_estimate"
    task["runpod_progress_message"] = (
        f"{label}: {format_elapsed(elapsed)} elapsed "
        f"(estimated, waiting for RunPod result)"
    )
    task["runpod_elapsed_seconds"] = int(elapsed)


@app.post("/diarize-cloud/{task_id}")
async def diarize_cloud(task_id: str, min_speakers: int = 1, max_speakers: int = 10, num_speakers: int = None):
    if task_id not in transcriptions:
        return {"error": "Task not found"}
    
    task = transcriptions[task_id]
    if not RUNPOD_ENDPOINT_ID:
        return {"error": "RUNPOD_ENDPOINT_ID not set in .env"}

    def poll_job(job_id, task_id):
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
        
        while True:
            try:
                resp = http_requests.get(status_url, headers=headers)
                data = resp.json()
                status = data.get("status")
                
                if status == "COMPLETED":
                    output = data.get("output") or {}
                    output_error = get_runpod_output_error(output)
                    if output_error:
                        transcriptions[task_id]["status"] = "error"
                        transcriptions[task_id]["error"] = output_error
                        print(f"❌ Serverless Job Completed With Error ({job_id}): {output_error}")
                        break
                    timeline = output.get("timeline", [])
                    
                    transcriptions[task_id]["timeline"] = timeline
                    transcriptions[task_id]["status"] = "diarization_complete"
                    transcriptions[task_id]["progress"] = 100
                    
                    # Cache the diarization back to JSON
                    persist_task_json(task_id)
                        
                    print(f"✅ Serverless Diarization Done: {task_id}")
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    error_msg = data.get("error", "Job failed")
                    transcriptions[task_id]["status"] = "error"
                    transcriptions[task_id]["error"] = error_msg
                    print(f"❌ Serverless Job Failed ({job_id}): {error_msg}")
                    break
                
                if status == "IN_PROGRESS":
                    transcriptions[task_id]["status"] = "diarizing"
                    if not refresh_runpod_progress(transcriptions[task_id]):
                        mark_runpod_processing(transcriptions[task_id], "Diarization")
                elif status == "IN_QUEUE":
                    transcriptions[task_id]["status"] = "diarizing"
                    mark_runpod_queued(transcriptions[task_id], "Diarization")

                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Error polling job {job_id}: {e}")
                time.sleep(10)

    try:
        s3_key, error_response = require_uploaded_s3_key(task)
        if error_response:
            return error_response
        task["progress_s3_key"] = get_progress_s3_key(task)
        url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "action": "diarize",
                "audio": s3_key,
                "progress_s3_key": task["progress_s3_key"],
                "s3_creds": {
                    "endpoint": S3_ENDPOINT,
                    "region": S3_REGION,
                    "access_key": os.getenv("RUNPOD_ACCESS_KEY"),
                    "secret_key": os.getenv("RUNPOD_SECRET_KEY"),
                    "bucket": S3_BUCKET
                },
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "num_speakers": num_speakers,
                "hf_token": HF_TOKEN
            }
        }
        
        resp = http_requests.post(url, headers=headers, json=payload)
        resp_data = resp.json()
        job_id = resp_data.get("id")
        
        if job_id:
            task["status"] = "diarizing"
            task["progress"] = 10
            task["job_id"] = job_id
            task["runpod_submitted_at"] = time.time()
            task.pop("runpod_started_at", None)
            
            threading.Thread(target=poll_job, args=(job_id, task_id), daemon=True).start()
            print(f"🚀 Serverless Diarization Job Started: {job_id} for {task_id}")
            return {"status": "started", "job_id": job_id}
        else:
            return {"status": "error", "error": f"Failed to start job: {resp_data}"}

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        return {"status": "error", "error": str(e)}

@app.post("/transcribe-cloud/{task_id}")
async def transcribe_cloud(task_id: str):
    """Trigger RunPod Serverless transcription using background polling."""
    if task_id not in transcriptions:
        return {"error": "Task not found"}
    
    task = transcriptions[task_id]
    if not RUNPOD_ENDPOINT_ID:
        return {"error": "RUNPOD_ENDPOINT_ID not set in .env"}

    def poll_job(job_id, task_id):
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
        
        while True:
            try:
                resp = http_requests.get(status_url, headers=headers)
                data = resp.json()
                status = data.get("status")
                
                if status == "COMPLETED":
                    output = data.get("output") or {}
                    output_error = get_runpod_output_error(output)
                    if output_error:
                        transcriptions[task_id]["status"] = "error"
                        transcriptions[task_id]["error"] = output_error
                        print(f"❌ Serverless Job Completed With Error ({job_id}): {output_error}")
                        break
                    transcriptions[task_id]["result"] = format_transcription_segments(output.get("result", []))
                    transcriptions[task_id]["status"] = "completed"
                    transcriptions[task_id]["progress"] = 100
                    persist_task_json(task_id)
                    print(f"✅ Serverless Transcription Done: {task_id}")
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    error_msg = data.get("error", "Job failed")
                    transcriptions[task_id]["status"] = "error"
                    transcriptions[task_id]["error"] = error_msg
                    print(f"❌ Serverless Job Failed ({job_id}): {error_msg}")
                    break
                
                if status == "IN_PROGRESS":
                    transcriptions[task_id]["status"] = "transcribing"
                    if not refresh_runpod_progress(transcriptions[task_id]):
                        mark_runpod_processing(transcriptions[task_id], "Transcription")
                elif status == "IN_QUEUE":
                    transcriptions[task_id]["status"] = "transcribing"
                    mark_runpod_queued(transcriptions[task_id], "Transcription")

                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Error polling job {job_id}: {e}")
                time.sleep(10)

    try:
        s3_key, error_response = require_uploaded_s3_key(task)
        if error_response:
            return error_response
        task["progress_s3_key"] = get_progress_s3_key(task)
        
        url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        timeline = task.get("timeline", [])
        
        payload = {
            "input": {
                "action": "transcribe",
                "audio": s3_key,
                "progress_s3_key": task["progress_s3_key"],
                "s3_creds": {
                    "endpoint": S3_ENDPOINT,
                    "region": S3_REGION,
                    "access_key": os.getenv("RUNPOD_ACCESS_KEY"),
                    "secret_key": os.getenv("RUNPOD_SECRET_KEY"),
                    "bucket": S3_BUCKET
                },
                "timeline": timeline,
                "hf_token": HF_TOKEN
            }
        }
        
        resp = http_requests.post(url, headers=headers, json=payload)
        resp_data = resp.json()
        job_id = resp_data.get("id")
        
        if job_id:
            task["status"] = "transcribing"
            task["progress"] = 10
            task["job_id"] = job_id
            task["runpod_submitted_at"] = time.time()
            task.pop("runpod_started_at", None)
            
            threading.Thread(target=poll_job, args=(job_id, task_id), daemon=True).start()
            print(f"🚀 Serverless Transcription Job Started: {job_id} for {task_id}")
            return {"status": "started", "job_id": job_id}
        else:
            return {"status": "error", "error": f"Failed to start job: {resp_data}"}

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        return {"status": "error", "error": str(e)}



# ═══════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════

@app.post("/process-cloud/{task_id}")
async def process_cloud(
    task_id: str,
    min_speakers: int = 1,
    max_speakers: int = 10,
    num_speakers: int = None,
    request_body: dict | None = Body(default=None),
):
    """Trigger the full serverless pipeline in a single RunPod job."""
    if task_id not in transcriptions:
        return {"error": "Task not found"}

    task = transcriptions[task_id]
    if not RUNPOD_ENDPOINT_ID:
        return {"error": "RUNPOD_ENDPOINT_ID not set in .env"}

    def poll_job(job_id, task_id):
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"

        while True:
            try:
                resp = http_requests.get(status_url, headers=headers)
                data = resp.json()
                status = data.get("status")

                if status == "COMPLETED":
                    output = data.get("output") or {}
                    output_error = get_runpod_output_error(output)
                    if output_error:
                        transcriptions[task_id]["status"] = "error"
                        transcriptions[task_id]["error"] = output_error
                        print(f"Serverless job completed with error ({job_id}): {output_error}")
                        break
                    transcriptions[task_id]["timeline"] = output.get("timeline", [])
                    if output.get("ocr_diarization"):
                        transcriptions[task_id]["ocr_diarization"] = output.get("ocr_diarization")
                    for layer_key in ("ocr_timeline", "pyannote_timeline", "speaker_name_mapping"):
                        if output.get(layer_key) is not None:
                            transcriptions[task_id][layer_key] = output.get(layer_key)
                    transcriptions[task_id]["result"] = format_transcription_segments(output.get("result", []))
                    transcriptions[task_id]["status"] = "completed"
                    transcriptions[task_id]["progress"] = 100
                    persist_task_json(task_id)
                    print("Serverless full pipeline done: " + task_id)
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    error_msg = data.get("error", "Job failed")
                    transcriptions[task_id]["status"] = "error"
                    transcriptions[task_id]["error"] = error_msg
                    print(f"Serverless job failed ({job_id}): {error_msg}")
                    break

                if status == "IN_PROGRESS":
                    transcriptions[task_id]["status"] = "processing"
                    if not refresh_runpod_progress(transcriptions[task_id]):
                        mark_runpod_processing(transcriptions[task_id], "Cloud transcription")
                elif status == "IN_QUEUE":
                    transcriptions[task_id]["status"] = "processing"
                    mark_runpod_queued(transcriptions[task_id], "Cloud transcription")

                time.sleep(5)
            except Exception as e:
                print(f"Polling error for job {job_id}: {e}")
                time.sleep(10)

    try:
        s3_key, error_response = require_uploaded_s3_key(task)
        if error_response:
            return error_response
        known_speakers = normalize_known_speakers((request_body or {}).get("known_speakers"))
        task["progress_s3_key"] = get_progress_s3_key(task)

        url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": {
                "action": "full",
                "audio": s3_key,
                "progress_s3_key": task["progress_s3_key"],
                "s3_creds": {
                    "endpoint": S3_ENDPOINT,
                    "region": S3_REGION,
                    "access_key": os.getenv("RUNPOD_ACCESS_KEY"),
                    "secret_key": os.getenv("RUNPOD_SECRET_KEY"),
                    "bucket": S3_BUCKET
                },
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "num_speakers": num_speakers,
                "diarization_mode": (request_body or {}).get("diarization_mode") or "robust",
                "known_speakers": known_speakers,
                "hf_token": HF_TOKEN
            }
        }

        resp = http_requests.post(url, headers=headers, json=payload)
        resp_data = resp.json()
        job_id = resp_data.get("id")

        if job_id:
            task["status"] = "processing"
            task["progress"] = 10
            task["job_id"] = job_id
            task["runpod_submitted_at"] = time.time()
            task.pop("runpod_started_at", None)

            threading.Thread(target=poll_job, args=(job_id, task_id), daemon=True).start()
            print(f"Serverless full pipeline job started: {job_id} for {task_id}")
            return {"status": "started", "job_id": job_id}
        else:
            return {"status": "error", "error": f"Failed to start job: {resp_data}"}

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        return {"status": "error", "error": str(e)}


@app.post("/process-local/{task_id}")
async def process_local(task_id: str, request_body: dict | None = Body(default=None)):
    """Run local CPU OCR diarization for .mp4 files without RunPod."""
    if task_id not in transcriptions:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    task = transcriptions[task_id]
    if not is_video_filename(task.get("filename", task_id)):
        return JSONResponse(status_code=400, content={"error": "Local OCR test is only available for .mp4 files."})

    from paddle_ocr_factory import get_ocr_device

    task["status"] = "local_ocr_processing"
    task["progress"] = 5
    task["ocr_frame"] = 0
    task["ocr_frames_total"] = 0
    task["ocr_device"] = get_ocr_device()
    task.pop("error", None)
    persist_task_json(task_id)

    known_speakers = normalize_known_speakers((request_body or {}).get("known_speakers"))
    ocr_engine = str((request_body or {}).get("ocr_engine") or "paddle").lower()
    if ocr_engine not in {"paddle", "hunyuan", "hybrid", "paddle_hunyuan_fallback"}:
        return JSONResponse(status_code=400, content={"error": f"Unsupported local OCR engine: {ocr_engine}"})

    threading.Thread(target=run_local_ocr_task, args=(task_id, known_speakers, ocr_engine), daemon=True).start()
    return {"status": "started", "mode": "local_ocr", "ocr_engine": ocr_engine}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save file locally and begin S3 upload in background."""
    if not is_supported_audio_filename(file.filename):
        return JSONResponse(
            status_code=400,
            content={"error": "Only .m4a, .mp3, and .wav audio files are supported."},
        )

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_id = file.filename

    # Check if we already have a transcription for this file
    json_path = file_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcriptions[task_id] = data
                print(f"📎 Found existing transcription for {task_id}")
                return {"task_id": task_id}
        except:
            pass

    transcriptions[task_id] = {
        "filename": file.filename,
        "is_video": False,
        "status": "uploading",
        "progress": 0,
        "result": [],
    }

    # Start S3 upload in background
    t = threading.Thread(target=upload_to_s3, args=(file_path, task_id), daemon=True)
    t.start()

    return {"task_id": task_id}


@app.post("/improve-audio")
async def improve_audio(file: UploadFile = File(...)):
    """Clean a bass-heavy M4A into a transcription-ready M4A."""
    original_name = Path(file.filename or "audio.m4a")
    if original_name.suffix.lower() != ".m4a":
        return JSONResponse(status_code=400, content={"error": "Only .m4a files are supported here."})

    temp_input_path = UPLOAD_DIR / f"_improve_{uuid.uuid4().hex}.m4a"
    output_name = f"{original_name.stem}_improved.m4a"
    output_path = UPLOAD_DIR / output_name

    if output_path.exists():
        output_name = f"{original_name.stem}_improved_{uuid.uuid4().hex[:6]}.m4a"
        output_path = UPLOAD_DIR / output_name

    try:
        with open(temp_input_path, "wb") as buffer:
            buffer.write(await file.read())

        improve_audio_for_transcription(temp_input_path, output_path)
        print(f"🎚️ Improved audio created: {output_name}")
        return {
            "status": "success",
            "filename": output_name,
            "download_url": f"/download/{output_name}",
        }
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        return JSONResponse(status_code=500, content={"error": f"Audio improvement failed: {e}"})
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink()


@app.get("/s3-files")
async def get_s3_files():
    """Return the current live file list from the RunPod-connected S3 bucket."""
    try:
        return {"files": list_s3_bucket_files()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not load S3 files: {e}"})


class DeleteS3FileRequest(BaseModel):
    key: str


class DeleteAllS3FilesRequest(BaseModel):
    confirm: str


class ResetTranscriptionRequest(BaseModel):
    filename: str


@app.post("/delete-s3-file")
async def delete_s3_file(req: DeleteS3FileRequest):
    """Delete a file from the configured S3 bucket."""
    if not req.key:
        return JSONResponse(status_code=400, content={"error": "Missing S3 key."})

    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=req.key)
        print(f"🗑️ Deleted from S3: {req.key}")
        return {"status": "deleted", "key": req.key}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not delete S3 file: {e}"})


@app.post("/delete-all-s3-files")
async def delete_all_s3_files(req: DeleteAllS3FilesRequest):
    """Delete every object from RunPod S3 and clear local server-side cache."""
    if req.confirm != "DELETE ALL":
        return JSONResponse(status_code=400, content={"error": "Confirmation phrase mismatch."})

    try:
        files = list_s3_bucket_files()
        keys = [item["key"] for item in files]
        deleted_remote = delete_s3_keys(keys)
        deleted_local = delete_local_upload_files()
        print(f"🗑️ Deleted all server files: remote={deleted_remote}, local={deleted_local}")
        return {"status": "deleted", "deleted": deleted_remote, "deleted_remote": deleted_remote, "deleted_local": deleted_local}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not delete all S3 files: {e}"})


@app.get("/transcription-storage/{filename}")
async def get_transcription_storage(filename: str):
    """Report where a transcription JSON currently exists."""
    safe_filename = Path(filename).name
    json_name = get_json_name_for_filename(safe_filename)
    local_path = UPLOAD_DIR / json_name
    task = transcriptions.get(safe_filename)

    try:
        remote_keys = find_remote_json_keys(safe_filename, task)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not check remote JSON files: {e}"})

    return {
        "filename": safe_filename,
        "json_path": json_name,
        "local_json_exists": local_path.exists(),
        "remote_json_keys": remote_keys,
        "remote_json_exists": bool(remote_keys),
    }


@app.post("/reset-transcription")
async def reset_transcription(req: ResetTranscriptionRequest):
    """Delete stored transcription JSON while preserving the uploaded media."""
    safe_filename = Path(req.filename).name
    if not safe_filename:
        return JSONResponse(status_code=400, content={"error": "Missing filename."})

    task = transcriptions.get(safe_filename)
    json_name = get_json_name_for_filename(safe_filename)
    local_path = UPLOAD_DIR / json_name
    removed_local = False

    try:
        remote_keys = find_remote_json_keys(safe_filename, task)
        deleted_remote = delete_s3_keys(remote_keys)

        if local_path.exists():
            local_path.unlink()
            removed_local = True

        reusable_task = None
        if task and task.get("s3_key"):
            reusable_task = {
                "filename": safe_filename,
                "is_video": is_video_filename(safe_filename),
                "status": "uploaded",
                "progress": 100,
                "result": [],
                "s3_key": task["s3_key"],
            }
            transcriptions[safe_filename] = reusable_task
        else:
            transcriptions.pop(safe_filename, None)

        print(f"↩️ Reset transcription for {safe_filename}: local={removed_local}, remote={deleted_remote}")
        return {
            "status": "reset",
            "filename": safe_filename,
            "json_path": json_name,
            "deleted_local": removed_local,
            "deleted_remote": deleted_remote,
            "remote_json_keys": remote_keys,
            "task": serialize_task(reusable_task) if reusable_task else None,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not reset transcription: {e}"})


@app.get("/check/{filename}")
async def check_transcription(filename: str):
    """Check if a transcription JSON already exists for this audio file."""
    safe_filename = Path(filename).name

    # Check in-memory first
    if safe_filename in transcriptions and transcriptions[safe_filename].get("status") == "completed":
        data = serialize_task(transcriptions[safe_filename])
        try:
            data["remote_json_keys"] = find_remote_json_keys(safe_filename, transcriptions[safe_filename])
            data["remote_json_exists"] = bool(data["remote_json_keys"])
        except Exception:
            data["remote_json_keys"] = []
            data["remote_json_exists"] = False
        return data

    # Check on disk
    json_path = UPLOAD_DIR / get_json_name_for_filename(safe_filename)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcriptions[safe_filename] = data
                serialized = serialize_task(data)
                try:
                    serialized["remote_json_keys"] = find_remote_json_keys(safe_filename, data)
                    serialized["remote_json_exists"] = bool(serialized["remote_json_keys"])
                except Exception:
                    serialized["remote_json_keys"] = []
                    serialized["remote_json_exists"] = False
                return serialized
        except:
            pass

    # Check RunPod/S3 for a matching result JSON even if it is not cached locally yet.
    try:
        remote_keys = find_remote_json_keys(safe_filename)
        if remote_keys:
            s3.download_file(S3_BUCKET, remote_keys[0], str(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("filename", safe_filename)
            transcriptions[safe_filename] = data
            serialized = serialize_task(data)
            serialized["remote_json_keys"] = remote_keys
            serialized["remote_json_exists"] = True
            return serialized
    except Exception as e:
        print(f"⚠️ Remote transcription check failed for {safe_filename}: {e}")

    return {"status": "not_found"}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Return task status."""
    task = transcriptions.get(task_id)
    if not task:
        return {"status": "not_found"}
    if task.get("status") in {"processing", "diarizing", "transcribing"}:
        refresh_runpod_progress(task)
    return serialize_task(task)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio file for the local player."""
    return FileResponse(UPLOAD_DIR / filename)


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated result or audio files."""
    path = UPLOAD_DIR / filename
    if path.exists():
        media_type = "application/octet-stream"
        if filename.endswith(".json"):
            media_type = "application/json"
        elif filename.endswith(".m4a"):
            media_type = "audio/mp4"
        elif filename.endswith(".wav"):
            media_type = "audio/wav"
        return FileResponse(path, media_type=media_type, filename=filename)
    return {"error": "File not found"}


class UpdateSpeakerRequest(BaseModel):
    task_id: str
    segment_index: int
    speaker_name: str


@app.post("/update_speaker")
async def update_speaker(req: UpdateSpeakerRequest):
    """Bulk rename a speaker across all segments."""
    if req.task_id in transcriptions:
        task = transcriptions[req.task_id]
        if 0 <= req.segment_index < len(task["result"]):
            old_name = task["result"][req.segment_index]["speaker"]
            new_name = req.speaker_name

            for seg in task["result"]:
                if seg["speaker"] == old_name:
                    seg["speaker"] = new_name

            persist_task_json(req.task_id)
            return {"status": "success"}

    return {"status": "error", "message": "Task or segment not found"}


@app.get("/list")
async def list_transcriptions():
    """List all available transcriptions (for a file picker)."""
    results = []
    for json_file in UPLOAD_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "filename": data.get("filename"),
                    "segments": len(data.get("result", [])),
                    "status": data.get("status", "unknown"),
                })
        except:
            pass
    return results


# ═══════════════════════════════════════════
#  FULL POD AUTOMATION (SSH + GQL)
# ═══════════════════════════════════════════

class PodConfigRequest(BaseModel):
    ip: str = ""
    ssh_port: int = 22
    pod_id: str = ""
    endpoint_id: str = ""
    key_path: str = None


class EndpointWorkersRequest(BaseModel):
    endpoint_id: str = ""
    workers_max: int = Field(..., ge=0)

@app.post("/update-pod-config")
async def update_pod_config(req: PodConfigRequest):
    """Save Pod metadata and Serverless Endpoint ID to the current session and .env."""
    global RUNPOD_POD_ID, RUNPOD_ENDPOINT_ID
    
    if req.ip: pod_config["ip"] = req.ip
    if req.ssh_port: pod_config["ssh_port"] = req.ssh_port
    if req.pod_id: RUNPOD_POD_ID = req.pod_id
    if req.endpoint_id: RUNPOD_ENDPOINT_ID = req.endpoint_id
    if req.key_path: pod_config["key_path"] = req.key_path
    
    # Cleanly update .env
    env_lines = []
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            env_lines = f.readlines()
    
    # Filter out old keys
    keys_to_remove = ["POD_IP=", "POD_SSH_PORT=", "RUNPOD_POD_ID=", "POD_KEY_PATH=", "RUNPOD_ENDPOINT_ID="]
    env_lines = [l for l in env_lines if not any(k in l for k in keys_to_remove)]
    
    # Add new values (preserving some defaults if not provided in req but present in session)
    if pod_config["ip"]: env_lines.append(f"POD_IP={pod_config['ip']}\n")
    if pod_config["ssh_port"]: env_lines.append(f"POD_SSH_PORT={pod_config['ssh_port']}\n")
    if RUNPOD_POD_ID: env_lines.append(f"RUNPOD_POD_ID={RUNPOD_POD_ID}\n")
    if RUNPOD_ENDPOINT_ID: env_lines.append(f"RUNPOD_ENDPOINT_ID={RUNPOD_ENDPOINT_ID}\n")
    if pod_config["key_path"]: env_lines.append(f"POD_KEY_PATH={pod_config['key_path']}\n")
    
    with open(".env", "w") as f:
        f.writelines(env_lines)
    
    print(f"📡 Config Updated. Endpoint: {RUNPOD_ENDPOINT_ID}")
    return {"status": "updated"}

@app.get("/get-pod-config")
async def get_pod_config():
    """Return the current active config."""
    return {
        "ip": pod_config.get("ip"),
        "ssh_port": pod_config.get("ssh_port"),
        "pod_id": RUNPOD_POD_ID,
        "endpoint_id": RUNPOD_ENDPOINT_ID,
        "key_path": pod_config.get("key_path")
    }

def get_ssh_client():
    """Create an SSH client for the Pod, auto-detecting the SSH key."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try key locations in order of preference
    key_candidates = [
        pod_config.get("key_path", "runpod"),           # Project key
        os.path.expanduser("~/.ssh/id_ed25519"),        # Default Ed25519
        os.path.expanduser("~/.ssh/id_rsa"),            # Default RSA
    ]

    last_error = None
    for key_path in key_candidates:
        if not os.path.exists(key_path):
            continue
        try:
            # Try Ed25519 first, then RSA
            for KeyClass in [paramiko.Ed25519Key, paramiko.RSAKey]:
                try:
                    key = KeyClass.from_private_key_file(key_path)
                    ssh.connect(pod_config["ip"], port=pod_config["ssh_port"], username="root", pkey=key, timeout=10)
                    print(f"✅ SSH connected via {key_path}")
                    return ssh
                except (paramiko.ssh_exception.SSHException, ValueError):
                    continue
        except Exception as e:
            last_error = e
            continue

    raise Exception(f"Could not connect with any SSH key. Last error: {last_error}")

@app.post("/setup-pod")
async def setup_pod():
    """Deploy code to Pod and run the setup script."""
    try:
        ssh = get_ssh_client()
        
        # 1. Archive current project (excluding uploads, cache, .git)
        archive_path = "project.tar.gz"
        print("📦 Creating project archive...")
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in os.listdir("."):
                if item in ["uploads", "cache", ".git", "__pycache__", "legacy", "logs", "project.tar.gz"]:
                    continue
                tar.add(item)
        
        # 2. Upload to Pod
        print("📤 Uploading project to Pod...")
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(archive_path, "/workspace/project.tar.gz")
        
        # 3. Extract and Setup
        print("🛠️ Running setup on Pod...")
        commands = [
            "cd /workspace && mkdir -p transcriber && tar -xzf project.tar.gz -C transcriber",
            "cd /workspace/transcriber && mkdir -p uploads cache",
            f"echo 'RUNPOD_ACCESS_KEY={os.getenv('RUNPOD_ACCESS_KEY')}' > /workspace/transcriber/.env",
            f"echo 'RUNPOD_SECRET_KEY={os.getenv('RUNPOD_SECRET_KEY')}' >> /workspace/transcriber/.env",
            "cd /workspace/transcriber && bash setup_runpod.sh > worker.log 2>&1"
        ]
        
        full_cmd = " && ".join(commands)
        stdin, stdout, stderr = ssh.exec_command(full_cmd)
        
        # We don't wait for setup to finish in the request, we return 200 and let it run
        # but for this simplified version, let's at least start it.
        return {"status": "setup_started", "message": "Deploying and installing dependencies..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/start-transcription")
async def start_transcription():
    """Start the remote worker in a screen session."""
    try:
        ssh = get_ssh_client()
        # Start in a screen session so it persists
        cmd = "screen -dmS transcriber bash -c 'cd /workspace/transcriber && python remote_worker.py --watch > worker.log 2>&1'"
        ssh.exec_command(cmd)
        return {"status": "started", "message": "Worker started on Pod."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/pod-logs")
async def get_pod_logs():
    """Fetch the latest logs from the Pod worker."""
    try:
        ssh = get_ssh_client()
        stdin, stdout, stderr = ssh.exec_command("tail -n 50 /workspace/transcriber/worker.log")
        logs = stdout.read().decode("utf-8")
        return {"logs": logs}
    except Exception as e:
        return {"logs": f"Error fetching logs: {str(e)}"}

def runpod_gql(query):
    """Send a GraphQL request to RunPod API."""
    headers = {"Content-Type": "application/json", "api-key": RUNPOD_API_KEY}
    resp = http_requests.post(
        RUNPOD_GQL,
        params={"api_key": RUNPOD_API_KEY},
        json={"query": query},
        headers=headers,
    )
    return resp.json()


def get_runpod_endpoint_id(endpoint_id: str = ""):
    resolved_endpoint_id = (endpoint_id or RUNPOD_ENDPOINT_ID or "").strip()
    if not resolved_endpoint_id:
        raise ValueError("Serverless Endpoint ID is not configured.")
    return resolved_endpoint_id


def get_live_endpoint_workers(endpoint_id: str = ""):
    resolved_endpoint_id = get_runpod_endpoint_id(endpoint_id)
    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY is not configured.")

    query = """
    query GetMyEndpoints {
      myself {
        endpoints {
          id
          name
          gpuIds
          templateId
          workersMax
          workersMin
        }
      }
    }
    """
    result = runpod_gql(query)
    if result.get("errors"):
        message = result["errors"][0].get("message", "RunPod endpoint query failed.")
        raise RuntimeError(message)

    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"RunPod returned no data for endpoint query: {result}")

    myself = data.get("myself")
    if not isinstance(myself, dict):
        raise RuntimeError(
            "RunPod returned no endpoint owner data. "
            "This usually means the API key does not have GraphQL account access "
            "for listing/updating endpoints. Verify the key in RunPod Settings has "
            "GraphQL endpoint permissions, then retry. "
            f"Raw response: {result}"
        )

    endpoints = myself.get("endpoints")
    if not isinstance(endpoints, list):
        raise RuntimeError(f"RunPod returned malformed endpoint list: {result}")

    endpoint = next(
        (
            item for item in endpoints
            if isinstance(item, dict) and item.get("id") == resolved_endpoint_id
        ),
        None,
    )
    if not endpoint:
        raise LookupError(f"RunPod endpoint '{resolved_endpoint_id}' was not found.")
    return endpoint


def update_live_endpoint_workers(workers_max: int, endpoint_id: str = ""):
    resolved_endpoint_id = get_runpod_endpoint_id(endpoint_id)
    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY is not configured.")
    endpoint = get_live_endpoint_workers(resolved_endpoint_id)

    required_fields = {
        "name": endpoint.get("name"),
        "gpuIds": endpoint.get("gpuIds"),
        "templateId": endpoint.get("templateId"),
    }
    missing_fields = [key for key, value in required_fields.items() if not value]
    if missing_fields:
        raise RuntimeError(
            f"RunPod endpoint is missing required fields for update: {', '.join(missing_fields)}."
        )

    query = """
    mutation UpdateEndpointWorkers(
      $id: String!,
      $name: String!,
      $gpuIds: String!,
      $templateId: String!,
      $workersMax: Int!
    ) {
      saveEndpoint(input: {
        id: $id,
        name: $name,
        gpuIds: $gpuIds,
        templateId: $templateId,
        workersMax: $workersMax
      }) {
        id
        name
        gpuIds
        templateId
        workersMax
        workersMin
      }
    }
    """
    payload = {
        "query": query,
        "variables": {
            "id": resolved_endpoint_id,
            "name": required_fields["name"],
            "gpuIds": required_fields["gpuIds"],
            "templateId": required_fields["templateId"],
            "workersMax": workers_max,
        },
    }
    headers = {"Content-Type": "application/json", "api-key": RUNPOD_API_KEY}
    resp = http_requests.post(
        RUNPOD_GQL,
        params={"api_key": RUNPOD_API_KEY},
        json=payload,
        headers=headers,
    )
    result = resp.json()
    if result.get("errors"):
        message = result["errors"][0].get("message", "RunPod endpoint update failed.")
        raise RuntimeError(message)

    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"RunPod returned no data for endpoint update: {result}")

    endpoint = data.get("saveEndpoint")
    if not isinstance(endpoint, dict):
        raise RuntimeError(f"RunPod did not return updated endpoint data: {result}")
    return endpoint


@app.get("/endpoint-workers")
async def get_endpoint_workers(endpoint_id: str = ""):
    """Return the live worker limits for the configured Serverless endpoint."""
    try:
        endpoint = get_live_endpoint_workers(endpoint_id)
        return {
            "endpoint_id": endpoint.get("id", ""),
            "name": endpoint.get("name", ""),
            "workers_max": endpoint.get("workersMax"),
            "workers_min": endpoint.get("workersMin"),
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except LookupError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Could not load endpoint workers: {e}"})


@app.post("/endpoint-workers")
async def save_endpoint_workers(req: EndpointWorkersRequest):
    """Update the live max workers setting for the configured Serverless endpoint."""
    try:
        updated = update_live_endpoint_workers(req.workers_max, req.endpoint_id)
        live = get_live_endpoint_workers(req.endpoint_id or updated.get("id", ""))
        return {
            "status": "updated",
            "endpoint_id": live.get("id", ""),
            "name": live.get("name", ""),
            "workers_max": live.get("workersMax"),
            "workers_min": live.get("workersMin"),
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except LookupError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Could not update endpoint workers: {e}"})


@app.post("/start-pod")
async def start_pod():
    """Start the RunPod GPU Pod remotely."""
    if not RUNPOD_POD_ID:
        return {"error": "RUNPOD_POD_ID not set in .env"}
    query = f'mutation {{ podResume(input: {{ podId: "{RUNPOD_POD_ID}", gpuCount: 1 }}) {{ id desiredStatus }} }}'
    result = runpod_gql(query)
    print(f"🚀 Pod start requested: {result}")
    return {"status": "starting", "result": result}


@app.post("/stop-pod")
async def stop_pod():
    """Stop the RunPod GPU Pod remotely."""
    if not RUNPOD_POD_ID:
        return {"error": "RUNPOD_POD_ID not set in .env"}
    query = f'mutation {{ podStop(input: {{ podId: "{RUNPOD_POD_ID}" }}) {{ id desiredStatus }} }}'
    result = runpod_gql(query)
    print(f"⏹️ Pod stop requested: {result}")
    return {"status": "stopping", "result": result}


@app.get("/pod-status")
async def pod_status():
    """Check the current status of the RunPod Pod."""
    if not RUNPOD_POD_ID:
        return {"status": "not_configured"}
    try:
        query = f'{{ pod(input: {{ podId: "{RUNPOD_POD_ID}" }}) {{ id name desiredStatus runtime {{ uptimeInSeconds }} }} }}'
        result = runpod_gql(query)
        pod = result.get("data", {}).get("pod") or {}
        if not pod:
            return {"status": "NOT_FOUND"}
        return {
            "status": pod.get("desiredStatus", "UNKNOWN"),
            "name": pod.get("name", ""),
            "uptime": pod.get("runtime", {}).get("uptimeInSeconds") if pod.get("runtime") else None,
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


@app.post("/sync-now")
async def sync_now():
    """Manually trigger a cloud sync check."""
    download_results_from_s3()
    return {"status": "synced", "tasks": len(transcriptions)}


# ─── Static Files & Startup ───
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"🚀 Transcriber Pro (Local Dashboard) starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
