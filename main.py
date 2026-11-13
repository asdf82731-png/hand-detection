import json
import time

import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import threading
import uuid
import torch
from pipeline import evaluate_event_metrics
from inference import analyze, load_checkpoint
from pipeline import LABELS

# === Configuration ===
MODEL_PATH = os.getenv("MODEL_PATH", "multi_motion_model_7class.pt")
STATIC_DIR = os.getenv("STATIC_DIR", "./static")
# Relative by default, so the images load whatever host and port the page is served from.
CM_BASE_URL = os.getenv("CM_BASE_URL", "/static")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limits, so one upload cannot fill the disk or tie the server up for hours.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "600"))
# Confusion-matrix images and finished jobs older than this are deleted.
RETENTION_SECONDS = float(os.getenv("RETENTION_SECONDS", "3600"))
# Comma-separated origins allowed to call the API from another site. Empty
# (the default) allows only the page this server serves itself.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

os.makedirs(STATIC_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}. Train it first with train.py (see LOCAL_SETUP.md).")
model, scaler, hand_to_slot = load_checkpoint(MODEL_PATH, DEVICE)
print("[INFO] Model weights loaded successfully.")

# === FastAPI Setup ===
app = FastAPI(title="Hand Strike Recognition API")

if ALLOWED_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("index.html")

class RecognitionResponse(BaseModel):
    status: str
    left_hand_counts: dict
    right_hand_counts: dict
    left_hand_accuracy: dict
    right_hand_accuracy: dict
    left_confusion_matrix_url: str
    right_confusion_matrix_url: str
    suggestions: list
    accuracy_basis: dict = {}
    stroke_metrics: dict = {}
    detection_rate: dict = {}

    
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi")


UPLOAD_CHUNK = 1024 * 1024


def check_video_length(path):
    """Raise HTTPException(413) for a video longer than MAX_VIDEO_SECONDS, (422) for one that cannot be read."""
    cap = cv2.VideoCapture(path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise HTTPException(422, "Cannot read this video")
    if frames / fps > MAX_VIDEO_SECONDS:
        raise HTTPException(413, f"Video is {frames / fps:.0f} s long; the limit is {MAX_VIDEO_SECONDS:.0f} s")


async def _save_upload(video_file):
    """
    Validate the extension, write the upload to temp/ in chunks (refusing it
    once it passes MAX_UPLOAD_MB) and check its length. Returns (vid_id, temp_path).
    """
    ext = os.path.splitext(video_file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid video format")
    print(f"[DEBUG] Uploaded video name: {video_file.filename}")
    vid_id = str(uuid.uuid4())
    temp_path = f"temp/{vid_id}{ext}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(temp_path, "wb") as f:
            while chunk := await video_file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Video is larger than {MAX_UPLOAD_MB:.0f} MB")
                f.write(chunk)
        check_video_length(temp_path)
    except HTTPException:
        _remove(temp_path)
        raise
    cleanup_old_files()
    return vid_id, temp_path


def cleanup_old_files(now=None):
    """Delete confusion-matrix images and finished jobs older than RETENTION_SECONDS."""
    now = time.time() if now is None else now
    for name in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, name)
        try:
            if name.endswith(".png") and now - os.path.getmtime(path) > RETENTION_SECONDS:
                os.remove(path)
        except OSError:
            pass
    with _jobs_lock:
        for job_id in [j for j, job in _jobs.items()
                       if job["status"] != "running" and now - job.get("finished_at", now) > RETENTION_SECONDS]:
            del _jobs[job_id]


def parse_expected_counts(text):
    """
    Optional form field: how many of each stroke the user set out to play, as
    JSON {"left": {"Full": 10, ...}, "right": {...}}. Empty means not given.
    Raises HTTPException(400) for anything else.
    """
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text)
        expected = {}
        for hand, counts in data.items():
            if hand not in ("left", "right") or not isinstance(counts, dict):
                raise ValueError(hand)
            for label, count in counts.items():
                if label not in LABELS or not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(label)
            positive = {label: count for label, count in counts.items() if count > 0}
            if positive:
                expected[hand] = positive
        return expected or None
    except (ValueError, AttributeError):
        raise HTTPException(400, 'expected_counts must be JSON like {"left": {"Full": 10}, "right": {"Down": 5}} '
                                 f"with stroke names from {LABELS} and whole numbers of 0 or more")


def _analyze(temp_path, vid_id, expected=None, on_progress=None):
    """Video on disk -> the response fields. Raises ValueError for a video that cannot be analyzed."""
    analysis = analyze(temp_path, model, scaler, hand_to_slot, DEVICE, on_progress)
    events = analysis["events"]
    print(f"[DEBUG] Stroke events - left: {len(events['left'])}, right: {len(events['right'])}")
    results = evaluate_event_metrics(events["left"], events["right"], CM_BASE_URL, vid_id, STATIC_DIR, expected)
    return {"status": "success", **results, "stroke_metrics": analysis["metrics"],
            "detection_rate": analysis["detection_rate"]}


def _remove(temp_path):
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[DEBUG] Temp video file deleted: {temp_path}")
    except Exception as e:
        print(f"[WARN] Failed to delete temp file: {e}")


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(video_file: UploadFile = File(...), expected_counts: str = Form("")):
    """Analyze an upload and return the result when done (no progress reporting; see /jobs)."""
    temp_path = ""
    try:
        expected = parse_expected_counts(expected_counts)
        vid_id, temp_path = await _save_upload(video_file)
        try:
            return JSONResponse(content=_analyze(temp_path, vid_id, expected))
        except ValueError as e:
            raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        _remove(temp_path)


# === Background jobs, so the page can show real progress ===
# In-memory: jobs are lost on restart, and finished jobs are dropped after
# RETENTION_SECONDS. Enough for a single server; several server processes
# would need a shared store.
_jobs = {}
_jobs_lock = threading.Lock()


def _update_job(job_id, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _run_job(job_id, temp_path, vid_id, expected):
    try:
        result = _analyze(temp_path, vid_id, expected, lambda fraction: _update_job(job_id, progress=round(fraction, 3)))
        _update_job(job_id, status="done", progress=1.0, result=result, finished_at=time.time())
    except ValueError as e:
        _update_job(job_id, status="error", detail=str(e), finished_at=time.time())
    except Exception as e:
        _update_job(job_id, status="error", detail=f"Processing error: {str(e)}", finished_at=time.time())
    finally:
        _remove(temp_path)


@app.post("/jobs")
async def create_job(video_file: UploadFile = File(...), expected_counts: str = Form("")):
    """Start analyzing an upload in the background. Returns {"job_id"}; poll GET /jobs/{job_id}."""
    expected = parse_expected_counts(expected_counts)
    vid_id, temp_path = await _save_upload(video_file)
    with _jobs_lock:
        _jobs[vid_id] = {"status": "running", "progress": 0.0}
    threading.Thread(target=_run_job, args=(vid_id, temp_path, vid_id, expected), daemon=True).start()
    return {"job_id": vid_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """{"status": "running" | "done" | "error", "progress": 0-1, "result" when done, "detail" on error}."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return {k: v for k, v in job.items() if k != "finished_at"}
