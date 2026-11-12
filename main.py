from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import torch
from pipeline import evaluate_event_metrics
from inference import load_checkpoint, recognize as recognize_strokes

# === Configuration ===
MODEL_PATH = os.getenv("MODEL_PATH", "multi_motion_model_7class.pt")
STATIC_DIR = os.getenv("STATIC_DIR", "./static")
CM_BASE_URL = os.getenv("CM_BASE_URL", "http://localhost:8000/static")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(STATIC_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}. Train it first with train.py (see LOCAL_SETUP.md).")
model, scaler, hand_to_slot = load_checkpoint(MODEL_PATH, DEVICE)
print("[INFO] Model weights loaded successfully.")

# === FastAPI Setup ===
app = FastAPI(title="Hand Strike Recognition API")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app.mount("/static", StaticFiles(directory="static"), name="static")


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

    
@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(video_file: UploadFile = File(...)):
    """Endpoint for robust hand strike recognition"""
    temp_path = ""
    try:
        # Validate input
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in [".mp4", ".mov", ".avi"]:
            raise HTTPException(400, "Invalid video format")
        
        print(f"[DEBUG] Uploaded video name: {video_file.filename}")

        # Save video
        vid_id = str(uuid.uuid4())
        temp_path = f"temp/{vid_id}{ext}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(await video_file.read())
        
        # Per-hand stroke events from the 7-class model
        try:
            events = recognize_strokes(temp_path, model, scaler, hand_to_slot, DEVICE)
        except ValueError as e:
            raise HTTPException(422, str(e))
        print(f"[DEBUG] Stroke events - left: {len(events['left'])}, right: {len(events['right'])}")

        # Process results
        results = evaluate_event_metrics(events["left"], events["right"], CM_BASE_URL, vid_id, STATIC_DIR)
        return JSONResponse(content={"status": "success", **results})
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[DEBUG] Temp video file deleted: {temp_path}")
        except Exception as e:
            print(f"[WARN] Failed to delete temp file: {e}")
