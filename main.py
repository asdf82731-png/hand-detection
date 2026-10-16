from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import cv2
import uuid
import numpy as np
import torch
from pipeline import (
    check_video_file,
    extract_screenshots,
    preprocess_landmarks,
    preprocess_landmarks_per_hand,
    create_feature_windows,
    evaluate_metrics
)
from model import MultiMotionDevLSTMModel

# === Configuration ===
MODEL_PATH = os.getenv("MODEL_PATH", "multi_motion_model.pt")
STATIC_DIR = os.getenv("STATIC_DIR", "./static")
CM_BASE_URL = os.getenv("CM_BASE_URL", "http://localhost:8000/static")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(STATIC_DIR, exist_ok=True)

model = MultiMotionDevLSTMModel(num_classes=4)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)  # strict: fail loudly if the architecture ever drifts from the checkpoint again

model.to(DEVICE).eval()
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
        
        # Extract frames
        cap = cv2.VideoCapture(temp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                print("[DEBUG] Skipped null frame (frame is None)")
                continue
            frames.append(frame)
        cap.release()
        print(f"[DEBUG] Total frames extracted: {len(frames)}")
        
        window_size = 30
        if len(frames) < window_size:
            raise HTTPException(422, f"Video too short (min {window_size} frames required)")
        
        # Process landmarks independently per hand (other hand's data zeroed out)
        left_landmarks = []
        right_landmarks = []
        for frame in frames:
            try:
                left_lm, right_lm = preprocess_landmarks_per_hand(frame)
            except Exception:
                left_lm, right_lm = np.zeros((62, 9)), np.zeros((62, 9))
            left_landmarks.append(left_lm if left_lm.shape == (62, 9) else np.zeros((62, 9)))
            right_landmarks.append(right_lm if right_lm.shape == (62, 9) else np.zeros((62, 9)))

        print(f"[DEBUG] Total landmark sets extracted: {len(left_landmarks)}")

        def build_windows(landmarks_list):
            windows = []
            for i in range(0, len(landmarks_list) - window_size + 1, window_size // 2):
                window = np.stack(landmarks_list[i:i+window_size])
                if window.shape == (window_size, 62, 9):
                    windows.append(window)
            return windows

        left_windows = build_windows(left_landmarks)
        right_windows = build_windows(right_landmarks)

        print(f"[DEBUG] Left windows: {len(left_windows)}, Right windows: {len(right_windows)}")
        if not left_windows or not right_windows:
            raise HTTPException(422, "No valid windows created")

        # Model inference
        model.eval()

        def run_inference(windows):
            predictions = []
            for window in windows:
                try:
                    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor)
                    predictions.append(output.numpy())
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")
                    continue
            return predictions

        left_predictions = run_inference(left_windows)
        right_predictions = run_inference(right_windows)

        print(f"[DEBUG] Left predictions: {len(left_predictions)}, Right predictions: {len(right_predictions)}")
        if not left_predictions or not right_predictions:
            raise HTTPException(422, "No valid predictions")

        # Process results
        results = evaluate_metrics(left_predictions, right_predictions, CM_BASE_URL, vid_id, STATIC_DIR)
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
