from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import cv2
import uuid
import numpy as np
import torch
import torch.nn as nn
from pipeline import (
    check_video_file,
    extract_screenshots,
    preprocess_landmarks,
    create_feature_windows,
    evaluate_metrics
)

class LieGroupEmbedding(nn.Module):
    def __init__(self, input_dim, lie_dim):
        super(LieGroupEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, lie_dim)

    def forward(self, x):
        batch, time, nodes, features = x.size()
        x = x.reshape(batch * time * nodes, features)  # Safe reshape
        xi = self.fc(x)
        xi = xi.view(batch, time, nodes, -1)
        return torch.cumsum(xi, dim=1)


class DynamicAdjacencyMatrix(nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(DynamicAdjacencyMatrix, self).__init__()
        self.num_nodes = num_nodes
        self.node_fc = nn.Linear(input_dim, num_nodes)

    def forward(self, X):
        # X: (B, T, N, F) -> average over time only, keep each node's identity
        X = X.mean(dim=1)  # (B, N, F)

        A = self.node_fc(X)  # (B, N, N)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
        return torch.softmax(A, dim=-1)

class ResidualCTRGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(ResidualCTRGCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x.clone()
        x = x.permute(0, 3, 1, 2)  # (B, C, T, N)
        x = self.conv(x)
        x = self.bn(x)
        residual = residual.permute(0, 3, 1, 2)
        residual = self.residual_conv(residual)
        out = self.relu(x + residual)  # (B, C, T, N)
        return out.permute(0, 2, 3, 1)  # back to (B, T, N, C)

class MultiMotionDevLSTMModel(nn.Module):
    def __init__(self, num_classes=4):  # Match saved model's 4 classes
        super(MultiMotionDevLSTMModel, self).__init__()

        # Joint path (42 joints)
        self.lie_embed_joint = LieGroupEmbedding(input_dim=9, lie_dim=16)
        self.dynamic_adj_joint = DynamicAdjacencyMatrix(input_dim=16, num_nodes=42)
        self.residual_joint = ResidualCTRGCNLayer(in_channels=16, out_channels=64, num_nodes=42)
        self.lstm_joint = nn.LSTM(input_size=64*42, hidden_size=64, batch_first=True)  # 2688 in features

        # Bone path (20 bones)
        self.lie_embed_bone = LieGroupEmbedding(input_dim=6, lie_dim=16)
        self.dynamic_adj_bone = DynamicAdjacencyMatrix(input_dim=16, num_nodes=20)
        self.residual_bone = ResidualCTRGCNLayer(in_channels=16, out_channels=64, num_nodes=20)
        self.lstm_bone = nn.LSTM(input_size=64*20, hidden_size=64, batch_first=True)  # 1280 in features

        self.fc = nn.Linear(128, num_classes)  # 4 output classes

    def forward(self, x):
        B, T, N, _ = x.shape
        
        # === JOINT PATH ===
        x_joint = self.lie_embed_joint(x[:, :, :42, :9])  # (B,T,42,16)
        A_joint = self.dynamic_adj_joint(x_joint)  # (B,42,42)
        x_joint = torch.einsum('bnm,btmf->btnf', A_joint, x_joint)  # (B,T,42,16)
        x_joint = self.residual_joint(x_joint)  # (B,T,42,64)
        x_joint = x_joint.reshape(B, T, -1)  # (B,T,2688)
        
        # === BONE PATH ===
        x_bone = self.lie_embed_bone(x[:, :, 42:, :6])  # (B,T,20,16)
        A_bone = self.dynamic_adj_bone(x_bone)  # (B,20,20)
        x_bone = torch.einsum('bnm,btmf->btnf', A_bone, x_bone)  # (B,T,20,16)
        x_bone = self.residual_bone(x_bone)  # (B,T,20,64)
        x_bone = x_bone.reshape(B, T, -1)  # (B,T,1280)
        
        # === LSTM PROCESSING ===
        _, (h_joint, _) = self.lstm_joint(x_joint)  # (1,B,64)
        _, (h_bone, _) = self.lstm_bone(x_bone)  # (1,B,64)
        
        # === FINAL CLASSIFICATION ===
        out = torch.cat([h_joint[-1], h_bone[-1]], dim=1)  # (B,128)
        return self.fc(out)  # (B,4)
    
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
        
        # Process landmarks
        landmarks = []
        for frame in frames:
            try:
                lm = preprocess_landmarks(frame)
                landmarks.append(lm if lm.shape == (62, 9) else np.zeros((62, 9)))
            except:
                landmarks.append(np.zeros((62, 9)))
        
        # Create windows
        windows = []
        print(f"[DEBUG] Total landmark sets extracted: {len(landmarks)}")
        print(f"[DEBUG] Example landmark shape: {landmarks[0].shape if landmarks else 'None'}")
        for i in range(0, len(landmarks) - window_size + 1, window_size // 2):
            window = np.stack(landmarks[i:i+window_size])
            if window.shape == (window_size, 62, 9):
                windows.append(window)
        
        print(f"[DEBUG] Total windows created: {len(windows)}")
        if windows:
            print(f"[DEBUG] Each window shape: {windows[0].shape}")
        if not windows:
            raise HTTPException(422, "No valid windows created")
        
        # Model inference
        model.eval()
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
        
        print(f"[DEBUG] Total predictions made: {len(predictions)}")
        if predictions:
            print(f"[DEBUG] Example prediction: {predictions[0]}")
        if not predictions:
            raise HTTPException(422, "No valid predictions")
        
        # Process results (✅ now passing all args)
        results = evaluate_metrics(predictions, CM_BASE_URL, vid_id, STATIC_DIR)
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
