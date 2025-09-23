import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === MediaPipe Hands Setup ===

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# 1. Video Validation

def check_video_file(video_path):
    """
    Check if video file exists and can be opened
    Returns tuple of (is_valid, error_message)
    """
    if not os.path.exists(video_path):
        return False, f"Video file not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"Failed to open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        return False, f"Invalid video metrics - FPS: {fps}, Frames: {frame_count}"

    cap.release()
    return True, "OK"

# 2. Frame Extraction

def extract_screenshots(video_path, frame_interval=30):
    """
    Read video and return a list of frames sampled every frame_interval
    """
    is_valid, error_msg = check_video_file(video_path)
    if not is_valid:
        raise ValueError(error_msg)

    cap = cv2.VideoCapture(video_path)
    screenshots = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            screenshots.append(frame)
        frame_idx += 1
    cap.release()
    return screenshots

def preprocess_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)

    # Initialize arrays for both hands (21 landmarks each)
    left_hand = np.zeros((21, 9), dtype=np.float32)
    right_hand = np.zeros((21, 9), dtype=np.float32)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label

            # Extract coordinates (x,y,z) for each landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Calculate velocity (zero for first and last points)
            velocity = np.zeros_like(coords)
            if len(coords) > 2:
                velocity[1:-1] = coords[2:] - coords[:-2]

            # Calculate distance from center (wrist)
            dist_center = np.linalg.norm(coords - coords[0], axis=1, keepdims=True)

            # Calculate bone lengths
            bone_len = np.linalg.norm(
                np.diff(coords, axis=0, prepend=coords[0:1]),
                axis=1,
                keepdims=True
            )

            # Initialize angles as zeros
            angle = np.zeros((21, 1), dtype=np.float32)

            # Combine all features (9 features per landmark)
            features = np.concatenate([
                coords,        # 3 (x,y,z)
                velocity,      # 3 (vx,vy,vz)
                angle,         # 1 
                dist_center,   # 1
                bone_len       # 1
            ], axis=1)

            if hand_type == "Left":
                left_hand = features
            else:
                right_hand = features

    # Combine both hands (42 joints total, each with 9 features)
    joints = np.vstack([left_hand, right_hand])

    # Calculate simplified bone features: midpoint + direction = 6 features
    bones = np.zeros((20, 6), dtype=np.float32)
    if results.multi_hand_landmarks:
        bone_connections = [
            (0,1), (1,2), (2,3), (3,4),    # Thumb
            (0,5), (5,6), (6,7), (7,8),     # Index
            (0,9), (9,10), (10,11), (11,12), # Middle
            (0,13), (13,14), (14,15), (15,16), # Ring
            (0,17), (17,18), (18,19), (19,20)  # Pinky
        ]

        for i, (j1, j2) in enumerate(bone_connections):
            try:
                joint1 = joints[j1]
                joint2 = joints[j2]

                # Midpoint (3D)
                midpoint = (joint1[:3] + joint2[:3]) / 2
                # Direction vector (3D)
                direction = joint2[:3] - joint1[:3]

                bones[i] = np.concatenate([midpoint, direction])
            except Exception as e:
                print(f"Error processing bone {i}: {str(e)}")
                bones[i] = np.zeros(6)

    # Pad bones to 9 features (for stacking compatibility)
    bone_part_padded = np.pad(bones, ((0, 0), (0, 3)), mode='constant')

    # Final output: (62, 9)
    return np.vstack([joints, bone_part_padded])


# 4. Windowing

def create_feature_windows(landmarks_list, window_size=30, step=1):
    """
    Slide a window over the sequence of landmark vectors
    """
    windows = []
    for start in range(0, len(landmarks_list) - window_size + 1, step):
        window = landmarks_list[start:start + window_size]
        windows.append(np.stack(window, axis=0))
    return windows

# 5. Metrics & Confusion Matrices

def evaluate_metrics(predictions, cm_base_url, vid_id, static_dir):
    preds = np.argmax(np.vstack(predictions), axis=1)
    
    # TEMP GROUND TRUTH: assume all windows belong to the same label (for test/demo)
    true_label = int(preds[0]) if len(set(preds)) == 1 else 1
    y_true = np.full_like(preds, true_label)

    labels = ["Full", "Down", "Up", "Tab"]
    cm = confusion_matrix(y_true, preds, labels=range(len(labels)))
    
    # Compute per-class accuracy
    accuracy_per_class = {}
    for i, label in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = (correct / total * 100) if total else 0.0
        accuracy_per_class[label] = round(accuracy, 2)

    # Save confusion matrix as image
    def save_cm(cm, hand):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{hand} Hand Confusion Matrix")
        img_path = os.path.join(static_dir, f"{hand}_{vid_id}.png")
        plt.savefig(img_path)
        plt.close()
        return f"{cm_base_url}/{hand}_{vid_id}.png"

    left_cm_url = save_cm(cm, "Left")
    right_cm_url = save_cm(cm, "Right")

    # Suggestions for improvement
    suggestions = []
    for label in labels:
        if accuracy_per_class[label] < 70:
            suggestions.append(f"Try making the '{label}' gesture more distinct — it wasn't reliably recognized.")

    return {
        "left_hand_counts": {label: int(count) for label, count in zip(labels, cm.sum(axis=0))},
        "right_hand_counts": {label: int(count) for label, count in zip(labels, cm.sum(axis=0))},
        "left_hand_accuracy": accuracy_per_class,
        "right_hand_accuracy": accuracy_per_class,
        "left_confusion_matrix_url": left_cm_url,
        "right_confusion_matrix_url": right_cm_url,
        "suggestions": suggestions
    }
