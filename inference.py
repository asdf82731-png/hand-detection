"""
Inference with the retrained 7-class checkpoint written by train.py: video (or
frame folder) -> each hand's detected strokes, ready for
pipeline.evaluate_event_metrics(). Used by main.py.
"""
import numpy as np
import torch

from features import (
    FeatureScaler, extract_hand_positions, pack_model_input, resample_to_fps, source_frame_count, video_fps,
    window_features,
)
from model import StrokeBiLSTMModel

WINDOW_SIZE = 30
# The training footage was filmed at 30fps (thesis 4.2). Uploads at another
# frame rate are resampled to it so a window spans the same 1 second and
# velocities mean the same thing; 29.97fps is close enough to leave alone.
TRAIN_FPS = 30
FPS_TOLERANCE = 0.02
# Share of the progress bar given to hand detection; the model takes the rest.
DETECTION_SHARE = 0.9
BATCH_SIZE = 64

# Labels 0-3 are the four strokes that get counted (same order as
# pipeline.LABELS); the rest are holds / none and only separate strokes.
NUM_STROKES = 4

# With one prediction per frame, a label that lasts fewer windows than this is
# treated as flicker and ignored. Untuned: revisit once a real checkpoint exists.
MIN_RUN_WINDOWS = 3


def load_checkpoint(path, device):
    """Returns (model, scaler, hand_to_slot) from a train.py checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = StrokeBiLSTMModel(num_classes=len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, FeatureScaler.from_state_dict(checkpoint["scaler"]), checkpoint["hand_to_slot"]


@torch.no_grad()
def predict_window_labels(positions, slot, model, scaler, device):
    """One hand's positions (N, 21, 3) -> predicted label index for every window, step 1 (N - 29 of them)."""
    inputs = np.stack([
        pack_model_input(*scaler.transform(*window_features(positions[start:start + WINDOW_SIZE])), slot)
        for start in range(len(positions) - WINDOW_SIZE + 1)
    ])
    labels = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[i:i + BATCH_SIZE]).to(device)
        labels.extend(model(batch).argmax(dim=1).tolist())
    return labels


def stroke_events(window_labels):
    """
    Per-window label indices -> stroke indices (0-3), one per detected stroke.
    Runs shorter than MIN_RUN_WINDOWS are dropped first, then consecutive
    windows with the same label count as one stroke; holds / none are not
    counted but do separate two strokes of the same kind.
    """
    runs = []
    for label in window_labels:
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])

    events, previous = [], None
    for label, length in runs:
        if length < MIN_RUN_WINDOWS:
            continue
        if label != previous and label < NUM_STROKES:
            events.append(label)
        previous = label
    return events


def to_training_fps(positions, fps):
    """Resample each hand's positions to TRAIN_FPS unless fps is unknown or already within FPS_TOLERANCE of it."""
    if fps is None or abs(fps - TRAIN_FPS) / TRAIN_FPS <= FPS_TOLERANCE:
        return positions
    return {hand: resample_to_fps(coords, fps, TRAIN_FPS) for hand, coords in positions.items()}


def recognize(source, model, scaler, hand_to_slot, device, on_progress=None):
    """
    Video file or frame folder -> {"left": [stroke indices], "right": [...]},
    with left/right meaning what they meant in the training labels.
    on_progress(fraction) is called as the work advances, ending at 1.0.
    """
    report = on_progress or (lambda fraction: None)
    total = source_frame_count(source)

    def on_frame(done):
        if total:
            report(DETECTION_SHARE * min(done / total, 1.0))

    positions, detection_rate = extract_hand_positions(source, on_frame)
    positions = to_training_fps(positions, video_fps(source))
    report(DETECTION_SHARE)
    events = {}
    for hand, slot in hand_to_slot.items():
        if len(positions[slot]) < WINDOW_SIZE:
            raise ValueError(f"Video too short (min {WINDOW_SIZE} frames required)")
        if detection_rate[slot] == 0:
            events[hand] = []  # hand never seen: nothing to classify
        else:
            events[hand] = stroke_events(predict_window_labels(positions[slot], slot, model, scaler, device))
    report(1.0)
    return events
