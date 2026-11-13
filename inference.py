"""
Inference with the retrained 7-class checkpoint written by train.py: video (or
frame folder) -> each hand's detected strokes, ready for
pipeline.evaluate_event_metrics(). Used by main.py.
"""
import numpy as np
import torch

from features import (
    FeatureScaler, extract_hand_tracks, pack_model_input, resample_to_fps, source_frame_count, video_fps,
    window_features, window_state_features,
)
from model import StrokeBiLSTMModel
from pipeline import LABELS

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
    """
    Returns (model, scaler, hand_to_slot) from a train.py checkpoint. The
    model is built with the embedding and hand-state settings the checkpoint
    was trained with; checkpoints from before those options use the defaults.
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = StrokeBiLSTMModel(
        num_classes=len(checkpoint["labels"]),
        embedding=config.get("embedding", "cumulative"),
        state_dim=config.get("state_dim", 0),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    # Whether the hand-state wrist angle was measured against the forearm;
    # inference must measure it the same way.
    model.pose_angle = bool(config.get("pose_angle", False))
    return model, FeatureScaler.from_state_dict(checkpoint["scaler"]), checkpoint["hand_to_slot"]


@torch.no_grad()
def predict_window_labels(positions, slot, model, scaler, device, states=None):
    """
    One hand's positions (N, 21, 3) -> predicted label index for every window,
    step 1 (N - 29 of them). states (N, 2) is required when the model uses
    hand-state features. Windows are built one batch at a time, so a long
    video does not need all of them in memory at once.
    """
    starts = range(len(positions) - WINDOW_SIZE + 1)
    labels = []
    for i in range(0, len(starts), BATCH_SIZE):
        batch = starts[i:i + BATCH_SIZE]
        inputs = np.stack([
            pack_model_input(*scaler.transform(*window_features(positions[s:s + WINDOW_SIZE])), slot) for s in batch
        ])
        args = [torch.from_numpy(inputs).to(device)]
        if model.state_dim:
            state_inputs = np.stack([
                scaler.transform_state(window_state_features(states[s:s + WINDOW_SIZE])) for s in batch
            ]).astype(np.float32)
            args.append(torch.from_numpy(state_inputs).to(device))
        labels.extend(model(*args).argmax(dim=1).tolist())
    return labels


def stroke_spans(window_labels):
    """
    Per-window label indices -> one (stroke index 0-3, first frame, last frame)
    per detected stroke. Runs shorter than MIN_RUN_WINDOWS are dropped first,
    then consecutive windows with the same label count as one stroke; holds /
    none are not counted but do separate two strokes of the same kind. A
    window's label describes its last frame, so window i covers frame
    i + WINDOW_SIZE - 1.
    """
    runs = []
    for index, label in enumerate(window_labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = index
        else:
            runs.append([label, index, index])

    spans, previous = [], None
    offset = WINDOW_SIZE - 1
    for label, first, last in runs:
        if last - first + 1 < MIN_RUN_WINDOWS:
            continue
        if label < NUM_STROKES:
            if label != previous:
                spans.append([label, first + offset, last + offset])
            else:
                spans[-1][2] = last + offset  # same stroke, interrupted only by a dropped flicker
        previous = label
    return [tuple(span) for span in spans]


def stroke_events(window_labels):
    """Per-window label indices -> stroke indices (0-3), one per detected stroke (see stroke_spans)."""
    return [label for label, _, _ in stroke_spans(window_labels)]


def _mean_and_cv(values):
    """(mean, coefficient of variation); CV is None with fewer than two values or a zero mean."""
    if not values:
        return None, None
    mean = float(np.mean(values))
    cv = float(np.std(values) / abs(mean)) if len(values) >= 2 and mean != 0 else None
    return round(mean, 3), None if cv is None else round(cv, 3)


def stroke_metrics(spans, states, fps=TRAIN_FPS):
    """
    Measurements of how each stroke was played (thesis 1.2-1.3: angle, force,
    consistency), from one hand's stroke spans and wrist states (N, 2):

    - peak_height: highest wrist position during the stroke, in palm lengths.
    - angle_range_deg: how far the wrist angle travelled during the stroke.
    - peak_down_speed: fastest downward wrist speed, in palm lengths per second.
      A camera cannot measure force; downward speed is used as a stand-in.
    - each of the above as a mean and a coefficient of variation (CV: lower
      is more consistent) per stroke type, and the time between consecutive
      strokes of this hand (interval_s) as a mean and CV.

    These are measurements, not judgements: no target values are defined.
    """
    per_type = {name: {"peak_height": [], "angle_range_deg": [], "peak_down_speed": []} for name in LABELS}
    for label, first, last in spans:
        segment = np.asarray(states[first:last + 1], dtype=np.float32)
        height, angle = segment[:, 0], segment[:, 1]
        record = per_type[LABELS[label]]
        record["peak_height"].append(float(height.max()))
        record["angle_range_deg"].append(float(np.degrees(angle.max() - angle.min())))
        record["peak_down_speed"].append(float(max((-np.diff(height) * fps).max(), 0.0)) if len(height) > 1 else 0.0)

    strokes = {}
    for name, record in per_type.items():
        entry = {"count": len(record["peak_height"])}
        for key, values in record.items():
            entry[f"{key}_mean"], entry[f"{key}_cv"] = _mean_and_cv(values)
        strokes[name] = entry
    starts = [first for _, first, _ in spans]
    interval_mean, interval_cv = _mean_and_cv([(b - a) / fps for a, b in zip(starts, starts[1:])])
    return {"strokes": strokes, "interval_s_mean": interval_mean, "interval_s_cv": interval_cv}


def to_training_fps(positions, fps):
    """Resample each hand's positions to TRAIN_FPS unless fps is unknown or already within FPS_TOLERANCE of it."""
    if fps is None or abs(fps - TRAIN_FPS) / TRAIN_FPS <= FPS_TOLERANCE:
        return positions
    return {hand: resample_to_fps(coords, fps, TRAIN_FPS) for hand, coords in positions.items()}


def analyze(source, model, scaler, hand_to_slot, device, on_progress=None):
    """
    Video file or frame folder -> {
        "events": {"left": [stroke indices], "right": [...]},
        "spans": {"left": [(stroke index, first frame, last frame)], ...},
        "metrics": {"left": stroke_metrics(...), ...},
        "detection_rate": {"left": 0-1, "right": 0-1},
        "frame_scale": source frames per analysis frame (1.0 unless resampled),
    }
    with left/right meaning what they meant in the training labels. Frames are
    counted at TRAIN_FPS after any resampling. on_progress(fraction) is called
    as the work advances, ending at 1.0.
    """
    report = on_progress or (lambda fraction: None)
    total = source_frame_count(source)

    def on_frame(done):
        if total:
            report(DETECTION_SHARE * min(done / total, 1.0))

    positions, states, detection_rate = extract_hand_tracks(source, on_frame, getattr(model, "pose_angle", False))
    fps = video_fps(source)
    resampled = to_training_fps(positions, fps)
    frame_scale = 1.0
    if resampled is not positions:
        states = to_training_fps(states, fps)
        frame_scale = fps / TRAIN_FPS
    positions = resampled
    report(DETECTION_SHARE)

    result = {"events": {}, "spans": {}, "metrics": {}, "detection_rate": {}, "frame_scale": frame_scale}
    for hand, slot in hand_to_slot.items():
        if len(positions[slot]) < WINDOW_SIZE:
            raise ValueError(f"Video too short (min {WINDOW_SIZE} frames required)")
        if detection_rate[slot] == 0:
            spans = []  # hand never seen: nothing to classify
        else:
            spans = stroke_spans(predict_window_labels(positions[slot], slot, model, scaler, device, states[slot]))
        result["events"][hand] = [label for label, _, _ in spans]
        result["spans"][hand] = spans
        result["metrics"][hand] = stroke_metrics(spans, states[slot])
        result["detection_rate"][hand] = round(detection_rate[slot], 3)
    report(1.0)
    return result


def recognize(source, model, scaler, hand_to_slot, device, on_progress=None):
    """Video file or frame folder -> {"left": [stroke indices], "right": [...]} (see analyze)."""
    return analyze(source, model, scaler, hand_to_slot, device, on_progress)["events"]
