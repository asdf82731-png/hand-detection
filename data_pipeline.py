"""
Training data pipeline: CSV label loading, frame alignment, window labeling,
and train/val split. Spec: TRAINING_DESIGN.md section 1.

Pure logic only — no video I/O here. Callers extract per-hand (62, 9) feature
blocks per frame (reusing pipeline.preprocess_landmarks_per_hand) and pass the
resulting list in to windows_to_samples().
"""
import csv
import re

import numpy as np

LABELS = ["full_stroke", "down_stroke", "up_stroke", "tab_stroke", "down_hold", "up_hold", "none"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

_LABEL_ALIASES = {
    "full_strike": "full_stroke",
    "down_strike": "down_stroke",
    "up_strike": "up_stroke",
    "tab_strike": "tab_stroke",
    "full_stroke": "full_stroke",
    "down_stroke": "down_stroke",
    "up_stroke": "up_stroke",
    "tab_stroke": "tab_stroke",
    "down_hold": "down_hold",
    "up_hold": "up_hold",
    "none": "none",
}

_FRAME_ID_RE = re.compile(r"(\d+)")


def normalize_label(raw_label):
    """Map CSV raw stroke values (e.g. '_strike' suffix) to the canonical 7-class label."""
    key = raw_label.strip().lower()
    label = _LABEL_ALIASES.get(key)
    if label is None:
        raise ValueError(f"Unknown stroke label: {raw_label!r}")
    return label


def parse_frame_id(frame_id):
    """'frame_0026.jpg' -> 26"""
    match = _FRAME_ID_RE.search(frame_id)
    if not match:
        raise ValueError(f"Cannot parse frame index from frame_id: {frame_id!r}")
    return int(match.group(1))


def load_frame_labels(csv_path):
    """
    Read a `frame_id, hand, stroke` CSV into {frame_index: {"left": label, "right": label}}.
    Frames/hands not present in the CSV are simply absent from the result (see
    TRAINING_DESIGN.md 1.3 — missing frames are excluded, never assumed "none").
    """
    frame_labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = parse_frame_id(row["frame_id"])
            hand = row["hand"].strip().lower()
            label = normalize_label(row["stroke"])
            frame_labels.setdefault(frame_idx, {})[hand] = label
    return frame_labels


def build_labeled_windows(frame_labels, hand, num_frames, window_size=30, step=15):
    """
    Slide a window over [0, num_frames) and label each window with its LAST
    frame's label for the given hand (TRAINING_DESIGN.md 1.4). A window is
    dropped entirely if any frame inside it lacks a label for this hand
    (TRAINING_DESIGN.md 1.3).

    Returns a list of (start_frame, label_index) tuples.
    """
    windows = []
    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size  # exclusive
        if not all(fi in frame_labels and hand in frame_labels[fi] for fi in range(start, end)):
            continue
        last_label = frame_labels[end - 1][hand]
        windows.append((start, LABEL_TO_INDEX[last_label]))
    return windows


def split_train_val(windows, num_frames, window_size=30, val_fraction=0.2, buffer=None):
    """
    Time-based train/val split (TRAINING_DESIGN.md 1.6). The first
    (1 - val_fraction) fraction of frames is train, the rest is val. Any
    window overlapping the buffer zone around the split point is dropped
    from both sets to prevent leakage from the 50%-overlapping windows.
    """
    if buffer is None:
        buffer = window_size
    split_point = round(num_frames * (1 - val_fraction))
    buffer_start = split_point - buffer
    buffer_end = split_point + buffer

    train, val = [], []
    for start, label in windows:
        end = start + window_size  # exclusive
        if end > buffer_start and start < buffer_end:
            continue  # straddles (or sits inside) the buffer zone — drop
        if end <= split_point:
            train.append((start, label))
        else:
            val.append((start, label))
    return train, val


def windows_to_samples(hand_features, hand, windows, window_size=30):
    """
    hand_features: sequence of per-frame (62, 9) feature blocks for one hand
    (already masked via pipeline.preprocess_landmarks_per_hand), indexed by
    frame number.

    Returns a list of (window: np.ndarray[window_size, 62, 9], hand: str, label: int).
    """
    samples = []
    for start, label in windows:
        window = np.stack(hand_features[start:start + window_size], axis=0)
        samples.append((window, hand, label))
    return samples
