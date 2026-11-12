"""
Skeleton features following the thesis method (chapter 3, modules A1-A2):
pixel coordinates, interpolation of missing joints, wrist-relative
normalization, and per-window position / velocity / acceleration features for
joints and bones.

Shared by training (train.py) and inference (inference.py).
"""
import os

import cv2
import numpy as np

from data_pipeline import parse_frame_id
from pipeline import BONE_CONNECTIONS, _assign_hand_slots, create_hand_tracker, read_upright_frames

# x/y are scaled to pixel positions in this frame size (the training videos'
# own resolution) whatever the input resolution, so the same motion gives the
# same numbers for a 720p and a 1080p upload.
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720

HANDS = ("left", "right")
_SLOT_TO_HAND = {"Left": "left", "Right": "right"}


def detect_hand_coords(image, hands):
    """
    One frame -> {"left": (21, 3) or None, "right": (21, 3) or None}, keyed by
    MediaPipe's handedness slot. x/y in reference-frame pixels, z as MediaPipe
    reports it. `hands` is a tracker from pipeline.create_hand_tracker().
    """
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    coords = {"left": None, "right": None}
    if results.multi_hand_landmarks:
        slots = _assign_hand_slots(
            [h.classification[0].label for h in results.multi_handedness],
            [h.classification[0].score for h in results.multi_handedness],
            [np.mean([lm.x for lm in hl.landmark]) for hl in results.multi_hand_landmarks],
        )
        for hand_landmarks, slot in zip(results.multi_hand_landmarks, slots):
            if slot is None:
                continue
            coords[_SLOT_TO_HAND[slot]] = np.array(
                [[lm.x * REFERENCE_WIDTH, lm.y * REFERENCE_HEIGHT, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
    return coords


def fill_missing(coords):
    """
    coords: (N, 21, 3) with NaN rows where the hand was not detected.
    Frames after the first detection are filled by linear interpolation
    between the surrounding detections (held at the last detection once there
    is no later one); frames before the first detection have no history and
    are zero (thesis A13).
    """
    coords = np.asarray(coords, dtype=np.float32)
    detected = ~np.isnan(coords).any(axis=(1, 2))
    filled = np.zeros_like(coords)
    if not detected.any():
        return filled
    known = np.flatnonzero(detected)
    frames = np.arange(len(coords))
    flat = coords.reshape(len(coords), -1)
    out = np.stack([np.interp(frames, known, flat[known, k]) for k in range(flat.shape[1])], axis=1)
    out[: known[0]] = 0.0
    return out.reshape(coords.shape).astype(np.float32)


def wrist_relative(coords):
    """(N, 21, 3) -> same shape, every joint relative to that frame's wrist (joint 0)."""
    return coords - coords[:, 0:1, :]


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def iter_frames(source):
    """
    Yield (frame_index, image) in order from a video file or from a directory
    of frame images. For a directory the index is the number in the file name
    (frame_0026.jpg -> 26), which is what the label CSV's frame_id refers to.
    """
    if os.path.isdir(source):
        names = [n for n in os.listdir(source) if n.lower().endswith(_IMAGE_EXTENSIONS)]
        for index, name in sorted((parse_frame_id(n), n) for n in names):
            image = cv2.imread(os.path.join(source, name))
            if image is None:
                raise ValueError(f"Cannot read image: {os.path.join(source, name)}")
            yield index, image
    else:
        yield from enumerate(read_upright_frames(source))


def extract_raw_hand_coords(source):
    """
    Video file or frame-image directory -> {"left": (N, 21, 3), "right": ...}
    in reference-frame pixels, NaN where the hand was not detected. N is the
    highest frame index + 1.
    """
    detected = {hand: {} for hand in HANDS}
    last_index = -1
    with create_hand_tracker() as tracker:
        for index, image in iter_frames(source):
            last_index = index
            coords = detect_hand_coords(image, tracker)
            for hand in HANDS:
                if coords[hand] is not None:
                    detected[hand][index] = coords[hand]
    if last_index < 0:
        raise ValueError(f"No frames could be read from {source}")

    raw = {}
    for hand in HANDS:
        raw[hand] = np.full((last_index + 1, 21, 3), np.nan, dtype=np.float32)
        for index, coords in detected[hand].items():
            raw[hand][index] = coords
    return raw


def detection_rates(raw):
    """Fraction of frames in which each hand was detected."""
    return {hand: float((~np.isnan(coords).any(axis=(1, 2))).mean()) for hand, coords in raw.items()}


def extract_hand_positions(source):
    """
    Video file or frame-image directory ->
    ({"left": (N, 21, 3), "right": (N, 21, 3)}, detection rates). Positions
    are interpolated and wrist-relative, ready for window_features().
    """
    raw = extract_raw_hand_coords(source)
    positions = {hand: wrist_relative(fill_missing(coords)) for hand, coords in raw.items()}
    return positions, detection_rates(raw)


def _with_velocity_and_acceleration(values):
    """
    (T, K, 3) -> (T, K, 9): value, frame-to-frame velocity, acceleration.
    Differencing leaves one fewer velocity and two fewer accelerations than
    frames; the tail is zero-padded (thesis A23).
    """
    velocity = np.zeros_like(values)
    velocity[:-1] = values[1:] - values[:-1]
    acceleration = np.zeros_like(values)
    acceleration[:-2] = velocity[1:-1] - velocity[:-2]
    return np.concatenate([values, velocity, acceleration], axis=-1)


def window_features(positions):
    """
    One window of one hand's wrist-relative positions (T, 21, 3) ->
    (joint features (T, 21, 9), bone features (T, 20, 9)) (thesis A22).
    """
    bones = np.stack([positions[:, j2] - positions[:, j1] for j1, j2 in BONE_CONNECTIONS], axis=1)
    return _with_velocity_and_acceleration(positions), _with_velocity_and_acceleration(bones)


class FeatureScaler:
    """Per-channel standardization, fitted on training windows only."""

    def __init__(self, joint_mean, joint_std, bone_mean, bone_std):
        self.joint_mean, self.joint_std = joint_mean, joint_std
        self.bone_mean, self.bone_std = bone_mean, bone_std

    @classmethod
    def fit(cls, joint_windows, bone_windows):
        joints = np.concatenate([w.reshape(-1, 9) for w in joint_windows])
        bones = np.concatenate([w.reshape(-1, 9) for w in bone_windows])
        return cls(joints.mean(axis=0), joints.std(axis=0) + 1e-6, bones.mean(axis=0), bones.std(axis=0) + 1e-6)

    def transform(self, joint_window, bone_window):
        return (joint_window - self.joint_mean) / self.joint_std, (bone_window - self.bone_mean) / self.bone_std

    def state_dict(self):
        return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in state.items()})


def pack_model_input(joint_window, bone_window, hand):
    """
    (T, 21, 9) joints + (T, 20, 9) bones of one hand -> model input (T, 62, 9).
    Joints 0-20 are the left-hand slot, 21-41 the right-hand slot; the other
    hand's slot stays zero so the model sees one hand at a time.
    """
    packed = np.zeros((joint_window.shape[0], 62, 9), dtype=np.float32)
    offset = 0 if hand == "left" else 21
    packed[:, offset:offset + 21] = joint_window
    packed[:, 42:] = bone_window
    return packed
