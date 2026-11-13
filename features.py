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
    labels, scores, hands_landmarks = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    coords = {"left": None, "right": None}
    if hands_landmarks:
        slots = _assign_hand_slots(labels, scores, [np.mean([lm.x for lm in hl]) for hl in hands_landmarks])
        for hand_landmarks, slot in zip(hands_landmarks, slots):
            if slot is None:
                continue
            coords[_SLOT_TO_HAND[slot]] = np.array(
                [[lm.x * REFERENCE_WIDTH, lm.y * REFERENCE_HEIGHT, lm.z] for lm in hand_landmarks],
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


def source_frame_count(source):
    """Number of frames a video file or frame-image directory holds (the video's own count may be approximate)."""
    if os.path.isdir(source):
        return sum(1 for n in os.listdir(source) if n.lower().endswith(_IMAGE_EXTENSIONS))
    cap = cv2.VideoCapture(source)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def video_fps(source):
    """Frame rate of a video file; None for a frame-image directory or when the file does not say."""
    if os.path.isdir(source):
        return None
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else None


def resample_to_fps(positions, fps, target_fps):
    """
    (N, 21, 3) positions sampled at fps -> the same motion sampled at
    target_fps, by linear interpolation in time. Velocity and acceleration are
    frame-to-frame differences, so without this a 60fps video gives half the
    velocity of the same stroke filmed at the 30fps the model was trained on.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) < 2:
        return positions
    duration = (len(positions) - 1) / fps
    target_times = np.arange(0.0, duration + 1e-9, 1.0 / target_fps)
    source_times = np.arange(len(positions)) / fps
    flat = positions.reshape(len(positions), -1)
    out = np.stack([np.interp(target_times, source_times, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(target_times),) + positions.shape[1:]).astype(np.float32)


def extract_raw_hand_coords(source, on_frame=None):
    """
    Video file or frame-image directory -> {"left": (N, 21, 3), "right": ...}
    in reference-frame pixels, NaN where the hand was not detected. N is the
    highest frame index + 1. on_frame(frames_done) is called after each frame.
    """
    detected = {hand: {} for hand in HANDS}
    last_index = -1
    with create_hand_tracker() as tracker:
        for done, (index, image) in enumerate(iter_frames(source), start=1):
            last_index = index
            if on_frame is not None:
                on_frame(done)
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


def extract_hand_positions(source, on_frame=None):
    """
    Video file or frame-image directory ->
    ({"left": (N, 21, 3), "right": (N, 21, 3)}, detection rates). Positions
    are interpolated and wrist-relative, ready for window_features().
    """
    positions, _, rates = extract_hand_tracks(source, on_frame)
    return positions, rates


def extract_hand_tracks(source, on_frame=None):
    """
    Like extract_hand_positions, plus each hand's wrist height and angle:
    returns (positions, hand_states {"left": (N, 2), "right": (N, 2)}, detection rates).
    """
    raw = extract_raw_hand_coords(source, on_frame)
    positions = {hand: wrist_relative(fill_missing(coords)) for hand, coords in raw.items()}
    states = {hand: hand_state(coords) for hand, coords in raw.items()}
    return positions, states, detection_rates(raw)


WRIST, MIDDLE_MCP = 0, 9


def hand_state(raw_coords):
    """
    One hand's raw coordinates (N, 21, 3), NaN where undetected -> (N, 2):
    wrist height and wrist angle per frame.

    The thesis defines the high and low positions by the wrist angle (table
    4.2), and the user decided both the height and the angle are used (see
    TRAINING_DESIGN.md 6.4). Both are measured from the raw coordinates,
    because wrist-relative positions have the hand's height removed.

    - Height: how far the wrist is above the lowest wrist position seen in
      this video, in palm lengths (wrist to middle-finger knuckle), so it does
      not depend on how far the camera is.
    - Angle: direction of the palm (wrist -> middle-finger knuckle) in the
      image, in radians, 0 = pointing straight up. MediaPipe has no forearm
      point, so this approximates the wrist angle by the palm's direction.

    Frames are interpolated like the positions; frames before the first
    detection are 0.
    """
    filled = fill_missing(raw_coords)
    detected = ~np.isnan(np.asarray(raw_coords, dtype=np.float32)).any(axis=(1, 2))
    states = np.zeros((len(filled), 2), dtype=np.float32)
    if not detected.any():
        return states
    palm = filled[:, MIDDLE_MCP, :2] - filled[:, WRIST, :2]
    palm_length = float(np.median(np.linalg.norm(palm[detected], axis=1))) or 1.0
    wrist_y = filled[:, WRIST, 1]
    lowest = float(wrist_y[detected].max())  # image y grows downward
    first = int(np.flatnonzero(detected)[0])
    states[first:, 0] = (lowest - wrist_y[first:]) / palm_length
    states[first:, 1] = np.unwrap(np.arctan2(palm[first:, 0], -palm[first:, 1]))
    return states


def resample_length(values, length):
    """Stretch or squeeze a sequence (T, ...) to `length` frames by linear interpolation in time."""
    values = np.asarray(values, dtype=np.float32)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(target, source, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((length,) + values.shape[1:]).astype(np.float32)


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


def window_state_features(states):
    """One window of one hand's (T, 2) wrist height and angle -> (T, 6): values, velocity, acceleration."""
    return _with_velocity_and_acceleration(states)


def window_features(positions):
    """
    One window of one hand's wrist-relative positions (T, 21, 3) ->
    (joint features (T, 21, 9), bone features (T, 20, 9)) (thesis A22).
    """
    bones = np.stack([positions[:, j2] - positions[:, j1] for j1, j2 in BONE_CONNECTIONS], axis=1)
    return _with_velocity_and_acceleration(positions), _with_velocity_and_acceleration(bones)


class FeatureScaler:
    """Per-channel standardization, fitted on training windows only."""

    def __init__(self, joint_mean, joint_std, bone_mean, bone_std, state_mean=None, state_std=None):
        self.joint_mean, self.joint_std = joint_mean, joint_std
        self.bone_mean, self.bone_std = bone_mean, bone_std
        self.state_mean, self.state_std = state_mean, state_std

    @classmethod
    def fit(cls, joint_windows, bone_windows, state_windows=None):
        joints = np.concatenate([w.reshape(-1, 9) for w in joint_windows])
        bones = np.concatenate([w.reshape(-1, 9) for w in bone_windows])
        state_mean = state_std = None
        if state_windows is not None:
            states = np.concatenate([w.reshape(-1, w.shape[-1]) for w in state_windows])
            state_mean, state_std = states.mean(axis=0), states.std(axis=0) + 1e-6
        return cls(joints.mean(axis=0), joints.std(axis=0) + 1e-6, bones.mean(axis=0), bones.std(axis=0) + 1e-6,
                   state_mean, state_std)

    def transform(self, joint_window, bone_window):
        return (joint_window - self.joint_mean) / self.joint_std, (bone_window - self.bone_mean) / self.bone_std

    def transform_state(self, state_window):
        return (state_window - self.state_mean) / self.state_std

    def state_dict(self):
        return {k: None if v is None else np.asarray(v, dtype=np.float32).tolist() for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state):
        # Checkpoints from before the hand-state option have no state_* keys.
        return cls(**{k: None if v is None else np.asarray(v, dtype=np.float32) for k, v in state.items()})


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
