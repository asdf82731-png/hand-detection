"""
Pre-labeling: the machine labels first, a person corrects.

    python autolabel.py label --source data/Full --csv-left-slot right --out Full_auto.csv
    python autolabel.py check --auto Full_auto.csv --human data/Full.csv

`label` writes a CSV in the same format as the hand-made label files
(frame_id, hand, strike), one row per frame and hand, from rules built on the
label definitions in TRAINING_DESIGN.md section 6 (thesis 3.5.1, table 4.2):

- A hand that is not moving is in a hold: up_hold or down_hold depending on
  whether it is high or low.
- A downswing (wrist moving down fast enough, far enough, and stopping
  abruptly, as a stick does at the drum) is the hit of a stroke. Lowering the
  hand slows down gradually and is not a hit. The stroke runs from where the hand starts to move to where it
  stops, or to the top of the hand's path between two hits when strokes
  follow without a pause.
- The stroke type comes from how high the hand rises before the hit and
  where it ends (TRAINING_DESIGN.md 5.2):
      high before, ends high -> full_stroke    high before, ends low -> down_stroke
      low before,  ends high -> up_stroke      low before,  ends low -> tab_stroke
- Movement that contains no hit (for example returning to the low position
  after an up stroke) is none.
- Frames where the hand was not detected get no row, so they are left out of
  training instead of being guessed.

High and low use both the wrist height and the wrist angle (the user's
decision in TRAINING_DESIGN.md 6.4): the two are standardized within the video
and averaged, with the angle's sign chosen so that it rises with the height.
The split between high and low is the midpoint between the two groups of
still frames found in that video. A video whose holds are all at one height
(the thesis's Full, Down and Tab clips) has no second group; then a hand
counts as high when the wrist is HIGH_HEIGHT palm lengths above its lowest
point in the video.

The thresholds are first guesses, not tuned. Run `check` against hand-made
labels before trusting the output, and always have a person review it: a model
trained on unreviewed rule output learns the rules, mistakes included, and a
test set labeled by the rules measures nothing.

`check` compares a pre-labeled CSV with a hand-made one: frame agreement and
Cohen's kappa per hand, and per stroke type the counts and event precision /
recall / F1 with the same ±3-frame tolerance as evaluate.py.
"""
import argparse
import csv
import os

import numpy as np

from data_pipeline import LABELS, load_frame_labels, parse_frame_id
from evaluate import labeled_strokes, score
from features import _IMAGE_EXTENSIONS, extract_raw_tracks, hand_state, video_fps

DEFAULT_FPS = 30  # frame folders carry no frame rate; the thesis footage is 30fps

# In palm lengths and seconds. HIT_SPEED to STOP_FRAMES were swept with `check`
# against the Full and Tab hand labels (ROADMAP.md, 2026-11-13); the rest are first guesses.
SMOOTH_FRAMES = 3          # moving-average width for the height and angle traces
MOVE_SPEED = 1.5           # palm lengths / s: slower than this counts as still
HIT_SPEED = 2.0            # palm lengths / s: a downswing must reach this speed...
HIT_DROP = 0.2             # ...and fall at least this far, in palm lengths...
STOP_SHARE = 0.5           # ...and slow from its top speed to this share of it...
STOP_FRAMES = 5            # ...within this many frames (an abrupt stop, as at the drum)
MERGE_GAP_FRAMES = 2       # still gaps this short do not end a movement
SEPARATION = 1.0           # high and low still frames must be this far apart (standardized score)...
MIN_GROUP_SHARE = 0.1      # ...and each be at least this share of still frames, to split within the video
HIGH_HEIGHT = 1.0          # otherwise: high = wrist this many palm lengths above its lowest point


def _smooth(values, width=SMOOTH_FRAMES):
    if width <= 1 or len(values) < width:
        return np.asarray(values, dtype=np.float64)
    padded = np.pad(np.asarray(values, dtype=np.float64), (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _runs(mask):
    """[(first, last)] of consecutive True frames."""
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def level_score(states):
    """
    (N, 2) wrist height and angle -> (N,) one number per frame, higher = hand
    higher: both standardized within the video and averaged, the angle flipped
    if it falls as the height rises.
    """
    height, angle = _smooth(states[:, 0]), _smooth(states[:, 1])

    def z(values):
        std = values.std()
        return (values - values.mean()) / std if std > 1e-9 else np.zeros_like(values)

    zh, za = z(height), z(angle)
    sign = 1.0 if np.dot(zh, za) >= 0 else -1.0
    return (zh + sign * za) / 2


def two_means(values, iterations=50):
    """1-D two-means: (low group mean, high group mean, high-group share)."""
    low, high = float(values.min()), float(values.max())
    for _ in range(iterations):
        split = (low + high) / 2
        lower, upper = values[values <= split], values[values > split]
        if not len(lower) or not len(upper):
            break
        low, high = float(lower.mean()), float(upper.mean())
    share = float((values > (low + high) / 2).mean())
    return low, high, share


def high_mask(states, still):
    """
    (N, 2) wrist states and (N,) still mask -> (N,) True where the hand is
    high. Uses the height-and-angle score split between the two groups of
    still frames when the video has both; otherwise the wrist height alone
    against HIGH_HEIGHT (see the module docstring).
    """
    scores = level_score(states)
    values = scores[still]
    if len(values) >= 4:
        low, high, share = two_means(values)
        if high - low >= SEPARATION and MIN_GROUP_SHARE <= share <= 1 - MIN_GROUP_SHARE:
            return scores > (low + high) / 2
    return _smooth(states[:, 0]) > HIGH_HEIGHT


def _stroke_type(rises_high, ends_high):
    return {(True, True): "full_stroke", (True, False): "down_stroke",
            (False, True): "up_stroke", (False, False): "tab_stroke"}[(rises_high, ends_high)]


def label_hand(states, detected, fps):
    """
    One hand's (N, 2) wrist states and (N,) detected mask -> list of N labels
    (None where the hand was never seen or for frames before it was first
    seen). See the module docstring for the rules.
    """
    n = len(states)
    labels = [None] * n
    if not detected.any():
        return labels
    first_seen = int(np.flatnonzero(detected)[0])

    height = _smooth(states[:, 0])
    velocity = np.gradient(height) * fps  # palm lengths / s, positive = up
    moving = np.abs(velocity) > MOVE_SPEED
    for first, last in _runs(~moving):  # short still gaps inside a movement do not split it
        if 0 < first and last < n - 1 and last - first + 1 <= MERGE_GAP_FRAMES:
            moving[first:last + 1] = True
    is_high = high_mask(states, ~moving & detected)

    # Hits: the last frame of each fast downswing that fell far enough. (Not
    # the frame after: when the hand stops at the drum, that frame is already
    # still and outside the movement.)
    hits = []
    for first, last in _runs(velocity < -HIT_SPEED):
        top, bottom = max(first - 1, 0), min(last + 1, n - 1)
        peak = first + int(np.argmin(velocity[first:last + 1]))
        slowed = np.flatnonzero(velocity[peak:] >= STOP_SHARE * velocity[peak])
        abrupt = len(slowed) > 0 and slowed[0] <= STOP_FRAMES
        if height[top] - height[bottom] >= HIT_DROP and abrupt:
            hits.append(last)

    for i in range(n):
        labels[i] = None if i < first_seen else ("up_hold" if is_high[i] else "down_hold")
    for first, last in _runs(moving):
        segment_hits = [h for h in hits if first <= h <= last]
        if not segment_hits:
            for i in range(max(first, first_seen), last + 1):
                labels[i] = "none"
            continue
        # Between two hits, the stroke boundary is the top of the hand's path.
        bounds = [first]
        for a, b in zip(segment_hits, segment_hits[1:]):
            bounds.append(a + int(np.argmax(height[a:b + 1])))
        bounds.append(last)
        for k, hit in enumerate(segment_hits):
            start, end = bounds[k], bounds[k + 1]
            rises_high = bool(is_high[start:hit + 1].any())
            ends_high = bool(is_high[end]) if k + 1 < len(segment_hits) else bool(is_high[min(end + 1, n - 1)])
            stroke = _stroke_type(rises_high, ends_high)
            for i in range(max(start, first_seen), end + 1):
                labels[i] = stroke
    return labels


def frame_ids(source, num_frames):
    """frame_id for each frame index: the file names for a folder, frame_XXXX.jpg for a video."""
    if os.path.isdir(source):
        names = {parse_frame_id(n): n for n in os.listdir(source) if n.lower().endswith(_IMAGE_EXTENSIONS)}
        return [names.get(i) for i in range(num_frames)]
    return [f"frame_{i:04d}.jpg" for i in range(num_frames)]


def label_source(source, csv_left_slot, with_pose=False):
    """
    Video or frame folder -> [(frame_id, csv hand, label)] rows; csv_left_slot
    as in train.py. with_pose measures the wrist angle against the forearm.
    """
    raw, elbows = extract_raw_tracks(source, with_pose=with_pose)
    fps = video_fps(source) or DEFAULT_FPS
    num_frames = len(raw["left"])
    ids = frame_ids(source, num_frames)
    slot_for = {"left": csv_left_slot, "right": "right" if csv_left_slot == "left" else "left"}
    per_hand = {}
    for csv_hand, slot in slot_for.items():
        coords = raw[slot]
        detected = ~np.isnan(coords).any(axis=(1, 2))
        per_hand[csv_hand] = label_hand(hand_state(coords, elbows[slot] if with_pose else None), detected, fps)
    rows = []
    for i in range(num_frames):
        for csv_hand in ("left", "right"):
            label = per_hand[csv_hand][i]
            if label is not None and ids[i] is not None:
                rows.append((ids[i], csv_hand, label))
    return rows


def cohen_kappa(a, b, labels=LABELS):
    """Agreement beyond chance between two equal-length label lists; None if undefined."""
    if not a:
        return None
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        matrix[index[x], index[y]] += 1
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / total ** 2
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def compare(auto_labels, human_labels):
    """
    Two {frame: {hand: label}} dicts -> {hand: {"frames", "agreement", "kappa", "strokes": score(...)}},
    comparing only frames both have a label for.
    """
    results = {}
    for hand in ("left", "right"):
        common = sorted(f for f in human_labels if hand in human_labels[f] and hand in auto_labels.get(f, {}))
        a = [auto_labels[f][hand] for f in common]
        h = [human_labels[f][hand] for f in common]
        agreement = sum(x == y for x, y in zip(a, h)) / len(common) if common else None
        both = set(common)
        auto_in = {f: auto_labels[f] for f in both}
        human_in = {f: human_labels[f] for f in both}
        results[hand] = {
            "frames": len(common),
            "agreement": agreement,
            "kappa": cohen_kappa(a, h),
            "strokes": score(labeled_strokes(auto_in, hand), labeled_strokes(human_in, hand)),
        }
    return results


def _fmt(value):
    return "  -  " if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="pre-label a video or frame folder")
    label.add_argument("--source", required=True, help="video file or frame-image folder")
    label.add_argument("--csv-left-slot", required=True, choices=["left", "right"],
                       help="which detected hand the CSV's 'left' means, as for train.py")
    label.add_argument("--out", required=True)
    label.add_argument("--pose", action="store_true",
                       help="measure the wrist angle against the forearm with the body model (elbows must be in frame)")
    check = commands.add_parser("check", help="compare a pre-labeled CSV with a hand-made one")
    check.add_argument("--auto", required=True)
    check.add_argument("--human", required=True)
    args = parser.parse_args()

    if args.command == "label":
        rows = label_source(args.source, args.csv_left_slot, args.pose)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "hand", "strike"])
            writer.writerows(rows)
        counts = {}
        for _, hand, value in rows:
            counts.setdefault(hand, {}).setdefault(value, 0)
            counts[hand][value] += 1
        print(f"wrote {len(rows)} rows to {args.out} (frames per label: {counts})")
        print("These are machine guesses: review and correct them before using them for training.")
        return

    results = compare(load_frame_labels(args.auto), load_frame_labels(args.human))
    for hand, r in results.items():
        print(f"\n== {hand} hand: {r['frames']} frames labeled in both, "
              f"agreement {_fmt(r['agreement'])}, kappa {_fmt(r['kappa'])}")
        print(f"{'stroke':6s} {'human':>6s} {'auto':>5s} {'error':>6s} {'prec':>5s} {'recall':>6s} {'f1':>5s}")
        for stroke, s in r["strokes"].items():
            if s["labeled"] or s["detected"]:
                print(f"{stroke:6s} {s['labeled']:6d} {s['detected']:5d} {s['count_error']:+6d} "
                      f"{_fmt(s['precision']):>5s} {_fmt(s['recall']):>6s} {_fmt(s['f1']):>5s}")


if __name__ == "__main__":
    main()
