"""
Event-level evaluation: does the app count each stroke type correctly?

    python evaluate.py --data-dir data_test --checkpoint multi_motion_model_7class.pt

For every video in --data-dir that has a label CSV, runs the checkpoint the
same way the app does and compares the detected strokes with the labeled
ones, per hand and per stroke type: labeled count, detected count, count
error, and event precision / recall / F1. This is the measure that matters
for the app (TRAINING_DESIGN.md 7.4); window accuracy from train.py is not.

It works the same for single-stroke videos and for mixed-stroke videos (the
thesis's second task, 4.5), as long as they are labeled in the same CSV
format. Data layout as in train.py, but any names: <name>.csv plus a frame
folder <name>/ or a video <name>.mp4. Use videos that were not used for
training, or the numbers say nothing about new videos.
"""
import argparse
import json
import os

import torch

from data_pipeline import LABEL_TO_INDEX, load_frame_labels
from inference import NUM_STROKES, analyze, load_checkpoint
from pipeline import LABELS

# A detected stroke counts as the labeled one if their frame ranges overlap,
# allowing this many frames either side (TRAINING_DESIGN.md 6.4).
BOUNDARY_TOLERANCE = 3


def labeled_strokes(frame_labels, hand):
    """
    One hand's labeled strokes: [(stroke index 0-3, first frame, last frame)],
    one per run of consecutive frames with the same stroke label. A missing
    frame or any other label ends the run.
    """
    strokes, current = [], None
    for frame in sorted(frame_labels):
        label = frame_labels[frame].get(hand)
        index = LABEL_TO_INDEX.get(label) if label is not None else None
        if index is not None and index < NUM_STROKES:
            if current and current[0] == index and current[2] == frame - 1:
                current[2] = frame
                continue
            current = [index, frame, frame]
            strokes.append(current)
        else:
            current = None
    return [tuple(stroke) for stroke in strokes]


def match_events(detected, labeled, tolerance=BOUNDARY_TOLERANCE):
    """
    Pair detected and labeled strokes one-to-one: same type and frame ranges
    overlapping within `tolerance`. Returns the number of matched pairs per
    stroke index.
    """
    used = set()
    matched = [0] * NUM_STROKES
    for label, first, last in detected:
        for i, (true_label, true_first, true_last) in enumerate(labeled):
            if i in used or true_label != label:
                continue
            if first <= true_last + tolerance and last >= true_first - tolerance:
                used.add(i)
                matched[label] += 1
                break
    return matched


def score(detected, labeled, tolerance=BOUNDARY_TOLERANCE):
    """Per stroke type name: labeled, detected, count_error, precision, recall, f1 (None when undefined)."""
    matched = match_events(detected, labeled, tolerance)
    result = {}
    for index, name in enumerate(LABELS):
        n_true = sum(1 for label, _, _ in labeled if label == index)
        n_detected = sum(1 for label, _, _ in detected if label == index)
        precision = matched[index] / n_detected if n_detected else None
        recall = matched[index] / n_true if n_true else None
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        result[name] = {
            "labeled": n_true, "detected": n_detected, "count_error": n_detected - n_true,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return result


def evaluate_source(source, csv_path, model, scaler, hand_to_slot, device):
    """One labeled video -> {hand: score(...)}, comparing only within the labeled frame range."""
    frame_labels = load_frame_labels(csv_path)
    first_labeled, last_labeled = min(frame_labels), max(frame_labels)
    analysis = analyze(source, model, scaler, hand_to_slot, device)
    scale = analysis["frame_scale"]
    results = {}
    for hand in hand_to_slot:
        detected = [(label, round(first * scale), round(last * scale)) for label, first, last in analysis["spans"][hand]]
        detected = [d for d in detected if d[2] >= first_labeled and d[1] <= last_labeled]
        results[hand] = score(detected, labeled_strokes(frame_labels, hand))
    return results


def find_sources(data_dir):
    """[(name, frames source, csv path)] for every <name>.csv with a matching folder or .mp4."""
    found = []
    for entry in sorted(os.listdir(data_dir)):
        if not entry.lower().endswith(".csv"):
            continue
        name = entry[:-4]
        folder, video = os.path.join(data_dir, name), os.path.join(data_dir, f"{name}.mp4")
        source = folder if os.path.isdir(folder) else video if os.path.exists(video) else None
        if source is None:
            print(f"[SKIPPED] {name}: no {name}/ frame folder or {name}.mp4 next to {entry}")
            continue
        found.append((name, source, os.path.join(data_dir, entry)))
    return found


def _fmt(value):
    return "  -  " if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="multi_motion_model_7class.pt")
    parser.add_argument("--json", help="also write the full results to this file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, hand_to_slot = load_checkpoint(args.checkpoint, device)
    all_results = {}
    for name, source, csv_path in find_sources(args.data_dir):
        all_results[name] = evaluate_source(source, csv_path, model, scaler, hand_to_slot, device)
        print(f"\n== {name}")
        print(f"{'hand':6s} {'stroke':6s} {'labeled':>7s} {'detected':>8s} {'error':>6s} {'prec':>5s} {'recall':>6s} {'f1':>5s}")
        for hand, per_type in all_results[name].items():
            for stroke, s in per_type.items():
                if s["labeled"] or s["detected"]:
                    print(f"{hand:6s} {stroke:6s} {s['labeled']:7d} {s['detected']:8d} {s['count_error']:+6d} "
                          f"{_fmt(s['precision']):>5s} {_fmt(s['recall']):>6s} {_fmt(s['f1']):>5s}")
    if not all_results:
        raise SystemExit(f"No labeled videos found in {args.data_dir}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
