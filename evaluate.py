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

Reporting rules (REV-7, see TRAINING_DESIGN.md "評估規範"):
  * The evaluation videos must not be training videos: a label CSV whose
    content matches one the checkpoint was trained on is refused.
  * Pass one checkpoint per training seed (at least 5); the summary gives
    each seed's macro-F1 and their mean and standard deviation.
  * Totals pool all videos. Macro-F1 averages event F1 over the stroke types
    that have labeled strokes, each listed with its support (labeled count).
  * 95% intervals come from a block bootstrap: blocks of BLOCK_FRAMES frames
    per video and hand are resampled, because neighbouring strokes are not
    independent observations.
"""
import argparse
import hashlib
import json
import os
import statistics

import numpy as np

import torch

from data_pipeline import LABEL_TO_INDEX, load_frame_labels
from inference import NUM_STROKES, analyze, load_checkpoint
from pipeline import LABELS

# A detected stroke counts as the labeled one if their frame ranges overlap,
# allowing this many frames either side (TRAINING_DESIGN.md 6.4).
BOUNDARY_TOLERANCE = 3

# Block bootstrap: block length (about 10 s at 30 fps), resamples, interval.
BLOCK_FRAMES = 300
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
# Reporting minimums: training seeds per configuration, and labeled strokes a
# stroke type needs in the test set before a number for it is claimed.
MIN_SEEDS = 5
MIN_SUPPORT = 20


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
    matched = [0] * NUM_STROKES
    for (label, _, _), hit in zip(detected, matched_flags(detected, labeled, tolerance)):
        matched[label] += hit
    return matched


def matched_flags(detected, labeled, tolerance=BOUNDARY_TOLERANCE):
    """For each detected stroke, whether match_events pairs it with a labeled one."""
    used, flags = set(), []
    for label, first, last in detected:
        hit = False
        for i, (true_label, true_first, true_last) in enumerate(labeled):
            if i in used or true_label != label:
                continue
            if first <= true_last + tolerance and last >= true_first - tolerance:
                used.add(i)
                hit = True
                break
        flags.append(hit)
    return flags


def _prf(n_true, n_detected, n_matched):
    precision = n_matched / n_detected if n_detected else None
    recall = n_matched / n_true if n_true else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score(detected, labeled, tolerance=BOUNDARY_TOLERANCE):
    """Per stroke type name: labeled, detected, count_error, precision, recall, f1 (None when undefined)."""
    matched = match_events(detected, labeled, tolerance)
    result = {}
    for index, name in enumerate(LABELS):
        n_true = sum(1 for label, _, _ in labeled if label == index)
        n_detected = sum(1 for label, _, _ in detected if label == index)
        precision, recall, f1 = _prf(n_true, n_detected, matched[index])
        result[name] = {
            "labeled": n_true, "detected": n_detected, "count_error": n_detected - n_true,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return result


def block_counts(detected, labeled, tolerance=BOUNDARY_TOLERANCE, block_frames=BLOCK_FRAMES):
    """
    One video and hand -> {block number: array (NUM_STROKES, 3) of labeled,
    detected, matched counts}. A stroke belongs to the block its first frame
    falls in; matching is done on the whole video first, so a stroke near a
    block edge is matched exactly as score() matches it.
    """
    blocks = {}
    def add(block, label, column):
        blocks.setdefault(block, np.zeros((NUM_STROKES, 3), dtype=np.int64))[label, column] += 1
    for label, first, _ in labeled:
        add(first // block_frames, label, 0)
    for (label, first, _), hit in zip(detected, matched_flags(detected, labeled, tolerance)):
        add(first // block_frames, label, 1)
        if hit:
            add(first // block_frames, label, 2)
    return blocks


def macro_f1(counts):
    """Mean event F1 over stroke types with labeled strokes; counts is (NUM_STROKES, 3). None if none."""
    f1s = [_prf(*counts[i])[2] or 0.0 for i in range(NUM_STROKES) if counts[i, 0]]
    return sum(f1s) / len(f1s) if f1s else None


def summarize(blocks, samples=BOOTSTRAP_SAMPLES, confidence=CONFIDENCE, seed=0):
    """
    Pooled result over a list of block count arrays (from block_counts, all
    videos and hands): per stroke type support, detected, count error,
    precision, recall, f1 and its interval, plus macro-F1 and total absolute
    count error with intervals. Intervals resample whole blocks.
    """
    stack = np.stack(blocks)
    total = stack.sum(axis=0)
    rng = np.random.default_rng(seed)
    resampled = stack[rng.integers(0, len(stack), size=(samples, len(stack)))].sum(axis=1)
    alpha = (1 - confidence) / 2

    def interval(values):
        values = [v for v in values if v is not None]
        return [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))] if values else None

    per_type = {}
    for i, name in enumerate(LABELS):
        precision, recall, f1 = _prf(*total[i])
        per_type[name] = {
            "support": int(total[i, 0]), "detected": int(total[i, 1]), "count_error": int(total[i, 1] - total[i, 0]),
            "precision": precision, "recall": recall, "f1": f1,
            "f1_ci": interval([_prf(*r[i])[2] for r in resampled]) if total[i, 0] else None,
            "enough_support": bool(total[i, 0] >= MIN_SUPPORT),
        }
    abs_error = lambda c: int(np.abs(c[:, 1] - c[:, 0]).sum())
    return {
        "per_type": per_type,
        "macro_f1": macro_f1(total), "macro_f1_ci": interval([macro_f1(r) for r in resampled]),
        "abs_count_error": abs_error(total), "abs_count_error_ci": interval([abs_error(r) for r in resampled]),
        "blocks": len(blocks),
    }


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_independent(checkpoint_path, csv_paths):
    """
    Refuse evaluation CSVs that the checkpoint was trained on (same content).
    Checkpoints saved before train.py recorded this cannot be checked: warn.
    """
    trained = torch.load(checkpoint_path, map_location="cpu").get("config", {}).get("train_csv_sha256")
    if trained is None:
        print(f"[WARNING] {checkpoint_path} does not record its training CSVs; "
              "make sure the evaluation videos were not used for training.")
        return
    leaked = [p for p in csv_paths if file_sha256(p) in set(trained)]
    if leaked:
        raise SystemExit(f"{checkpoint_path} was trained on {', '.join(leaked)}; evaluate on videos it has not seen.")


def evaluate_source(source, csv_path, model, scaler, hand_to_slot, device):
    """
    One labeled video -> ({hand: score(...)}, [block count arrays]), comparing
    only within the labeled frame range.
    """
    frame_labels = load_frame_labels(csv_path)
    first_labeled, last_labeled = min(frame_labels), max(frame_labels)
    analysis = analyze(source, model, scaler, hand_to_slot, device)
    scale = analysis["frame_scale"]
    results, blocks = {}, []
    for hand in hand_to_slot:
        detected = [(label, round(first * scale), round(last * scale)) for label, first, last in analysis["spans"][hand]]
        detected = [d for d in detected if d[2] >= first_labeled and d[1] <= last_labeled]
        labeled = labeled_strokes(frame_labels, hand)
        results[hand] = score(detected, labeled)
        blocks.extend(block_counts(detected, labeled).values())
    return results, blocks


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


def _ci(interval, digits=2):
    return "  -  " if interval is None else f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def evaluate_checkpoint(checkpoint, sources, device):
    """Per-video tables printed, and {"videos": ..., "summary": summarize(...)} returned."""
    check_independent(checkpoint, [csv_path for _, _, csv_path in sources])
    model, scaler, hand_to_slot = load_checkpoint(checkpoint, device)
    videos, blocks = {}, []
    for name, source, csv_path in sources:
        videos[name], video_blocks = evaluate_source(source, csv_path, model, scaler, hand_to_slot, device)
        blocks.extend(video_blocks)
        print(f"\n== {checkpoint} | {name}")
        print(f"{'hand':6s} {'stroke':6s} {'labeled':>7s} {'detected':>8s} {'error':>6s} {'prec':>5s} {'recall':>6s} {'f1':>5s}")
        for hand, per_type in videos[name].items():
            for stroke, s in per_type.items():
                if s["labeled"] or s["detected"]:
                    print(f"{hand:6s} {stroke:6s} {s['labeled']:7d} {s['detected']:8d} {s['count_error']:+6d} "
                          f"{_fmt(s['precision']):>5s} {_fmt(s['recall']):>6s} {_fmt(s['f1']):>5s}")
    if not blocks:
        raise SystemExit("No labeled or detected strokes in the evaluation videos.")
    summary = summarize(blocks)
    print(f"\n== {checkpoint} | all videos ({summary['blocks']} bootstrap blocks)")
    print(f"{'stroke':6s} {'support':>7s} {'detected':>8s} {'error':>6s} {'f1':>5s}  95% CI")
    for stroke, s in summary["per_type"].items():
        if s["support"] or s["detected"]:
            note = "" if s["enough_support"] else f"  (fewer than {MIN_SUPPORT} labeled: do not claim)"
            print(f"{stroke:6s} {s['support']:7d} {s['detected']:8d} {s['count_error']:+6d} {_fmt(s['f1']):>5s}  {_ci(s['f1_ci'])}{note}")
    print(f"macro-F1 {_fmt(summary['macro_f1'])} {_ci(summary['macro_f1_ci'])}   "
          f"total |count error| {summary['abs_count_error']} {_ci(summary['abs_count_error_ci'], 0)}")
    return {"videos": videos, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", nargs="+", default=["multi_motion_model_7class.pt"],
                        help=f"one checkpoint per training seed; report needs at least {MIN_SEEDS}")
    parser.add_argument("--json", help="also write the full results to this file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = find_sources(args.data_dir)
    if not sources:
        raise SystemExit(f"No labeled videos found in {args.data_dir}")
    all_results = {ckpt: evaluate_checkpoint(ckpt, sources, device) for ckpt in args.checkpoint}

    scores = [r["summary"]["macro_f1"] for r in all_results.values() if r["summary"]["macro_f1"] is not None]
    print("\n== seeds")
    for ckpt, r in all_results.items():
        print(f"{ckpt}: macro-F1 {_fmt(r['summary']['macro_f1'])}")
    if len(scores) >= 2:
        print(f"mean {statistics.mean(scores):.3f}  sd {statistics.stdev(scores):.3f}  (n={len(scores)})")
    if len(args.checkpoint) < MIN_SEEDS:
        print(f"[WARNING] {len(args.checkpoint)} checkpoint(s); report numbers only from at least {MIN_SEEDS} seeds.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
