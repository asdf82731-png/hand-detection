"""
Hand detection rate on stroke frames vs hold frames (REV-9, PEER_REVIEW.md
R1-8): fill_missing interpolates over undetected frames, which is harmless if
misses are random but rewrites the strokes themselves if misses cluster in
fast motion (motion blur). This measures whether they do.

    python detection_by_label.py --data-dir data --csv-left-slot left

For every video with a label CSV and each hand: labeled frames and the share
in which MediaPipe found that hand, grouped as stroke (the four *_stroke
labels), hold (down_hold, up_hold) and none, plus the longest run of
undetected frames inside each group (how long a gap the interpolation has to
bridge). A stroke rate clearly below the hold rate means the misses are not
random. No data is changed.
"""
import argparse
import os

import numpy as np

from data_pipeline import load_frame_labels
from features import extract_raw_hand_coords
from workflow.data_setup import VIDEO_NAMES, csv_left_slot_spec, data_sources, video_hand_mapping

GROUPS = {
    "stroke": {"full_stroke", "down_stroke", "up_stroke", "tab_stroke"},
    "hold": {"down_hold", "up_hold"},
    "none": {"none"},
}


def longest_run(frames, missed):
    """Longest run of consecutive frame numbers in `frames` that are all in `missed`."""
    best = current = 0
    previous = None
    for frame in sorted(frames):
        if frame in missed:
            current = current + 1 if previous is not None and frame == previous + 1 and previous in missed else 1
            best = max(best, current)
        previous = frame
    return best


def detection_by_group(frame_labels, csv_hand, detected):
    """
    One hand: {group: (labeled frames, detected share or None, longest undetected run)}.
    detected[i] is whether the hand was found in frame i; frames past its end count as missed.
    """
    missed = {f for f in frame_labels if not (f < len(detected) and detected[f])}
    result = {}
    for group, labels in GROUPS.items():
        frames = [f for f, hands in frame_labels.items() if hands.get(csv_hand) in labels]
        rate = 1 - sum(f in missed for f in frames) / len(frames) if frames else None
        result[group] = (len(frames), rate, longest_run(frames, missed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-left-slot", required=True, type=csv_left_slot_spec, help="as for train.py")
    args = parser.parse_args()

    print(f"{'video':6s} {'hand':5s} {'group':6s} {'frames':>6s} {'detected':>8s} {'longest gap':>11s}")
    for name in VIDEO_NAMES:
        source, csv_path = data_sources(args.data_dir, name)
        if not (os.path.exists(source) and os.path.exists(csv_path)):
            print(f"[SKIPPED] {name}: need {csv_path} and {source}")
            continue
        frame_labels = load_frame_labels(csv_path)
        raw = extract_raw_hand_coords(source)
        for csv_hand, slot in video_hand_mapping(args.csv_left_slot, name).items():
            detected = ~np.isnan(raw[slot]).any(axis=(1, 2))
            for group, (count, rate, gap) in detection_by_group(frame_labels, csv_hand, detected).items():
                shown = "-" if rate is None else f"{rate:.0%}"
                print(f"{name:6s} {csv_hand:5s} {group:6s} {count:6d} {shown:>8s} {gap:11d}")


if __name__ == "__main__":
    main()
