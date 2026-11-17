"""
Check a hand-labeled CSV before it is trained or evaluated on: does each hand's
sequence of labels make sense as drumming? A mistake here does not crash
anything, it just trains on the wrong answer, and a mistake found after
training means labeling that video again.

    python check_labels.py data/Full.csv data/Down.csv
    python check_labels.py my_session/*.csv

Two things are checked for each hand, and reported as warnings:

1. Stroke order. Which stroke may follow which is the thesis's rule
   (TRAINING_DESIGN.md 6.1, "之後可以接"): after full_stroke only Full or Down,
   after down_stroke Up or Tab, after up_stroke Full or Down, after tab_stroke
   Up or Tab. Holds between two strokes do not change that. A none between
   them does: after an up_stroke the hand comes back down, and that return is
   none (decision of 2026-11-13), so what follows is not compared.
2. Holds. A stroke ends high (full, up) or low (down, tab), and up_hold means
   the hand rests high, down_hold low. A hold that contradicts where the last
   stroke ended, or one that changes the resting place with no movement
   labeled in between, is what a wrong frame looks like.

What is not checked: where a stroke starts. The lift before a stroke counts as
part of the stroke (6.4, rule 4), so a down_hold before a down_stroke is right.
Frames with no row (the hand was not detected) are counted, not reported: the
guide says not to fill them in, and a gap forgets everything before it.

Exit status 1 if there is any warning, so a script can stop on it. A warning
is for a person to look at; a real unusual order is allowed to stay.
"""
import argparse
import sys

from data_pipeline import load_frame_labels

# Which stroke may come next (TRAINING_DESIGN.md 6.1), where a stroke ends and where a hold rests.
STROKE_NEXT = {
    "full_stroke": {"full_stroke", "down_stroke"}, "down_stroke": {"up_stroke", "tab_stroke"},
    "up_stroke": {"full_stroke", "down_stroke"}, "tab_stroke": {"up_stroke", "tab_stroke"},
}
STROKE_END = {"full_stroke": "high", "down_stroke": "low", "up_stroke": "high", "tab_stroke": "low"}
HOLD_AT = {"up_hold": "high", "down_hold": "low"}


def label_runs(frame_labels, hand):
    """
    One hand -> [(label, first frame, last frame)] in order, one entry per
    stretch of consecutive frames with the same label. A frame with no row
    ends the stretch, so a gap never joins two runs.
    """
    runs = []
    for frame in sorted(f for f, hands in frame_labels.items() if hand in hands):
        label = frame_labels[frame][hand]
        if runs and runs[-1][0] == label and runs[-1][2] == frame - 1:
            runs[-1][2] = frame
        else:
            runs.append([label, frame, frame])
    return [tuple(run) for run in runs]


def stroke_order_warnings(runs):
    """
    One hand's runs -> warnings about a stroke that may not follow the last
    one, or a hold that contradicts where the hand was. Everything before a
    none or a gap of frames with no row is forgotten.
    """
    warnings, previous_stroke, position, previous_last = [], None, None, None
    for label, first, last in runs:
        if previous_last is not None and first != previous_last + 1:
            previous_stroke = position = None  # frames with no row in between: the hand may have gone anywhere
        if label in STROKE_NEXT:
            if previous_stroke is not None and label not in STROKE_NEXT[previous_stroke]:
                allowed = " or ".join(sorted(STROKE_NEXT[previous_stroke]))
                warnings.append(f"frame {first}: {label} after {previous_stroke}, which may only be followed by {allowed}")
            previous_stroke, position = label, STROKE_END[label]
        elif label in HOLD_AT:
            if position is not None and position != HOLD_AT[label]:
                warnings.append(f"frame {first}: {label} (rests {HOLD_AT[label]}) but the hand was {position}")
            position = HOLD_AT[label]
        else:  # none: moving between places
            previous_stroke = position = None
        previous_last = last
    return warnings


def uncovered_frames(frame_labels, hand):
    """Frames between this hand's first and last labeled frame that have no row for it."""
    frames = sorted(f for f, hands in frame_labels.items() if hand in hands)
    if not frames:
        return []
    have = set(frames)
    return [f for f in range(frames[0], frames[-1] + 1) if f not in have]


def check_file(path):
    """One CSV -> (report lines, number of warnings). load_frame_labels refuses unknown labels and hands."""
    frame_labels = load_frame_labels(path)
    lines, total = [f"{path}: {len(frame_labels)} labeled frames"], 0
    for hand in ("left", "right"):
        runs = label_runs(frame_labels, hand)
        if not runs:
            lines.append(f"  {hand}: no labels")
            continue
        strokes = [run for run in runs if run[0] in STROKE_NEXT]
        counts = {name: sum(1 for run in strokes if run[0] == name) for name in STROKE_NEXT}
        missing = uncovered_frames(frame_labels, hand)
        warnings = stroke_order_warnings(runs)
        total += len(warnings)
        lines.append(f"  {hand}: {len(strokes)} strokes {counts}, {len(missing)} frames with no row")
        lines.extend(f"    WARNING {w}" for w in warnings)
    return lines, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", nargs="+", help="label CSV files (frame_id, hand, strike)")
    args = parser.parse_args()
    total = 0
    for path in args.csv:
        lines, warnings = check_file(path)
        print("\n".join(lines))
        total += warnings
    print(f"\n{total} warning(s)" + ("" if total == 0 else ": look at each frame, fix the label or leave it if the order is real"))
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
