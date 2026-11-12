"""
Check the training data before training, on the machine that holds it:

    python check_data.py --data-dir data

For each action it reports what the CSV contains, whether its frame numbers
fit the frames, how often each hand is detected, and which detected hand the
CSV's left/right rows describe (the value train.py needs for
--csv-left-slot). It also saves one preview image per action so the
orientation and the hand assignment can be checked by eye. No data is changed.
"""
import argparse
import collections
import os

import cv2
import numpy as np

from data_pipeline import LABELS, load_frame_labels
from features import HANDS, detection_rates, extract_raw_hand_coords, iter_frames
from train import VIDEO_NAMES, data_sources

STROKE_LABELS = set(LABELS[:4])

# The two mappings must differ by at least this fraction of the larger score
# before one is recommended.
MIN_RELATIVE_MARGIN = 0.25


def hand_speed(coords):
    """(N, 21, 3) raw coords with NaN gaps -> (N,) pixels the hand centre moves to the next frame (NaN if unknown)."""
    centre = coords[:, :, :2].mean(axis=1)
    speed = np.full(len(coords), np.nan)
    speed[:-1] = np.linalg.norm(centre[1:] - centre[:-1], axis=1)
    return speed


def mapping_scores(frame_labels, speeds):
    """
    How well each hand mapping explains the motion: a hand should move more
    during frames labeled as a stroke than during frames labeled hold / none.
    Returns {"same": score, "swapped": score}; "same" means CSV left = slot
    left. Higher is better.
    """
    def contrast(csv_hand, slot):
        speed = speeds[slot]
        stroke = [f for f, hands in frame_labels.items() if f < len(speed) and hands.get(csv_hand) in STROKE_LABELS]
        rest = [f for f, hands in frame_labels.items()
                if f < len(speed) and csv_hand in hands and hands[csv_hand] not in STROKE_LABELS]
        if not stroke or not rest or np.isnan(speed[stroke]).all() or np.isnan(speed[rest]).all():
            return 0.0
        return float(np.nanmean(speed[stroke]) - np.nanmean(speed[rest]))

    return {
        "same": contrast("left", "left") + contrast("right", "right"),
        "swapped": contrast("left", "right") + contrast("right", "left"),
    }


def recommend_csv_left_slot(total_scores):
    """'left', 'right', or None when the two mappings are too close to call."""
    same, swapped = total_scores["same"], total_scores["swapped"]
    largest = max(abs(same), abs(swapped))
    if largest == 0 or abs(same - swapped) < MIN_RELATIVE_MARGIN * largest:
        return None
    return "left" if same > swapped else "right"


def save_preview(source, raw, frame_labels, path):
    """One labeled frame with each detected hand boxed and named, to check orientation and left/right by eye."""
    both = [f for f in sorted(frame_labels) if f < len(raw["left"]) and not any(np.isnan(raw[h][f]).any() for h in HANDS)]
    if not both:
        return None
    target = both[len(both) // 2]
    for index, image in iter_frames(source):
        if index != target:
            continue
        scale_x, scale_y = image.shape[1] / 1280, image.shape[0] / 720  # raw coords are in the 1280x720 reference frame
        for hand in HANDS:
            xs, ys = raw[hand][target][:, 0] * scale_x, raw[hand][target][:, 1] * scale_y
            top_left = (int(xs.min()), int(ys.min()))
            cv2.rectangle(image, top_left, (int(xs.max()), int(ys.max())), (0, 255, 0), 2)
            cv2.putText(image, f"slot {hand}", (top_left[0], max(25, top_left[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        text = f"frame {target}  CSV left={frame_labels[target].get('left')}  CSV right={frame_labels[target].get('right')}"
        cv2.putText(image, text, (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(path, image)
        return target
    return None


def check_action(data_dir, name):
    """Prints the report for one action; returns its mapping scores, or None if it cannot be used."""
    source, csv_path = data_sources(data_dir, name)
    print(f"\n=== {name} ===")
    if not os.path.exists(csv_path):
        print(f"  MISSING label file: {csv_path}")
        return None
    if not os.path.exists(source):
        print(f"  MISSING frames: need a {name}/ folder of frame images or {name}.mp4 in {data_dir}")
        return None

    frame_labels = load_frame_labels(csv_path)
    print(f"  labels : {csv_path}: {len(frame_labels)} frames, numbered {min(frame_labels)} to {max(frame_labels)}")
    for hand in ("left", "right"):
        counts = collections.Counter(hands[hand] for hands in frame_labels.values() if hand in hands)
        print(f"           CSV {hand:5s}: {sum(counts.values())} frames labeled  {dict(counts.most_common())}")

    raw = extract_raw_hand_coords(source)
    num_frames = len(raw["left"])
    rates = detection_rates(raw)
    kind = "frame folder" if os.path.isdir(source) else "video"
    print(f"  frames : {source} ({kind}): frames 0 to {num_frames - 1}")
    print(f"           detected: slot left {rates['left']:.0%}, slot right {rates['right']:.0%}")

    if max(frame_labels) >= num_frames:
        print(f"  PROBLEM: the CSV labels frame {max(frame_labels)} but there are only frames 0 to {num_frames - 1}.")
        print("           These labels were not made on these frames; train.py will refuse this action.")
        return None
    print("  frame numbers fit: OK")

    scores = mapping_scores(frame_labels, {hand: hand_speed(raw[hand]) for hand in HANDS})
    print(f"  hand mapping evidence: CSV left = slot left -> {scores['same']:.2f},  CSV left = slot right -> {scores['swapped']:.2f}")

    preview_path = os.path.join(data_dir, f"check_preview_{name}.jpg")
    target = save_preview(source, raw, frame_labels, preview_path)
    if target is not None:
        print(f"  preview: {preview_path} (frame {target}) — the person should be upright, each box on a hand")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    total = {"same": 0.0, "swapped": 0.0}
    usable = []
    for name in VIDEO_NAMES:
        scores = check_action(args.data_dir, name)
        if scores is not None:
            usable.append(name)
            for key in total:
                total[key] += scores[key]

    print("\n=== Summary ===")
    print(f"  usable actions: {usable or 'none'}")
    if not usable:
        print("  Nothing to train on yet — fix the problems above.")
        return
    slot = recommend_csv_left_slot(total)
    print(f"  hand mapping totals: CSV left = slot left -> {total['same']:.2f},  CSV left = slot right -> {total['swapped']:.2f}")
    print("  Confirm with the check_preview_*.jpg images: is the hand in the 'slot left' box the one the CSV")
    print("  calls left? Then the answer is --csv-left-slot left; if the CSV calls that hand right, it is right.")
    if slot is None:
        print("  The motion evidence is too close to call, so the images decide.")
    else:
        print(f"  The motion evidence suggests: --csv-left-slot {slot}  (a suggestion; the images decide)")
    print(f"  Then:  python train.py --data-dir {args.data_dir} --csv-left-slot <left or right>")


if __name__ == "__main__":
    main()
