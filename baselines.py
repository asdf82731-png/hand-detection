"""
Baselines for the 7-class window task (REV-6, PEER_REVIEW.md B3): without
them the thesis model's validation numbers cannot say whether its
architecture is needed.

    python baselines.py --data-dir data --csv-left-slot left

Every baseline is scored on the same validation windows train.py uses (same
videos, same time split and buffer, label = last frame of the window), with
accuracy and macro-F1, so the numbers line up with train.py's output for the
same --seed:

- majority:  always the most common training label. Any model must beat this.
- rule:      autolabel.py's wrist-height rules, label of the window's last
             frame. "rule + type" gives every hit the clip's known stroke type
             (Full -> full_stroke, ...), as autolabel --stroke-type does; that
             is knowledge the models do not get, so it is an upper bound for
             the rules, not a fair comparison. Frames the rules leave unlabeled
             (hand not detected) count as "none".
- gru:       a one-layer GRU on the raw wrist-relative keypoints (positions
             only, no bones, no Lie group embedding, no graph), trained by the
             same loop as the thesis model.

Not covered: the audio-onset rule (the pipeline reads no audio) and ST-GCN.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from autolabel import label_source
from data_pipeline import LABEL_TO_INDEX, LABELS, build_labeled_windows, load_frame_labels, parse_frame_id, split_train_val
from features import FeatureScaler
from train import (
    VIDEO_NAMES, WINDOW_SIZE, csv_left_slot_spec, data_sources, fit, load_split_windows, macro_f1, set_seed, to_dataset,
)

STROKE_TYPE_OF_VIDEO = {"Full": "full_stroke", "Down": "down_stroke", "Up": "up_stroke", "Tab": "tab_stroke"}


def majority_predictions(train_labels, num_val):
    """The most common training label for every validation window."""
    return [int(np.bincount(train_labels, minlength=len(LABELS)).argmax())] * num_val


def rule_window_predictions(rule_labels, human_labels, csv_hand, num_frames):
    """
    One hand of one video -> (true, predicted) for its validation windows, the
    windows train.py builds; predicted is the rule's label for the window's
    last frame, "none" where the rules gave none.
    """
    windows = build_labeled_windows(human_labels, csv_hand, num_frames, WINDOW_SIZE, 1)
    _, val = split_train_val(windows, max(human_labels) + 1, WINDOW_SIZE, first_frame=min(human_labels))
    predicted = [LABEL_TO_INDEX[rule_labels.get(start + WINDOW_SIZE - 1, {}).get(csv_hand, "none")] for start, _ in val]
    return [label for _, label in val], predicted


def rule_baseline(data_dir, csv_left_slot, with_type):
    y_true, y_pred = [], []
    for name in VIDEO_NAMES:
        source, csv_path = data_sources(data_dir, name)
        if not (os.path.exists(source) and os.path.exists(csv_path)):
            continue  # skipped by load_split_windows too
        human = load_frame_labels(csv_path)
        default, overrides = csv_left_slot
        rows = label_source(source, overrides.get(name, default), stroke_type=STROKE_TYPE_OF_VIDEO[name] if with_type else None)
        rule = {}
        for frame_id, hand, label in rows:
            rule.setdefault(parse_frame_id(frame_id), {})[hand] = label
        num_frames = max(max(human), max(rule, default=0)) + 1
        for csv_hand in ("left", "right"):
            t, p = rule_window_predictions(rule, human, csv_hand, num_frames)
            y_true += t
            y_pred += p
    return y_true, y_pred


class GRUBaseline(nn.Module):
    """
    Raw keypoints -> GRU -> last hidden state -> class scores. Takes the same
    packed input (B, T, 62, 9) as the thesis model and keeps only the 42
    joint slots' positions (channels 0-2); the unused hand's slot is zero.
    """

    def __init__(self, num_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=42 * 3, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, T = x.shape[:2]
        _, h = self.gru(x[:, :, :42, :3].reshape(B, T, -1))
        return self.fc(h[-1])


def scores(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred))), macro_f1(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-left-slot", required=True, type=csv_left_slot_spec, help="as for train.py")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    train_windows, val_windows = load_split_windows(args.data_dir, args.csv_left_slot)
    train_labels = np.array([label for *_, label in train_windows])
    val_labels = [label for *_, label in val_windows]
    results = {"majority": scores(val_labels, majority_predictions(train_labels, len(val_labels)))}

    for with_type, row in ((False, "rule"), (True, "rule + type")):
        y_true, y_pred = rule_baseline(args.data_dir, args.csv_left_slot, with_type)
        if len(y_true) != len(val_labels):
            raise RuntimeError(f"{row}: {len(y_true)} validation windows, train.py has {len(val_labels)}")
        results[row] = scores(y_true, y_pred)

    scaler = FeatureScaler.fit([w[0] for w in train_windows], [w[1] for w in train_windows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUBaseline(len(LABELS))
    best_state, _ = fit(to_dataset(train_windows, scaler), to_dataset(val_windows, scaler), device, model=model)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        inputs = to_dataset(val_windows, scaler).tensors[0].to(device)
        results["gru"] = scores(val_labels, model(inputs).argmax(dim=1).tolist())

    print(f"\nvalidation windows: {len(val_labels)}")
    print(f"{'baseline':12s} {'accuracy':>8s} {'macro_f1':>8s}")
    for row, (accuracy, f1) in results.items():
        print(f"{row:12s} {accuracy:8.3f} {f1:8.3f}")
    print("Compare with train.py's best val accuracy / macro-F1 for the same --seed.")


if __name__ == "__main__":
    main()
