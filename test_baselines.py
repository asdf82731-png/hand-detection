"""
Tests for baselines.py (REV-6) and detection_by_label.py (REV-9). A baseline
scored on different windows from the model, or a detection rate that mixes
stroke and hold frames, would give a comparison that looks valid and is not.
"""
import unittest

import numpy as np
import torch

from baselines import GRUBaseline, majority_predictions, rule_window_predictions
from data_pipeline import LABEL_TO_INDEX, LABELS, build_labeled_windows, split_train_val
from detection_by_label import detection_by_group, longest_run
from train import WINDOW_SIZE


def _labels(sequence, hand="left"):
    return {f: {hand: label} for f, label in enumerate(sequence)}


class TestMajority(unittest.TestCase):
    def test_predicts_the_most_common_training_label_everywhere(self):
        # The floor any model has to clear: holds outnumber strokes, so a model
        # that only predicts holds can look accurate.
        train = np.array([LABEL_TO_INDEX["down_hold"]] * 5 + [LABEL_TO_INDEX["full_stroke"]] * 2)
        self.assertEqual(majority_predictions(train, 3), [LABEL_TO_INDEX["down_hold"]] * 3)


class TestRuleWindows(unittest.TestCase):
    def setUp(self):
        self.human = _labels(["down_hold"] * 340 + ["full_stroke"] * 60)
        self.n = 400

    def test_scored_on_the_same_validation_windows_as_train_py(self):
        windows = build_labeled_windows(self.human, "left", self.n, WINDOW_SIZE, 1)
        _, val = split_train_val(windows, self.n, WINDOW_SIZE, first_frame=0)
        y_true, y_pred = rule_window_predictions(self.human, self.human, "left", self.n)
        self.assertTrue(val)
        self.assertEqual(y_true, [label for _, label in val])
        self.assertEqual(y_pred, y_true)  # rules that agree with the person score perfectly

    def test_the_last_frame_decides_and_missing_rule_labels_count_as_none(self):
        rule = _labels(["down_hold"] * self.n)
        del rule[self.n - 1]  # hand not detected in the very last frame
        _, y_pred = rule_window_predictions(rule, self.human, "left", self.n)
        self.assertEqual(y_pred[-1], LABEL_TO_INDEX["none"])
        self.assertEqual(set(y_pred[:-1]), {LABEL_TO_INDEX["down_hold"]})


class TestGRUBaseline(unittest.TestCase):
    def test_sees_only_raw_joint_positions(self):
        # It is the "no bones, no velocity, no graph" control: changing those
        # channels must not change its output.
        torch.manual_seed(0)
        model = GRUBaseline(len(LABELS)).eval()
        x = torch.randn(2, WINDOW_SIZE, 62, 9)
        y = x.clone()
        y[:, :, 42:] = 0.0      # bones
        y[:, :, :, 3:] = 0.0    # velocity and acceleration
        out = model(x)
        self.assertEqual(out.shape, (2, len(LABELS)))
        torch.testing.assert_close(out, model(y))


class TestDetectionByLabel(unittest.TestCase):
    def test_misses_during_strokes_are_reported_apart_from_holds(self):
        # The question is whether misses cluster in fast motion; an overall
        # rate of 75% would hide that every miss is in a stroke.
        labels = _labels(["down_hold"] * 4 + ["full_stroke"] * 4)
        detected = np.array([True] * 4 + [True, False, False, True])
        result = detection_by_group(labels, "left", detected)
        self.assertEqual(result["hold"], (4, 1.0, 0))
        self.assertEqual(result["stroke"], (4, 0.5, 2))
        self.assertEqual(result["none"], (0, None, 0))

    def test_longest_gap_does_not_join_misses_across_a_detected_or_unlabeled_frame(self):
        self.assertEqual(longest_run([1, 2, 3, 4, 5], {1, 2, 4, 5}), 2)
        self.assertEqual(longest_run([1, 2, 4, 5], {1, 2, 4, 5}), 2)

    def test_frames_past_the_end_of_the_video_count_as_missed(self):
        result = detection_by_group(_labels(["down_hold"] * 4), "left", np.array([True, True]))
        self.assertEqual(result["hold"], (4, 0.5, 2))


if __name__ == "__main__":
    unittest.main()
