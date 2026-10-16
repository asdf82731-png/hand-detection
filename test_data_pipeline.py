"""
Unit tests for data_pipeline.py, per TRAINING_DESIGN.md section 4. Each test
encodes a specific decision from the design doc so changing that decision
later should make the matching test fail loudly (CLAUDE.md Rule 9).
"""
import csv
import os
import tempfile
import unittest

import numpy as np

from data_pipeline import (
    LABEL_TO_INDEX,
    build_labeled_windows,
    load_frame_labels,
    normalize_label,
    split_train_val,
    windows_to_samples,
)


class TestNormalizeLabel(unittest.TestCase):
    def test_strike_suffix_maps_to_stroke(self):
        # The CSVs use "_strike", the paper's label space uses "_stroke" — these
        # must collapse to the same training label or the loss signal is wrong.
        self.assertEqual(normalize_label("full_strike"), "full_stroke")
        self.assertEqual(normalize_label("down_strike"), "down_stroke")
        self.assertEqual(normalize_label("up_strike"), "up_stroke")
        self.assertEqual(normalize_label("tab_strike"), "tab_stroke")

    def test_hold_and_none_pass_through_unchanged(self):
        self.assertEqual(normalize_label("down_hold"), "down_hold")
        self.assertEqual(normalize_label("up_hold"), "up_hold")
        self.assertEqual(normalize_label("none"), "none")

    def test_unknown_label_raises_instead_of_silently_dropping(self):
        # A silently-ignored typo would vanish into the "excluded frame" path
        # and look identical to a legitimate CSV gap — must fail loud instead.
        with self.assertRaises(ValueError):
            normalize_label("sideways_strike")


class TestFrameAlignment(unittest.TestCase):
    def _write_csv(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "hand", "stroke"])
            writer.writerows(rows)
        return path

    def test_frame_id_parsed_and_aligned_by_index(self):
        path = self._write_csv([
            ("frame_0010.jpg", "left", "full_strike"),
            ("frame_0010.jpg", "right", "down_strike"),
        ])
        try:
            frame_labels = load_frame_labels(path)
            self.assertEqual(frame_labels[10]["left"], "full_stroke")
            self.assertEqual(frame_labels[10]["right"], "down_stroke")
        finally:
            os.remove(path)

    def test_gap_in_csv_excludes_window_spanning_it(self):
        # Mirrors the real Tab.csv gap (frame_0115 -> frame_0127): the missing
        # frames must not be treated as "none" and any window needing them
        # must be dropped, not silently filled in.
        path = self._write_csv(
            [(f"frame_{i:04d}.jpg", "left", "full_strike") for i in range(0, 10)]
            + [(f"frame_{i:04d}.jpg", "left", "full_strike") for i in range(20, 40)]
        )
        try:
            frame_labels = load_frame_labels(path)
            windows = build_labeled_windows(frame_labels, "left", num_frames=40, window_size=10, step=5)
            starts = [start for start, _ in windows]
            # A window starting anywhere in [1..10] would need a frame from the
            # [10, 20) gap, so none of those starts may appear.
            for gap_start in range(1, 11):
                self.assertNotIn(gap_start, starts)
            # A window fully inside the second labeled block must survive.
            self.assertIn(20, starts)
        finally:
            os.remove(path)


class TestLeftRightNotSwapped(unittest.TestCase):
    def test_hand_specific_windows_use_only_that_hand_label(self):
        frame_labels = {}
        for i in range(40):
            frame_labels[i] = {"left": "down_stroke", "right": "up_stroke"}

        left_windows = build_labeled_windows(frame_labels, "left", num_frames=40, window_size=10, step=10)
        right_windows = build_labeled_windows(frame_labels, "right", num_frames=40, window_size=10, step=10)

        self.assertTrue(left_windows)
        self.assertTrue(right_windows)
        for _, label in left_windows:
            self.assertEqual(label, LABEL_TO_INDEX["down_stroke"])
        for _, label in right_windows:
            self.assertEqual(label, LABEL_TO_INDEX["up_stroke"])

    def test_windows_to_samples_tags_the_hand_it_was_built_for(self):
        hand_features = [np.full((62, 9), i, dtype=np.float32) for i in range(20)]
        windows = [(0, LABEL_TO_INDEX["full_stroke"])]
        samples = windows_to_samples(hand_features, "left", windows, window_size=10)
        window, hand, label = samples[0]
        self.assertEqual(hand, "left")
        self.assertEqual(label, LABEL_TO_INDEX["full_stroke"])
        self.assertEqual(window.shape, (10, 62, 9))


class TestWindowLabelingRule(unittest.TestCase):
    def test_window_label_is_last_frame_not_majority(self):
        # 8 frames of down_stroke followed by 2 frames of full_stroke: majority
        # vote would say down_stroke, but the design decision (1.4) is to use
        # the window's last frame, matching the LSTM's many-to-one readout.
        frame_labels = {}
        for i in range(8):
            frame_labels[i] = {"left": "down_stroke"}
        for i in range(8, 10):
            frame_labels[i] = {"left": "full_stroke"}

        windows = build_labeled_windows(frame_labels, "left", num_frames=10, window_size=10, step=10)
        self.assertEqual(len(windows), 1)
        _, label = windows[0]
        self.assertEqual(label, LABEL_TO_INDEX["full_stroke"])


class TestTrainValNoLeakage(unittest.TestCase):
    def test_no_frame_range_overlap_between_train_and_val(self):
        # Needs enough frames that the 20%-val tail still has room for whole
        # windows once the buffer zone around the split point is carved out.
        frame_labels = {i: {"left": "full_stroke"} for i in range(1000)}
        windows = build_labeled_windows(frame_labels, "left", num_frames=1000, window_size=30, step=15)
        train, val = split_train_val(windows, num_frames=1000, window_size=30, val_fraction=0.2)

        self.assertTrue(train)
        self.assertTrue(val)

        train_ranges = [(s, s + 30) for s, _ in train]
        val_ranges = [(s, s + 30) for s, _ in val]
        for t_start, t_end in train_ranges:
            for v_start, v_end in val_ranges:
                overlap = t_start < v_end and v_start < t_end
                self.assertFalse(
                    overlap,
                    f"train window [{t_start},{t_end}) overlaps val window [{v_start},{v_end})",
                )

    def test_buffer_zone_windows_are_dropped_from_both_sets(self):
        frame_labels = {i: {"left": "full_stroke"} for i in range(100)}
        windows = build_labeled_windows(frame_labels, "left", num_frames=100, window_size=30, step=15)
        train, val = split_train_val(windows, num_frames=100, window_size=30, val_fraction=0.2, buffer=30)

        split_point = round(100 * 0.8)  # 80
        for s, _ in train + val:
            e = s + 30
            # None of the surviving windows may overlap [split_point-30, split_point+30)
            self.assertFalse(e > split_point - 30 and s < split_point + 30)


if __name__ == "__main__":
    unittest.main()
