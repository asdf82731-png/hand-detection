"""
Unit tests for turning per-frame predictions into stroke counts. The count is
the number the user sees, so the rules that decide "is this one stroke or
two" are pinned here.
"""
import os
import tempfile
import unittest

import cv2
import numpy as np

from features import extract_hand_positions
from inference import MIN_RUN_WINDOWS, stroke_events

FULL, DOWN, UP, TAB, DOWN_HOLD, UP_HOLD, NONE = range(7)


class TestStrokeEvents(unittest.TestCase):
    def test_consecutive_windows_of_one_stroke_count_once(self):
        # Step-1 windows mean one physical stroke spans many predictions.
        self.assertEqual(stroke_events([FULL] * 20), [FULL])

    def test_hold_between_two_identical_strokes_separates_them(self):
        # Without the hold class, two Full strokes in a row merged into one.
        labels = [FULL] * 10 + [UP_HOLD] * 10 + [FULL] * 10
        self.assertEqual(stroke_events(labels), [FULL, FULL])

    def test_holds_and_none_are_never_counted(self):
        labels = [UP_HOLD] * 10 + [NONE] * 10 + [DOWN_HOLD] * 10
        self.assertEqual(stroke_events(labels), [])

    def test_short_flicker_inside_a_stroke_does_not_split_it(self):
        flicker = [NONE] * (MIN_RUN_WINDOWS - 1)
        self.assertEqual(stroke_events([DOWN] * 10 + flicker + [DOWN] * 10), [DOWN])

    def test_short_flicker_of_a_stroke_is_not_counted(self):
        flicker = [TAB] * (MIN_RUN_WINDOWS - 1)
        self.assertEqual(stroke_events([DOWN_HOLD] * 10 + flicker + [DOWN_HOLD] * 10), [])

    def test_different_strokes_back_to_back_are_each_counted_in_order(self):
        labels = [DOWN] * 8 + [DOWN_HOLD] * 8 + [UP] * 8 + [UP_HOLD] * 8 + [FULL] * 8
        self.assertEqual(stroke_events(labels), [DOWN, UP, FULL])


class TestFrameFolderIndexing(unittest.TestCase):
    def test_frame_index_comes_from_the_file_name_not_the_file_order(self):
        # The label CSV refers to frame_0026.jpg as frame 26. If a folder is
        # missing files, counting by position would shift every later label.
        with tempfile.TemporaryDirectory() as folder:
            blank = np.zeros((72, 128, 3), dtype=np.uint8)
            for number in (2, 5):
                cv2.imwrite(os.path.join(folder, f"frame_{number:04d}.jpg"), blank)
            positions, detection_rate = extract_hand_positions(folder)
        self.assertEqual(positions["left"].shape, (6, 21, 3))
        self.assertEqual(detection_rate, {"left": 0.0, "right": 0.0})


if __name__ == "__main__":
    unittest.main()
