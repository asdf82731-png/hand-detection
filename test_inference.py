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

from features import extract_hand_positions, resample_to_fps, video_fps, window_features
from inference import MIN_RUN_WINDOWS, TRAIN_FPS, recognize, stroke_events, to_training_fps

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



def _moving_hand(num_frames, fps, speed_px_per_s=300.0):
    """A hand moving down at a constant speed, filmed at fps: (num_frames, 21, 3)."""
    times = np.arange(num_frames) / fps
    coords = np.zeros((num_frames, 21, 3), dtype=np.float32)
    coords[:, :, 1] = (speed_px_per_s * times)[:, None] + np.arange(21)[None, :]
    return coords


class TestFrameRate(unittest.TestCase):
    # The model was trained on 30fps footage and its velocity features are
    # frame-to-frame differences. A phone filming at 60fps must not look to
    # the model like a stroke played at half speed.

    def test_same_motion_at_60fps_gives_the_30fps_velocity_after_resampling(self):
        at_30 = _moving_hand(31, fps=30)
        at_60 = resample_to_fps(_moving_hand(61, fps=60), 60, TRAIN_FPS)
        self.assertEqual(at_60.shape, at_30.shape)
        joints_30, _ = window_features(at_30[:30])
        joints_60, _ = window_features(at_60[:30])
        np.testing.assert_allclose(joints_60[:-1, :, 3:6], joints_30[:-1, :, 3:6], atol=1e-3)

    def test_without_resampling_60fps_halves_the_velocity(self):
        # Pins the problem the resampling exists to fix.
        joints_30, _ = window_features(_moving_hand(30, fps=30))
        joints_60, _ = window_features(_moving_hand(30, fps=60))
        np.testing.assert_allclose(joints_60[:-1, :, 4], joints_30[:-1, :, 4] / 2, atol=1e-3)

    def test_training_frame_rate_footage_is_left_untouched(self):
        # The training clips are 29.97fps; resampling them would shift the
        # windows away from what the model saw in training.
        positions = {"left": _moving_hand(100, fps=29.97), "right": _moving_hand(100, fps=29.97)}
        self.assertIs(to_training_fps(positions, 29.97), positions)
        self.assertIs(to_training_fps(positions, None), positions)

    def test_frame_folders_have_no_frame_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(video_fps(folder))


class TestProgress(unittest.TestCase):
    def test_progress_rises_and_ends_at_one(self):
        # The page's progress bar shows these numbers, so they must not jump
        # backwards and must finish at 100%.
        with tempfile.TemporaryDirectory() as folder:
            blank = np.zeros((72, 128, 3), dtype=np.uint8)
            for number in range(31):
                cv2.imwrite(os.path.join(folder, f"frame_{number:04d}.jpg"), blank)
            seen = []
            events = recognize(folder, model=None, scaler=None, hand_to_slot={"left": "left", "right": "right"},
                               device=None, on_progress=seen.append)
        self.assertEqual(events, {"left": [], "right": []})  # no hands detected, model never runs
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 1.0)
        self.assertGreater(len(seen), 30)


if __name__ == "__main__":
    unittest.main()
