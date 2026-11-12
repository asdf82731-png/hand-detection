"""
Unit tests for the hand-mapping check. Training on a swapped mapping gives
every sample the other hand's label with no error, so the check that picks
the mapping must point the right way and must refuse to guess.
"""
import unittest

import numpy as np

from check_data import hand_speed, mapping_scores, recommend_csv_left_slot


def _labels_and_speeds(moving_slot_for_csv_left):
    """CSV left strokes in frames 10-19, CSV right in 30-39; the given slot moves with CSV left."""
    frame_labels = {}
    for f in range(50):
        frame_labels[f] = {
            "left": "full_stroke" if 10 <= f < 20 else "up_hold",
            "right": "down_stroke" if 30 <= f < 40 else "down_hold",
        }
    with_left = np.where((np.arange(50) >= 10) & (np.arange(50) < 20), 20.0, 1.0)
    with_right = np.where((np.arange(50) >= 30) & (np.arange(50) < 40), 20.0, 1.0)
    other = "right" if moving_slot_for_csv_left == "left" else "left"
    return frame_labels, {moving_slot_for_csv_left: with_left, other: with_right}


class TestHandMapping(unittest.TestCase):
    def test_csv_left_matching_slot_left_is_recognised(self):
        scores = mapping_scores(*_labels_and_speeds("left"))
        self.assertGreater(scores["same"], scores["swapped"])
        self.assertEqual(recommend_csv_left_slot(scores), "left")

    def test_csv_left_matching_slot_right_is_recognised(self):
        scores = mapping_scores(*_labels_and_speeds("right"))
        self.assertGreater(scores["swapped"], scores["same"])
        self.assertEqual(recommend_csv_left_slot(scores), "right")

    def test_no_recommendation_when_the_evidence_is_too_close(self):
        # Both hands moving together in every stroke cannot tell the mappings
        # apart; guessing here would be a coin flip presented as a result.
        self.assertIsNone(recommend_csv_left_slot({"same": 10.0, "swapped": 9.5}))
        self.assertIsNone(recommend_csv_left_slot({"same": 0.0, "swapped": 0.0}))

    def test_frames_without_a_detection_do_not_count_as_motion(self):
        coords = np.zeros((4, 21, 3), dtype=np.float32)
        coords[1] = np.nan
        coords[3, :, 0] = 5.0
        speed = hand_speed(coords)
        self.assertTrue(np.isnan(speed[0]) and np.isnan(speed[1]) and np.isnan(speed[3]))
        self.assertEqual(speed.shape, (4,))


if __name__ == "__main__":
    unittest.main()
