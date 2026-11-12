"""
Unit tests for the hand-slot assignment in pipeline.py. Left/right hand
identity feeds straight into per-hand training labels and per-hand counts,
so a detection landing in the wrong slot (or silently overwriting the other
hand) corrupts both without any error.
"""
import unittest

from pipeline import _assign_hand_slots


class TestAssignHandSlots(unittest.TestCase):
    def test_distinct_labels_are_trusted_as_is(self):
        # When MediaPipe disagrees with screen position but gave two different
        # labels, its labels win — position is only a tie-breaker.
        self.assertEqual(_assign_hand_slots(["Left", "Right"], [0.9, 0.9], [0.2, 0.6]), ["Left", "Right"])

    def test_single_detection_keeps_its_label(self):
        self.assertEqual(_assign_hand_slots(["Right"], [0.9], [0.6]), ["Right"])

    def test_same_label_on_two_separate_hands_is_split_by_screen_side(self):
        # Previously both went to the same slot and one hand overwrote the
        # other, leaving the second hand's features all zero for that frame.
        self.assertEqual(_assign_hand_slots(["Right", "Right"], [0.9, 0.8], [0.22, 0.61]), ["Right", "Left"])
        self.assertEqual(_assign_hand_slots(["Left", "Left"], [0.9, 0.8], [0.61, 0.22]), ["Left", "Right"])

    def test_same_label_on_overlapping_detections_keeps_only_the_best(self):
        # Two boxes on one physical hand must not be turned into a fake second
        # hand on the other side.
        self.assertEqual(_assign_hand_slots(["Right", "Right"], [0.6, 0.95], [0.22, 0.25]), [None, "Right"])


if __name__ == "__main__":
    unittest.main()
