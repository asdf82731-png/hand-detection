"""
Unit tests for pipeline.py: frame orientation and hand-slot assignment.
Left/right hand identity feeds straight into per-hand training labels and
per-hand counts, so an upside-down frame or a detection landing in the wrong
slot corrupts both without any error.
"""
import os
import unittest

import cv2
import numpy as np

from pipeline import _assign_hand_slots, mp_hands, read_upright_frames

TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def _bright_corner(frame):
    ys, xs = np.nonzero(frame.mean(axis=2) > 128)
    height, width = frame.shape[:2]
    return ("top" if ys.mean() < height / 2 else "bottom", "left" if xs.mean() < width / 2 else "right")


class TestReadUprightFrames(unittest.TestCase):
    # The fixtures are 64x48 clips with a white box in the top-left corner of
    # the stored frames plus a rotation tag, like phone recordings. The
    # training videos carry a 180-degree tag; read without it, every frame is
    # upside down and left/right swap sides.

    def test_180_degree_tag_is_applied(self):
        frames = list(read_upright_frames(os.path.join(TEST_DATA, "rotation_tag_180.mp4")))
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape[:2], (48, 64))
        self.assertEqual(_bright_corner(frames[0]), ("bottom", "right"))

    def test_90_degree_tag_turns_the_frame_to_portrait(self):
        # Expected values are what ffmpeg / a video player shows for this file.
        frames = list(read_upright_frames(os.path.join(TEST_DATA, "rotation_tag_90.mp4")))
        self.assertEqual(frames[0].shape[:2], (64, 48))
        self.assertEqual(_bright_corner(frames[0]), ("bottom", "left"))


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
        self.assertEqual(_assign_hand_slots(["Right", "Right"], [0.9, 0.8], [0.38, 0.77]), ["Left", "Right"])
        self.assertEqual(_assign_hand_slots(["Left", "Left"], [0.9, 0.8], [0.77, 0.38]), ["Right", "Left"])

    def test_same_label_on_overlapping_detections_keeps_only_the_best(self):
        # Two boxes on one physical hand must not be turned into a fake second
        # hand on the other side.
        self.assertEqual(_assign_hand_slots(["Right", "Right"], [0.6, 0.95], [0.22, 0.25]), [None, "Right"])


class TestHandednessConvention(unittest.TestCase):
    # The trained model and the hand slots were built on the labels the old
    # mp.solutions.hands API gave. MediaPipe's current API labels the same
    # hand the other way round; if that leaked through, left-hand and
    # right-hand features and counts would swap without any error.

    def test_label_matches_what_the_legacy_api_reported(self):
        # mediapipe 0.10.21 (mp.solutions.hands) labelled this photo's single hand "Left".
        image = cv2.cvtColor(cv2.imread(os.path.join(TEST_DATA, "one_hand.jpg")), cv2.COLOR_BGR2RGB)
        labels, _, hands_landmarks = mp_hands.process(image)
        self.assertEqual(labels, ["Left"])
        self.assertEqual(len(hands_landmarks[0]), 21)


if __name__ == "__main__":
    unittest.main()
