"""
Unit tests for features.py. Each one pins a rule from the thesis method
(chapter 3, A1-A2) so that drifting away from it fails a test instead of
silently changing what the model is trained on.
"""
import unittest

import numpy as np

from features import FeatureScaler, fill_missing, pack_model_input, window_features, wrist_relative
from pipeline import BONE_CONNECTIONS


def _hand(value):
    return np.full((21, 3), value, dtype=np.float32)


class TestFillMissing(unittest.TestCase):
    def test_gap_between_detections_is_linearly_interpolated(self):
        # A missed detection must not show up as the hand jumping to the
        # origin and back, which the velocity features would read as motion.
        coords = np.stack([_hand(10.0), _hand(np.nan), _hand(np.nan), _hand(40.0)])
        filled = fill_missing(coords)
        np.testing.assert_allclose(filled[1], _hand(20.0))
        np.testing.assert_allclose(filled[2], _hand(30.0))

    def test_frames_before_first_detection_are_zero(self):
        # No history to interpolate from: the thesis fills these with zeros.
        coords = np.stack([_hand(np.nan), _hand(np.nan), _hand(5.0)])
        filled = fill_missing(coords)
        np.testing.assert_allclose(filled[:2], 0.0)
        np.testing.assert_allclose(filled[2], _hand(5.0))

    def test_frames_after_last_detection_hold_the_last_position(self):
        coords = np.stack([_hand(5.0), _hand(7.0), _hand(np.nan)])
        np.testing.assert_allclose(fill_missing(coords)[2], _hand(7.0))

    def test_hand_never_detected_is_all_zero(self):
        self.assertFalse(fill_missing(np.stack([_hand(np.nan)] * 3)).any())


class TestWristRelative(unittest.TestCase):
    def test_moving_the_whole_hand_does_not_change_positions(self):
        # Where the player stands in the frame must not affect the features.
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(5, 21, 3)).astype(np.float32)
        shifted = coords + np.array([300.0, -120.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(wrist_relative(coords), wrist_relative(shifted), atol=1e-4)
        self.assertFalse(wrist_relative(coords)[:, 0].any())


class TestWindowFeatures(unittest.TestCase):
    def test_velocity_and_acceleration_are_differences_over_time(self):
        # Joint 8 moves with constant acceleration along x: x = t^2.
        # (The old pipeline differenced across joint indices, not frames.)
        T = 6
        positions = np.zeros((T, 21, 3), dtype=np.float32)
        positions[:, 8, 0] = np.arange(T) ** 2
        joints, _ = window_features(positions)

        self.assertEqual(joints.shape, (T, 21, 9))
        np.testing.assert_allclose(joints[:, 8, 0], np.arange(T) ** 2)       # position
        np.testing.assert_allclose(joints[:-1, 8, 3], [1, 3, 5, 7, 9])       # velocity
        np.testing.assert_allclose(joints[:-2, 8, 6], [2, 2, 2, 2])          # acceleration

    def test_tail_of_velocity_and_acceleration_is_zero_padded(self):
        positions = np.random.default_rng(1).normal(size=(6, 21, 3)).astype(np.float32)
        joints, bones = window_features(positions)
        for features in (joints, bones):
            self.assertFalse(features[-1, :, 3:6].any())
            self.assertFalse(features[-2:, :, 6:9].any())

    def test_bone_vector_is_child_joint_minus_parent_joint(self):
        positions = np.random.default_rng(2).normal(size=(4, 21, 3)).astype(np.float32)
        _, bones = window_features(positions)
        self.assertEqual(bones.shape, (4, 20, 9))
        for i, (parent, child) in enumerate(BONE_CONNECTIONS):
            np.testing.assert_allclose(bones[:, i, :3], positions[:, child] - positions[:, parent], atol=1e-6)


class TestPackModelInput(unittest.TestCase):
    def test_each_hand_fills_only_its_own_joint_slot(self):
        # A hand written into the other hand's slot would be trained and
        # counted as the wrong hand.
        joints = np.ones((30, 21, 9), dtype=np.float32)
        bones = np.full((30, 20, 9), 2.0, dtype=np.float32)

        left = pack_model_input(joints, bones, "left")
        self.assertEqual(left.shape, (30, 62, 9))
        self.assertTrue((left[:, :21] == 1).all())
        self.assertFalse(left[:, 21:42].any())

        right = pack_model_input(joints, bones, "right")
        self.assertFalse(right[:, :21].any())
        self.assertTrue((right[:, 21:42] == 1).all())

        for packed in (left, right):
            self.assertTrue((packed[:, 42:] == 2).all())


class TestFeatureScaler(unittest.TestCase):
    def test_training_data_is_standardized_per_channel(self):
        # Pixel positions are hundreds, depth and accelerations are tiny;
        # without a common scale the large channels dominate.
        rng = np.random.default_rng(3)
        scale = np.array([300, 300, 0.05, 20, 20, 0.01, 5, 5, 0.005], dtype=np.float32)
        joints = [rng.normal(size=(30, 21, 9)).astype(np.float32) * scale for _ in range(20)]
        bones = [rng.normal(size=(30, 20, 9)).astype(np.float32) * scale for _ in range(20)]
        scaler = FeatureScaler.fit(joints, bones)

        scaled = np.concatenate([scaler.transform(j, b)[0].reshape(-1, 9) for j, b in zip(joints, bones)])
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-3)

    def test_state_dict_round_trip_gives_identical_scaling(self):
        # Inference must scale with exactly the statistics the model was trained with.
        rng = np.random.default_rng(4)
        joints = [rng.normal(size=(30, 21, 9)).astype(np.float32) for _ in range(5)]
        bones = [rng.normal(size=(30, 20, 9)).astype(np.float32) for _ in range(5)]
        scaler = FeatureScaler.fit(joints, bones)
        restored = FeatureScaler.from_state_dict(scaler.state_dict())
        for a, b in zip(scaler.transform(joints[0], bones[0]), restored.transform(joints[0], bones[0])):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
