"""
Tests for the pre-labeling rules (autolabel.py). Pre-labels are what a person
starts correcting from, so each test pins a rule from the label definitions
(TRAINING_DESIGN.md section 6): if a rule drifts, every pre-labeled frame of
that kind starts out wrong.
"""
import unittest

import numpy as np

from autolabel import cohen_kappa, compare, label_hand

FPS = 30


def trace(*segments, start=0.0):
    """Wrist states (N, 2) from (target height in palm lengths, frames to get there) pieces; angle follows height."""
    heights, current = [], start
    for target, frames in segments:
        heights.extend(np.linspace(current, target, frames + 1)[1:])
        current = target
    heights = np.array(heights)
    return np.stack([heights, 0.2 * heights], axis=1)


def eased(start, end, frames):
    """Wrist states moving start -> end with a smooth start and stop, the way a hand is lowered or raised."""
    heights = start + (end - start) * (1 - np.cos(np.linspace(0, np.pi, frames + 1)[1:])) / 2
    return np.stack([heights, 0.2 * heights], axis=1)


def labels_of(states, detected=None):
    if detected is None:
        detected = np.ones(len(states), dtype=bool)
    return label_hand(states, detected, FPS)


def strokes(labels):
    """The stroke labels in order, one per run."""
    out = []
    for i, label in enumerate(labels):
        if label and label.endswith("_stroke") and (i == 0 or labels[i - 1] != label):
            out.append(label)
    return out


HOLD = 20
# Wrist heights: 0 at the drum, 3 palm lengths high, 0.8 for a small lift.
HIGH, LIFT = 3.0, 0.8
FAST, SLOW = 3, 8  # frames for a hit and for a lift


class TestSingleStrokes(unittest.TestCase):
    # The thesis clips each hold at one height (Full stays high, Down and
    # Tab stay low), so these also check the one-height fallback.

    def test_full_starts_high_hits_and_returns_high(self):
        s = trace((HIGH, HOLD), (0, FAST), (HIGH, SLOW), (HIGH, HOLD), start=HIGH)
        labels = labels_of(s)
        self.assertEqual(strokes(labels), ["full_stroke"])
        self.assertEqual((labels[5], labels[-5]), ("up_hold", "up_hold"))

    def test_down_lifts_high_hits_and_stays_low(self):
        s = trace((0, HOLD), (HIGH, SLOW), (0, FAST), (0, HOLD))
        labels = labels_of(s)
        self.assertEqual(strokes(labels), ["down_stroke"])
        self.assertEqual((labels[5], labels[-5]), ("down_hold", "down_hold"))

    def test_up_hits_from_low_and_rises_to_stay_high(self):
        s = trace((0, HOLD), (LIFT, 4), (0, FAST), (HIGH, SLOW), (HIGH, HOLD))
        labels = labels_of(s)
        self.assertEqual(strokes(labels), ["up_stroke"])
        self.assertEqual((labels[5], labels[-5]), ("down_hold", "up_hold"))

    def test_tap_hits_from_low_and_stays_low(self):
        s = trace((0, HOLD), (LIFT, 4), (0, FAST), (0, HOLD))
        self.assertEqual(strokes(labels_of(s)), ["tab_stroke"])


class TestSequences(unittest.TestCase):
    def test_full_down_tap_up_played_without_pauses_gives_each_stroke_in_order(self):
        # The accent exercise in RECORDING_GUIDE.md; one stroke per hit, and
        # the boundary between strokes is the top of the path between hits.
        s = trace((HIGH, HOLD), (0, FAST), (HIGH, SLOW),   # full
                  (0, FAST),                               # down (ends low)
                  (LIFT, 4), (0, FAST),                    # tap
                  (LIFT, 4), (0, FAST), (HIGH, SLOW),      # up (ends high)
                  (HIGH, HOLD), start=HIGH)
        self.assertEqual(strokes(labels_of(s)), ["full_stroke", "down_stroke", "tab_stroke", "up_stroke"])

    def test_moving_back_down_after_an_up_stroke_is_none(self):
        # The user's decision: preparation that is not a stroke is none.
        s = np.concatenate([trace((0, HOLD), (LIFT, 4), (0, FAST), (HIGH, SLOW), (HIGH, HOLD)),
                            eased(HIGH, 0, 20), trace((0, HOLD))])
        labels = labels_of(s)
        self.assertEqual(strokes(labels), ["up_stroke"])
        self.assertIn("none", labels)
        self.assertEqual(labels[-5], "down_hold")

    def test_lowering_the_hand_is_not_counted_as_a_hit_even_when_quick(self):
        # Peak speed here (about 7 palm lengths/s) is above HIT_SPEED; what
        # tells it apart from a hit is that it slows down before stopping.
        s = np.concatenate([trace((HIGH, HOLD), start=HIGH), eased(HIGH, 0, 20), trace((0, HOLD))])
        self.assertEqual(strokes(labels_of(s)), [])


class TestUndetected(unittest.TestCase):
    def test_frames_before_the_hand_is_seen_get_no_label(self):
        # No label means "left out of training", never a guess.
        s = trace((0, HOLD), (LIFT, 4), (0, FAST), (0, HOLD))
        detected = np.ones(len(s), dtype=bool)
        detected[:5] = False
        labels = labels_of(s, detected)
        self.assertEqual(labels[:5], [None] * 5)
        self.assertIsNotNone(labels[5])

    def test_a_hand_never_seen_gets_no_labels(self):
        s = trace((0, 40))
        self.assertEqual(labels_of(s, np.zeros(len(s), dtype=bool)), [None] * len(s))


class TestAgreement(unittest.TestCase):
    def test_kappa_is_one_for_identical_labels_and_zero_for_chance_level(self):
        a = ["up_hold", "down_hold"] * 10
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0)
        # b agrees on half the frames, exactly what chance gives here.
        b = ["up_hold", "up_hold", "down_hold", "down_hold"] * 5
        self.assertAlmostEqual(cohen_kappa(a, b), 0.0)

    def test_compare_uses_only_frames_both_labeled_and_scores_strokes(self):
        human = {f: {"left": "full_stroke" if 10 <= f < 20 else "up_hold"} for f in range(40)}
        auto = {f: {"left": "full_stroke" if 12 <= f < 22 else "up_hold"} for f in range(5, 40)}
        result = compare(auto, human)["left"]
        self.assertEqual(result["frames"], 35)
        self.assertAlmostEqual(result["agreement"], 31 / 35)
        full = result["strokes"]["Full"]
        self.assertEqual((full["labeled"], full["detected"], full["f1"]), (1, 1, 1.0))  # 2 frames late: within ±3


if __name__ == "__main__":
    unittest.main()
