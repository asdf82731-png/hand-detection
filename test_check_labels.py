"""
Tests for check_labels.py. The checker is the only thing that notices a label
sequence a hand cannot make; without it a wrong frame is found after training,
as a model that is quietly worse, and the fix is labeling the video again.
"""
import unittest

from check_labels import label_runs, stroke_order_warnings, uncovered_frames


def _frames(sequence, hand="left", first=0):
    """[(label, count)] -> {frame: {hand: label}} on consecutive frames."""
    frames, frame = {}, first
    for label, count in sequence:
        for _ in range(count):
            frames[frame] = {hand: label}
            frame += 1
    return frames


def _warnings(sequence):
    return stroke_order_warnings(label_runs(_frames(sequence), "left"))


class TestStrokeOrder(unittest.TestCase):
    def test_a_legal_cycle_gives_no_warning(self):
        # Full -> Down -> Tap -> Up is the thesis's own accent cycle (RECORDING_GUIDE E1):
        # every stroke starts where the last one ended.
        self.assertEqual(_warnings([
            ("up_hold", 3), ("full_stroke", 10), ("down_stroke", 15), ("down_hold", 2), ("tab_stroke", 12),
            ("up_stroke", 16), ("up_hold", 2)]), [])

    def test_two_strokes_the_hand_cannot_chain_are_reported(self):
        # The thesis allows only Up or Tap after a Down: either a stroke was missed
        # between them or one of the two is mislabeled. The same with only a hold between.
        for sequence in ([("down_stroke", 10), ("full_stroke", 10)],
                         [("down_stroke", 10), ("down_hold", 2), ("down_stroke", 10)]):
            warnings = _warnings(sequence)
            self.assertEqual(len(warnings), 1, sequence)
            self.assertIn("may only be followed by", warnings[0])

    def test_the_lift_before_a_stroke_may_sit_after_a_hold_at_the_other_end(self):
        # Decision of 2026-11-13: the lift before a Down is part of the Down, so a
        # down_hold right before a down_stroke is correct and must not be reported.
        self.assertEqual(_warnings([("down_hold", 3), ("down_stroke", 15), ("down_hold", 3)]), [])

    def test_a_hold_that_changes_place_with_no_movement_is_reported(self):
        # The hand cannot go from low to high while every frame is labeled at rest:
        # either a stroke or none is missing between the two holds.
        for sequence in ([("down_hold", 5), ("up_hold", 5)],
                         [("up_stroke", 16), ("up_hold", 3), ("down_hold", 3)]):
            warnings = _warnings(sequence)
            self.assertEqual(len(warnings), 1, sequence)
            self.assertIn("but the hand was", warnings[0])

    def test_a_hold_that_contradicts_the_last_stroke_is_reported(self):
        # Down ends low; calling the rest after it up_hold puts the hand high.
        warnings = _warnings([("down_stroke", 10), ("up_hold", 5)])
        self.assertEqual(len(warnings), 1)
        self.assertIn("up_hold", warnings[0])

    def test_none_between_strokes_is_never_a_warning(self):
        # The decision of 2026-11-13: after an up_stroke the hand returns to low
        # and that return is none, so what follows may start low without a hold.
        self.assertEqual(_warnings([
            ("down_hold", 3), ("up_stroke", 16), ("up_hold", 2), ("none", 4), ("tab_stroke", 12)]), [])

    def test_nothing_is_checked_until_the_position_is_known(self):
        # A video can start in any pose: the first stroke or hold has nothing to contradict.
        self.assertEqual(_warnings([("up_stroke", 16)]), [])
        self.assertEqual(_warnings([("none", 5), ("down_stroke", 10)]), [])

    def test_a_gap_of_frames_with_no_row_forgets_the_position(self):
        # Frames with no row are frames where the hand was not found: it may have
        # moved, so the next label is not compared with the one before the gap.
        frames = _frames([("down_stroke", 10)])
        frames.update(_frames([("down_stroke", 10)], first=30))
        self.assertEqual(stroke_order_warnings(label_runs(frames, "left")), [])


class TestRunsAndCoverage(unittest.TestCase):
    def test_runs_join_only_consecutive_frames(self):
        frames = _frames([("down_hold", 3)])
        frames.update(_frames([("down_hold", 2)], first=10))
        self.assertEqual(label_runs(frames, "left"), [("down_hold", 0, 2), ("down_hold", 10, 11)])

    def test_hands_are_checked_separately(self):
        frames = {0: {"left": "down_stroke", "right": "up_hold"}, 1: {"left": "down_stroke"}}
        self.assertEqual(label_runs(frames, "right"), [("up_hold", 0, 0)])

    def test_frames_with_no_row_are_counted_not_filled(self):
        frames = _frames([("down_hold", 3)])
        frames.update(_frames([("down_hold", 2)], first=6))
        self.assertEqual(uncovered_frames(frames, "left"), [3, 4, 5])
        self.assertEqual(uncovered_frames(frames, "right"), [])


if __name__ == "__main__":
    unittest.main()
