"""
Tests for the thesis features added to the code on 2026-11-13: the
speed-invariant Lie group embedding, wrist height and angle, time-stretch
augmentation, stroke timing and movement metrics, the thesis accuracy and
suggestion rule, and the event-level evaluation. The app counts strokes, so
each test says what would go wrong in the counts or the feedback if the
behaviour changed.
"""
import unittest

import numpy as np
import torch

from evaluate import labeled_strokes, match_events, score
from features import hand_state, resample_length
from inference import WINDOW_SIZE, load_checkpoint, stroke_metrics, stroke_spans
from model import LieGroupEmbedding, PathDevelopment
from pipeline import build_suggestions, expected_count_accuracy
from train import TIME_STRETCH_FACTORS, stretched_starts

FULL, DOWN, UP, TAB, DOWN_HOLD, UP_HOLD, NONE = range(7)


def _path(num_frames, curved=False):
    """One node moving from (0, 0, 0) to (1, 1, 0) along a line or a quarter circle: (1, T, 1, 9)."""
    s = np.linspace(0.0, 1.0, num_frames)
    xyz = np.stack([np.sin(s * np.pi / 2), 1 - np.cos(s * np.pi / 2), 0 * s], axis=1) if curved else np.stack([s, s, 0 * s], axis=1)
    x = np.zeros((1, num_frames, 1, 9), dtype=np.float32)
    x[0, :, 0, :3] = xyz
    return torch.from_numpy(x)


class TestPathDevelopment(unittest.TestCase):
    # The thesis credits the Lie group embedding with not depending on how
    # fast a motion is made. Drummers change tempo all the time; if the same
    # stroke played slower looks different to the model, counts break.

    def setUp(self):
        torch.manual_seed(0)

    def test_same_straight_path_at_half_speed_gives_the_same_embedding(self):
        dev = PathDevelopment()
        fast, slow = dev(_path(11)), dev(_path(21))
        torch.testing.assert_close(fast[0, -1], slow[0, -1], atol=1e-5, rtol=1e-5)

    def test_same_curved_path_at_half_speed_gives_nearly_the_same_embedding(self):
        dev = PathDevelopment()
        fast, slow = dev(_path(16, curved=True)), dev(_path(31, curved=True))
        self.assertLess((fast[0, -1] - slow[0, -1]).abs().max().item(), 0.02)

    def test_the_original_cumulative_embedding_does_depend_on_speed(self):
        # Pins why PathDevelopment was added: summing a linear map over frames
        # gives a different result when the same path takes more frames.
        emb = LieGroupEmbedding(input_dim=9, lie_dim=16)
        fast, slow = emb(_path(11)), emb(_path(21))
        self.assertGreater((fast[0, -1] - slow[0, -1]).abs().max().item(), 0.1)

    def test_output_is_a_rotation(self):
        out = PathDevelopment()(_path(10, curved=True))[0, -1, 0].view(4, 4)
        torch.testing.assert_close(out @ out.T, torch.eye(4), atol=1e-5, rtol=1e-5)


class TestOldCheckpoint(unittest.TestCase):
    def test_checkpoint_without_options_loads_as_the_original_model(self):
        # The app must keep working with the model trained before these
        # options existed.
        model, scaler, _ = load_checkpoint("multi_motion_model_7class.pt", torch.device("cpu"))
        self.assertEqual((model.embedding, model.state_dim), ("cumulative", 0))
        self.assertIsNone(scaler.state_mean)


def _raw_hand(wrist_y, palm=(0.0, -50.0), scale=1.0):
    """Raw coords (N, 21, 3) with the wrist at wrist_y and the middle knuckle at wrist + palm, all scaled."""
    coords = np.zeros((len(wrist_y), 21, 3), dtype=np.float32)
    coords[:, 0, 1] = wrist_y
    coords[:, 9, 0] = palm[0]
    coords[:, 9, 1] = np.asarray(wrist_y) + palm[1]
    return coords * scale


class TestHandState(unittest.TestCase):
    # High and low positions decide between the four strokes (thesis table
    # 4.2), and the wrist-relative positions have the height removed.

    def test_height_is_measured_up_from_the_lowest_wrist_position(self):
        states = hand_state(_raw_hand([400.0, 350.0, 300.0]))
        np.testing.assert_allclose(states[:, 0], [0.0, 1.0, 2.0])  # palm length 50 px

    def test_height_does_not_change_with_camera_distance(self):
        # A user filming from twice as far must get the same heights.
        near = hand_state(_raw_hand([400.0, 350.0, 300.0]))
        far = hand_state(_raw_hand([400.0, 350.0, 300.0], scale=0.5))
        np.testing.assert_allclose(near, far, atol=1e-5)

    def test_angle_is_zero_with_the_palm_pointing_up_and_grows_as_it_tips(self):
        up = hand_state(_raw_hand([300.0, 300.0]))
        tipped = hand_state(_raw_hand([300.0, 300.0], palm=(50.0, -50.0)))
        np.testing.assert_allclose(up[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(tipped[:, 1], np.pi / 4, atol=1e-6)

    def test_frames_before_the_hand_is_first_seen_are_zero(self):
        raw = _raw_hand([0.0, 300.0, 250.0])
        raw[0] = np.nan
        self.assertTrue(np.all(hand_state(raw)[0] == 0))


class TestTimeStretch(unittest.TestCase):
    def test_stretched_copies_end_on_the_same_frame_so_they_keep_its_label(self):
        # A window's label is its last frame's label; a copy ending elsewhere
        # would be trained on the wrong label.
        start = 100
        copies = stretched_starts(start)
        self.assertEqual(len(copies), len(TIME_STRETCH_FACTORS))
        for first, span in copies:
            self.assertEqual(first + span, start + WINDOW_SIZE)
        self.assertEqual(sorted(span for _, span in copies), sorted(round(WINDOW_SIZE * f) for f in TIME_STRETCH_FACTORS))

    def test_copies_that_would_start_before_the_video_are_skipped(self):
        self.assertTrue(all(first >= 0 for first, _ in stretched_starts(0)))

    def test_resampling_a_span_keeps_its_start_and_end(self):
        values = np.arange(38, dtype=np.float32)[:, None]
        out = resample_length(values, WINDOW_SIZE)
        self.assertEqual(out.shape, (WINDOW_SIZE, 1))
        self.assertEqual((out[0, 0], out[-1, 0]), (0.0, 37.0))


class TestStrokeSpans(unittest.TestCase):
    def test_span_frames_are_the_last_frames_of_its_windows(self):
        spans = stroke_spans([UP_HOLD] * 5 + [FULL] * 6 + [UP_HOLD] * 5)
        offset = WINDOW_SIZE - 1
        self.assertEqual(spans, [(FULL, 5 + offset, 10 + offset)])

    def test_a_dropped_flicker_does_not_split_one_stroke_into_two(self):
        spans = stroke_spans([DOWN] * 6 + [NONE] + [DOWN] * 6)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][2] - spans[0][1], 12)


class TestStrokeMetrics(unittest.TestCase):
    def test_evenly_spaced_identical_strokes_are_perfectly_consistent(self):
        # Three identical strokes, one every second: zero spread everywhere.
        states = np.zeros((100, 2), dtype=np.float32)
        for start in (0, 30, 60):
            states[start:start + 10, 0] = [0, 1, 2, 3, 2, 1, 0, 0, 0, 0]
        spans = [(FULL, s, s + 9) for s in (0, 30, 60)]
        m = stroke_metrics(spans, states, fps=30)
        full = m["strokes"]["Full"]
        self.assertEqual(full["count"], 3)
        self.assertEqual((full["peak_height_mean"], full["peak_height_cv"]), (3.0, 0.0))
        self.assertEqual(full["peak_down_speed_mean"], 30.0)  # 1 palm length per frame at 30fps
        self.assertEqual((m["interval_s_mean"], m["interval_s_cv"]), (1.0, 0.0))

    def test_uneven_timing_shows_up_as_interval_spread(self):
        states = np.zeros((100, 2), dtype=np.float32)
        m = stroke_metrics([(TAB, 0, 5), (TAB, 20, 25), (TAB, 60, 65)], states, fps=30)
        self.assertGreater(m["interval_s_cv"], 0.2)

    def test_no_strokes_gives_empty_measurements_not_zeros(self):
        m = stroke_metrics([], np.zeros((10, 2), dtype=np.float32))
        self.assertEqual(m["strokes"]["Up"]["count"], 0)
        self.assertIsNone(m["strokes"]["Up"]["peak_height_mean"])
        self.assertIsNone(m["interval_s_mean"])


class TestThesisAccuracyAndSuggestions(unittest.TestCase):
    COUNTS = {"Full": 8, "Down": 12, "Up": 0, "Tab": 5}

    def test_accuracy_is_correct_over_actual_with_extra_detections_not_rewarded(self):
        # Thesis 4.6.5. Detecting 12 Down when 10 were played must not
        # score above 100%.
        acc = expected_count_accuracy(self.COUNTS, {"Full": 10, "Down": 10, "Up": 4})
        self.assertEqual((acc["Full"], acc["Down"], acc["Up"]), (80.0, 100.0, 0.0))
        self.assertIsNone(acc["Tab"])  # no actual count given: cannot be computed

    def test_below_90_gets_an_improvement_suggestion_and_90_or_more_a_positive_one(self):
        # Thesis 4.6.6.
        acc = {"Full": 80.0, "Down": 100.0, "Up": None, "Tab": None}
        suggestions = build_suggestions([("left", self.COUNTS, acc, True)])
        self.assertEqual(len(suggestions), 2)
        self.assertIn("more distinct", suggestions[0])
        self.assertIn("Full", suggestions[0])
        self.assertIn("keep it up", suggestions[1])

    def test_strokes_that_were_never_detected_get_no_self_consistency_suggestion(self):
        # Before, a stroke the user never played was told it "wasn't reliably
        # recognized"; there is nothing to judge.
        acc = {"Full": 100.0, "Down": 100.0, "Up": 0.0, "Tab": 100.0}
        suggestions = build_suggestions([("right", self.COUNTS, acc, False)])
        self.assertFalse(any("'Up'" in s for s in suggestions))


class TestEventEvaluation(unittest.TestCase):
    # The app's number is a count per stroke type, so this is what is scored.

    def test_runs_of_the_same_stroke_label_become_one_labeled_stroke_each(self):
        labels = {f: {"left": "full_stroke"} for f in range(0, 10)}
        labels.update({f: {"left": "up_hold"} for f in range(10, 15)})
        labels.update({f: {"left": "full_stroke"} for f in range(15, 25)})
        self.assertEqual(labeled_strokes(labels, "left"), [(FULL, 0, 9), (FULL, 15, 24)])

    def test_a_missing_frame_ends_a_labeled_stroke(self):
        labels = {f: {"left": "tab_stroke"} for f in (0, 1, 2, 4, 5)}
        self.assertEqual(len(labeled_strokes(labels, "left")), 2)

    def test_detection_matches_within_the_boundary_tolerance_only(self):
        labeled = [(DOWN, 100, 115)]
        self.assertEqual(match_events([(DOWN, 117, 130)], labeled)[DOWN], 1)  # 2 frames late
        self.assertEqual(match_events([(DOWN, 120, 130)], labeled)[DOWN], 0)  # 5 frames late
        self.assertEqual(match_events([(UP, 100, 115)], labeled)[DOWN], 0)  # wrong type

    def test_one_labeled_stroke_cannot_be_matched_twice(self):
        # Two detections of one stroke is a double count: one must be a false positive.
        result = score([(FULL, 0, 5), (FULL, 6, 10)], [(FULL, 0, 10)])["Full"]
        self.assertEqual((result["labeled"], result["detected"], result["count_error"]), (1, 2, 1))
        self.assertEqual((result["precision"], result["recall"]), (0.5, 1.0))



class TestExpectedCountsInput(unittest.TestCase):
    # The user types these numbers; a bad entry must be refused with a
    # message, never silently turned into a wrong accuracy.

    @classmethod
    def setUpClass(cls):
        from fastapi import HTTPException
        from main import parse_expected_counts
        cls.parse, cls.error = staticmethod(parse_expected_counts), HTTPException

    def test_empty_means_not_given(self):
        self.assertIsNone(self.parse(""))
        self.assertIsNone(self.parse('{"left": {"Full": 0}}'))  # zeros only: nothing to compare against

    def test_valid_counts_are_kept(self):
        self.assertEqual(self.parse('{"left": {"Full": 10}, "right": {"Down": 4, "Up": 0}}'),
                         {"left": {"Full": 10}, "right": {"Down": 4}})

    def test_bad_entries_are_refused(self):
        for text in ('{"left": {"Full": -1}}', '{"left": {"Full": 2.5}}', '{"left": {"Roll": 3}}',
                     '{"middle": {"Full": 3}}', '{"left": {"Full": true}}', "not json", "[1, 2]"):
            with self.assertRaises(self.error, msg=text):
                self.parse(text)



def _hand_with_arm(forearm_angle, wrist_bend, frames=3):
    """
    Raw hand coords (N, 21, 3) and elbows (N, 2) for an arm pointing at
    forearm_angle (radians from straight up) with the hand bent by wrist_bend
    relative to the forearm. Palm length 50 px, forearm 200 px.
    """
    wrist = np.array([640.0, 400.0])
    up = lambda a: np.array([np.sin(a), -np.cos(a)])  # image y grows downward
    elbow = wrist - 200 * up(forearm_angle)
    knuckle = wrist + 50 * up(forearm_angle + wrist_bend)
    coords = np.zeros((frames, 21, 3), dtype=np.float32)
    coords[:, 0, :2], coords[:, 9, :2] = wrist, knuckle
    return coords, np.tile(elbow, (frames, 1)).astype(np.float32)


class TestForearmWristAngle(unittest.TestCase):
    # The thesis defines high and low by the wrist angle, i.e. hand against
    # forearm. Without an elbow, the palm's direction is used instead, which
    # also changes when the whole arm moves. With the elbow, it should not.

    def test_straight_wrist_is_zero_and_a_bent_wrist_gives_the_bend(self):
        coords, elbows = _hand_with_arm(0.3, 0.0)
        np.testing.assert_allclose(hand_state(coords, elbows)[:, 1], 0.0, atol=1e-5)
        coords, elbows = _hand_with_arm(0.3, 0.6)
        np.testing.assert_allclose(np.abs(hand_state(coords, elbows)[:, 1]), 0.6, atol=1e-5)

    def test_tilting_the_whole_arm_does_not_change_the_wrist_angle(self):
        a = hand_state(*_hand_with_arm(0.0, 0.5))[:, 1]
        b = hand_state(*_hand_with_arm(0.8, 0.5))[:, 1]
        np.testing.assert_allclose(a, b, atol=1e-5)
        # The palm-direction stand-in does change, which is why the elbow helps.
        pa = hand_state(_hand_with_arm(0.0, 0.5)[0])[:, 1]
        pb = hand_state(_hand_with_arm(0.8, 0.5)[0])[:, 1]
        self.assertGreater(abs(pa[0] - pb[0]), 0.5)

    def test_without_any_elbow_the_palm_direction_is_used(self):
        coords, _ = _hand_with_arm(0.0, 0.5)
        no_elbows = np.full((len(coords), 2), np.nan, dtype=np.float32)
        np.testing.assert_allclose(hand_state(coords, no_elbows), hand_state(coords), atol=1e-6)


class TestElbowMatching(unittest.TestCase):
    def test_the_arm_whose_wrist_is_nearest_the_hand_is_taken(self):
        from features import REFERENCE_HEIGHT, REFERENCE_WIDTH, match_elbow
        coords, _ = _hand_with_arm(0.0, 0.0, frames=1)
        hand = coords[0]
        near_wrist = (hand[0, 0] / REFERENCE_WIDTH + 0.005, hand[0, 1] / REFERENCE_HEIGHT)
        far_wrist = (0.1, 0.1)
        elbow = match_elbow(hand, [((0.2, 0.2), far_wrist), ((0.5, 0.9), near_wrist)])
        np.testing.assert_allclose(elbow, [0.5 * REFERENCE_WIDTH, 0.9 * REFERENCE_HEIGHT], atol=1e-3)

    def test_an_arm_too_far_from_the_hand_is_not_used(self):
        # The other person's arm, or the other arm, must not be paired.
        from features import match_elbow
        coords, _ = _hand_with_arm(0.0, 0.0, frames=1)
        self.assertIsNone(match_elbow(coords[0], [((0.0, 0.0), (0.05, 0.05))]))


class TestTrainingHelpers(unittest.TestCase):
    def test_rare_classes_get_proportionally_larger_loss_weights(self):
        # Holds and none outnumber strokes; unweighted, mostly predicting
        # them scores well while missing strokes, which are what is counted.
        from train import class_weights
        w = class_weights([0] * 10 + [5] * 90, 7)
        self.assertAlmostEqual(w[0].item() / w[5].item(), 9.0, places=5)
        self.assertEqual(w[1].item(), 0.0)  # absent from training
        self.assertAlmostEqual(w[0].item() * 10, w[5].item() * 90, places=4)  # equal total weight per class

    def test_macro_f1_is_not_fooled_by_predicting_the_common_class(self):
        from train import macro_f1
        y_true = [5] * 90 + [0] * 10
        always_hold = [5] * 100
        self.assertLess(macro_f1(y_true, always_hold), 0.5)  # accuracy would be 0.90

    def test_the_same_seed_gives_the_same_model(self):
        from model import StrokeBiLSTMModel
        from train import set_seed
        set_seed(3)
        a = StrokeBiLSTMModel(7).fc.weight.clone()
        set_seed(3)
        b = StrokeBiLSTMModel(7).fc.weight.clone()
        torch.testing.assert_close(a, b)


class TestServiceLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main

    def test_old_result_images_and_finished_jobs_are_cleaned_up(self):
        # Without this, every upload left a PNG and a job behind forever.
        import os
        import time
        main = self.main
        now = time.time()
        old_png = os.path.join(main.STATIC_DIR, "Left_test-old.png")
        new_png = os.path.join(main.STATIC_DIR, "Left_test-new.png")
        for path in (old_png, new_png):
            open(path, "wb").close()
        os.utime(old_png, (now - main.RETENTION_SECONDS - 10,) * 2)
        with main._jobs_lock:
            main._jobs["old"] = {"status": "done", "finished_at": now - main.RETENTION_SECONDS - 10}
            main._jobs["running"] = {"status": "running", "progress": 0.5}
        try:
            main.cleanup_old_files(now)
            self.assertFalse(os.path.exists(old_png))
            self.assertTrue(os.path.exists(new_png))
            self.assertNotIn("old", main._jobs)
            self.assertIn("running", main._jobs)
        finally:
            for path in (old_png, new_png):
                if os.path.exists(path):
                    os.remove(path)
            main._jobs.pop("running", None)

    def test_a_video_over_the_length_limit_is_refused(self):
        from fastapi import HTTPException
        main = self.main
        limit = main.MAX_VIDEO_SECONDS
        main.MAX_VIDEO_SECONDS = 0.1  # the 5-frame 10fps fixture is 0.5 s
        try:
            with self.assertRaises(HTTPException) as caught:
                main.check_video_length("test_data/rotation_tag_90.mp4")
            self.assertEqual(caught.exception.status_code, 413)
        finally:
            main.MAX_VIDEO_SECONDS = limit


if __name__ == "__main__":
    unittest.main()
