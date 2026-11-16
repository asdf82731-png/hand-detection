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
        self.assertEqual((model.lstm_input, model.bidir_ends, model.output_gelu), (None, False, True))
        self.assertIsNone(scaler.state_mean)


class TestArchitectureFixes(unittest.TestCase):
    # Flow report M1-M3 (REV-8). Off by default so old checkpoints still load.

    def test_projection_shrinks_the_lstms(self):
        # M1: the LSTMs held 99.8% of 4.3M parameters for a few thousand
        # overlapping windows; projecting first must cut that sharply.
        from model import StrokeBiLSTMModel
        count = lambda m: sum(p.numel() for p in m.parameters())
        original, compact = StrokeBiLSTMModel(7), StrokeBiLSTMModel(7, state_dim=6, lstm_input=128)
        self.assertLess(count(compact), count(original) / 4)
        out = compact.eval()(torch.randn(2, 30, 62, 9), torch.randn(2, 30, 6))
        self.assertEqual(out.shape, (2, 7))

    def test_bidir_ends_lets_the_backward_direction_see_the_whole_window(self):
        # M2: read at the last step, the backward LSTM has seen only the last
        # frame, so earlier frames cannot change that half of the summary.
        from model import StrokeBiLSTMModel
        torch.manual_seed(0)
        lstm = torch.nn.LSTM(8, 128, batch_first=True, bidirectional=True)
        x = torch.randn(1, 30, 8)
        y = x.clone()
        y[:, :-1] += 1.0  # change every frame but the last
        for bidir_ends, should_change in ((False, False), (True, True)):
            model = StrokeBiLSTMModel(7, bidir_ends=bidir_ends)
            backward = [model._summary(lstm(v)[0])[:, 128:] for v in (x, y)]
            self.assertEqual(not torch.allclose(*backward), should_change, f"bidir_ends={bidir_ends}")

    def test_without_output_gelu_wrong_classes_can_be_scored_low(self):
        # M3: GELU floors every score at about -0.17.
        from model import StrokeBiLSTMModel
        x = torch.randn(2, 30, 62, 9)
        for output_gelu in (True, False):
            model = StrokeBiLSTMModel(7, output_gelu=output_gelu).eval()
            with torch.no_grad():
                model.fc.bias.fill_(-5.0)
                model.fc.weight.zero_()
                lowest = float(model(x).min())
            if output_gelu:
                self.assertGreater(lowest, -0.2)
            else:
                self.assertAlmostEqual(lowest, -5.0, places=4)

    def test_checkpoint_rebuilds_the_architecture_it_was_trained_with(self):
        import os, tempfile
        from features import FeatureScaler
        from model import StrokeBiLSTMModel
        config = {"lstm_input": 128, "bidir_ends": True, "output_gelu": False}
        model = StrokeBiLSTMModel(7, **config)
        scaler = FeatureScaler(*[np.zeros(9, dtype=np.float32)] * 4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.pt")
            torch.save({"model_state": model.state_dict(), "scaler": scaler.state_dict(), "labels": list(range(7)),
                        "hand_to_slot": {"left": "left", "right": "right"}, "config": config}, path)
            loaded, _, _ = load_checkpoint(path, torch.device("cpu"))
        self.assertEqual((loaded.lstm_input, loaded.bidir_ends, loaded.output_gelu), (128, True, False))


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
        # In a portrait frame too: the pose model's normalized coordinates must
        # be scaled like the hand's, or the elbow lands off the arm.
        from features import match_elbow
        coords, _ = _hand_with_arm(0.0, 0.0, frames=1)
        hand = coords[0]
        for width, height in ((1280, 720), (720, 1280)):
            with self.subTest(frame=(width, height)):
                near_wrist = (hand[0, 0] / width + 0.005, hand[0, 1] / height)
                far_wrist = (0.1, 0.1)
                elbow = match_elbow(hand, [((0.2, 0.2), far_wrist), ((0.5, 0.9), near_wrist)], (width, height))
                np.testing.assert_allclose(elbow, [0.5 * width, 0.9 * height], atol=1e-3)

    def test_an_arm_too_far_from_the_hand_is_not_used(self):
        # The other person's arm, or the other arm, must not be paired.
        from features import match_elbow
        coords, _ = _hand_with_arm(0.0, 0.0, frames=1)
        self.assertIsNone(match_elbow(coords[0], [((0.0, 0.0), (0.05, 0.05))], (1280, 720)))


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


class TestLongGaps(unittest.TestCase):
    # Flow report H2: fill_missing bridges any gap, so a hand that leaves the
    # frame for seconds still shows smooth "motion" that can be counted.

    def test_only_gaps_longer_than_the_cap_are_marked(self):
        from features import long_gaps
        detected = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(long_gaps(detected, 3), [0, 0, 0, 0, 1, 1, 1, 1, 0])

    def test_frames_before_the_first_detection_count_as_a_gap(self):
        # They are filled with zeros, not with anything the hand did.
        from features import long_gaps
        np.testing.assert_array_equal(long_gaps(np.array([0, 0, 0, 1], dtype=bool), 2), [1, 1, 1, 0])

    def test_a_gap_separates_two_strokes_of_the_same_kind(self):
        # Without it, two strokes on either side of a blind spot merge into one.
        from inference import GAP_LABEL, MIN_RUN_WINDOWS, stroke_spans
        run = [0] * MIN_RUN_WINDOWS
        self.assertEqual(len(stroke_spans(run + [GAP_LABEL] * MIN_RUN_WINDOWS + run)), 2)
        self.assertEqual(len(stroke_spans([GAP_LABEL] * 10)), 0)

    def test_gap_windows_line_up_with_the_predicted_windows(self):
        # One flag per window as predict_window_labels makes them, also when a
        # 60fps upload is resampled to 30fps.
        from inference import WINDOW_SIZE, gap_windows
        detected = np.ones(120, dtype=bool)
        detected[50:70] = False  # 20 frames: past the cap at both rates
        at_30 = gap_windows(detected, 30.0)
        self.assertEqual(len(at_30), 120 - WINDOW_SIZE + 1)
        self.assertTrue(at_30[50] and not at_30[0] and not at_30[-1])
        detected_60 = np.ones(240, dtype=bool)
        detected_60[50:70] = False  # 0.33 s, frames 25-35 after resampling
        at_60 = gap_windows(detected_60, 60.0)
        self.assertEqual(len(at_60), 120 - WINDOW_SIZE + 1)  # 4 s at 30fps
        self.assertTrue(at_60[30] and not at_60[36] and not at_60[-1])  # window i covers frames i to i+29


class TestFeatureOptions(unittest.TestCase):
    # Flow report F2, F4, F5: training options, off by default, applied the
    # same way at inference through features.window_inputs.

    def test_backward_differences_give_the_last_frame_its_motion(self):
        # F2: a window is labeled by its last frame and the LSTM output is read
        # there, yet the thesis layout leaves that frame's velocity at 0.
        from features import window_features
        positions = np.zeros((6, 21, 3), dtype=np.float32)
        positions[:, 8, 0] = np.arange(6) ** 2
        forward, _ = window_features(positions)
        backward, _ = window_features(positions, backward=True)
        self.assertEqual(forward[-1, 8, 3], 0.0)
        self.assertEqual(backward[-1, 8, 3], 9.0)    # 25 - 16
        self.assertEqual(backward[-1, 8, 6], 2.0)
        self.assertFalse(backward[0, :, 3:].any())   # padding moved to the start

    def test_palm_scale_makes_camera_distance_irrelevant(self):
        # F4: a hand twice as close is twice as big in pixels, in depth too;
        # in palm lengths it is the same hand. With z left out, a third of the
        # channels (z and its velocity and acceleration) still told the model
        # how far the camera was.
        import features
        rng = np.random.default_rng(4)
        near = rng.uniform(100, 300, size=(10, 21, 3)).astype(np.float32)
        far = near * 0.5
        results = []
        original = features.extract_raw_tracks
        try:
            for raw in (near, far):
                features.extract_raw_tracks = lambda *a, raw=raw: ({"left": raw, "right": raw},
                                                                   {h: np.full((10, 2), np.nan) for h in ("left", "right")})
                results.append(features.extract_hand_tracks("unused", palm_scale=True)[0]["left"])
        finally:
            features.extract_raw_tracks = original
        np.testing.assert_allclose(results[0], results[1], atol=1e-4)

    def test_depth_is_put_on_the_scale_of_x_only_on_request(self):
        # MediaPipe reports z as a fraction of the image width, so palm_scale
        # needs it in pixels like x before dividing by the palm length (a real
        # photo shrunk to half size gave z 0.57x, x 0.52x). Without the option
        # z must stay as reported: existing checkpoints were trained on that.
        from types import SimpleNamespace
        from features import detect_hand_coords

        class Tracker:
            def process(self, image_rgb):
                return ["Left"], [0.9], [[SimpleNamespace(x=0.5, y=0.5, z=0.01 * i) for i in range(21)]]

        image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 1280x720 in reference pixels
        reported = np.array([0.01 * i for i in range(21)], dtype=np.float32)
        np.testing.assert_allclose(detect_hand_coords(image, Tracker())["left"][:, 2], reported, atol=1e-6)
        np.testing.assert_allclose(detect_hand_coords(image, Tracker(), z_pixels=True)["left"][:, 2], reported * 1280,
                                   rtol=1e-5)

    def test_mirrored_right_hand_looks_like_a_left_hand(self):
        # F5: the same stroke played by the right hand, seen in a mirror, must
        # reach the model exactly as the left hand's does, in the same slot.
        from features import window_inputs
        rng = np.random.default_rng(5)
        left = rng.normal(size=(30, 21, 3)).astype(np.float32)
        right = left * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        left_states = rng.normal(size=(30, 2)).astype(np.float32)
        right_states = left_states * np.array([1.0, -1.0], dtype=np.float32)  # the wrist angle flips too
        a = window_inputs(left, left_states, "left", mirror_hands=True)
        b = window_inputs(right, right_states, "right", mirror_hands=True)
        self.assertEqual((a[3], b[3]), ("left", "left"))
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_allclose(x, y, atol=1e-5)
        self.assertEqual(window_inputs(right, right_states, "right")[3], "right")  # off by default

    def test_checkpoint_applies_the_feature_options_it_was_trained_with(self):
        import os, tempfile
        from features import FeatureScaler
        from model import StrokeBiLSTMModel
        config = {"backward_diff": True, "palm_scale": True, "mirror_hands": True}
        scaler = FeatureScaler(*[np.zeros(9, dtype=np.float32)] * 4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.pt")
            torch.save({"model_state": StrokeBiLSTMModel(7).state_dict(), "scaler": scaler.state_dict(),
                        "labels": list(range(7)), "hand_to_slot": {"left": "left", "right": "right"},
                        "config": config}, path)
            loaded, _, _ = load_checkpoint(path, torch.device("cpu"))
        self.assertEqual((loaded.backward_diff, loaded.palm_scale, loaded.mirror_hands), (True, True, True))
        old, _, _ = load_checkpoint("multi_motion_model_7class.pt", torch.device("cpu"))
        self.assertEqual((old.backward_diff, old.palm_scale, old.mirror_hands), (False, False, False))


class TestStrokeTimes(unittest.TestCase):
    def test_spans_become_seconds_from_the_start(self):
        # Flow report I1: counts alone cannot be drawn on a timeline or
        # checked against the video.
        from inference import stroke_times
        times = stroke_times([(0, 30, 44), (2, 90, 104)], 30.0)
        self.assertEqual(times[0], {"stroke": "Full", "start_s": 1.0, "end_s": 1.5})
        self.assertEqual(times[1]["stroke"], "Up")
        self.assertEqual(stroke_times([(1, 60, 74)], 60.0)[0]["start_s"], 1.0)  # the analysis frame rate decides
