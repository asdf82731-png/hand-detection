"""
Tests for run_experiments.py's comparison table. The table is what decides
which options are kept, so a wrong sign or a hidden missing class would steer
the retraining the wrong way.
"""
import json
import os
import sys
import tempfile
import unittest

from run_experiments import (
    BASELINE_ROWS, COMPARED_WITH, EXPERIMENTS, baselines_done, chosen_experiments, claim_out_dir, results_table, run,
)


def _metrics(acc, f1, up=None):
    return {"accuracy": acc, "macro_f1": f1,
            "f1": {"full_stroke": 0.9, "down_stroke": 0.8, "up_stroke": up, "tab_stroke": 0.7}}


class TestResultsTable(unittest.TestCase):
    def setUp(self):
        self.table = results_table(
            {"baseline": [_metrics(0.80, 0.50), _metrics(0.82, 0.54)], "hand_state": [_metrics(0.85, 0.60)]},
            {"majority": [{"accuracy": 0.6, "macro_f1": 0.1}]},
        )
        self.rows = {line.split("|")[1].strip(): line for line in self.table.splitlines()[2:]}

    def test_change_is_measured_against_the_baseline_mean(self):
        self.assertIn("+0.080", self.rows["hand_state"])  # 0.60 - mean(0.50, 0.54)

    def test_spread_across_seeds_is_shown(self):
        # Without it, noise between seeds looks like an effect.
        self.assertIn("0.520 ± 0.028", self.rows["baseline"])

    def test_a_class_missing_from_validation_shows_as_missing_not_zero(self):
        # Up has no validation windows today; 0.000 would read as "fails".
        self.assertEqual(self.rows["hand_state"].split("|")[8].strip(), "-")

    def test_baselines_are_listed_for_comparison(self):
        self.assertIn("(majority)", self.rows)

    def test_every_row_differs_by_one_option_from_the_row_it_is_compared_with(self):
        # A row that changes two things gives a difference nobody can assign
        # to either: class weights also select the epoch by macro-F1, which
        # raises macro-F1 by itself, so they are compared with a row that
        # only does that.
        arguments = dict(EXPERIMENTS)
        self.assertEqual(len(arguments), len(EXPERIMENTS))
        self.assertEqual(arguments["baseline"], [])
        for name, extra in EXPERIMENTS[1:]:
            reference = arguments[COMPARED_WITH.get(name, "baseline")]
            self.assertEqual(extra[len(extra) - len(reference):], reference, name)
            added = [a for a in extra[:len(extra) - len(reference)] if a.startswith("--")]
            self.assertEqual(len(added), 1, f"{name} adds {added}")

    def test_class_weights_are_measured_against_macro_f1_selection(self):
        table = results_table(
            {"baseline": [_metrics(0.80, 0.50)], "select_macro_f1": [_metrics(0.78, 0.58)],
             "class_weights": [_metrics(0.77, 0.60)]}, {})
        row = next(line for line in table.splitlines() if line.startswith("| class_weights"))
        self.assertIn("+0.020 vs select_macro_f1", row)  # not +0.100, most of which the selection gives
        self.assertNotIn("+0.100", row)

    def test_asking_for_one_row_also_runs_the_row_it_is_compared_with(self):
        # Otherwise its change column has nothing to be measured against.
        self.assertEqual([name for name, _ in chosen_experiments(["class_weights"])],
                         ["baseline", "select_macro_f1", "class_weights"])
        self.assertEqual(chosen_experiments(None), EXPERIMENTS)

    def test_options_tried_on_top_of_palm_scale_are_measured_against_it(self):
        # palm_scale gains +0.05 on its own; measured against "baseline", the
        # rows built on it would all look like wins whatever their own option does.
        for name in ("palm_no_output_gelu", "palm_lstm_input_128"):
            self.assertEqual(COMPARED_WITH[name], "palm_scale")
            self.assertIn("--palm-scale", dict(EXPERIMENTS)[name])
        self.assertEqual([n for n, _ in chosen_experiments(["palm_lstm_input_128"])],
                         ["baseline", "palm_scale", "palm_lstm_input_128"])


class TestRuns(unittest.TestCase):
    def test_runs_start_in_the_callers_folder(self):
        # --data-dir data must mean the same folder to train.py as to the
        # person who typed it; started in the project folder instead, every
        # video was skipped as missing.
        here = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertTrue(run([sys.executable, "-c", "import os; print(os.getcwd())"], "run.log"))
                with open("run.log", encoding="utf-8") as log:
                    self.assertEqual(os.path.realpath(log.read().strip()), os.path.realpath(folder))
            finally:
                os.chdir(here)

    def test_baseline_results_from_before_a_row_existed_are_redone(self):
        # An existing file is skipped; one written before "gru + palm_scale"
        # existed would leave that row out of the table with no sign of why.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "baselines_seed0.json")
            self.assertFalse(baselines_done(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump({row: {} for row in BASELINE_ROWS if row != "gru + palm_scale"}, f)
            self.assertFalse(baselines_done(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump({row: {} for row in BASELINE_ROWS}, f)
            self.assertTrue(baselines_done(path))

    def test_an_output_folder_is_not_reused_for_other_settings(self):
        # Finished runs are skipped, so after a change of --csv-left-slot the
        # table would show the old models under the new value.
        settings = {"data_dir": "/data", "csv_left_slot": ["right", {"Down": "left"}]}
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "experiments")
            claim_out_dir(out, settings)
            claim_out_dir(out, json.loads(json.dumps(settings)))  # the same command again: resumes
            with self.assertRaises(ValueError):
                claim_out_dir(out, {**settings, "csv_left_slot": ["right", {}]})

    def test_results_with_no_record_of_their_settings_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, "baseline_seed0.pt"), "w").close()
            with self.assertRaises(ValueError):
                claim_out_dir(folder, {"data_dir": "/data", "csv_left_slot": ["left", {}]})


if __name__ == "__main__":
    unittest.main()
