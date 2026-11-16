"""
Tests for run_experiments.py's comparison table. The table is what decides
which options are kept, so a wrong sign or a hidden missing class would steer
the retraining the wrong way.
"""
import unittest

from run_experiments import EXPERIMENTS, results_table


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

    def test_every_option_row_changes_one_thing_from_the_default(self):
        names = [name for name, _ in EXPERIMENTS]
        self.assertEqual(names[0], "baseline")
        self.assertEqual(EXPERIMENTS[0][1], [])
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
