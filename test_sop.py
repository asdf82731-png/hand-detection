"""
Tests for sop.py. It exists so the retraining flow runs in one fixed order and a failed
stage stops it: training on data preflight rejected, or on a hand mapping nobody chose,
wastes hours without any error. Nothing trains here; the subprocess call is replaced.
"""
import unittest
from unittest import mock

import sop


class TestSop(unittest.TestCase):
    def _run(self, argv, errors=()):
        with mock.patch.object(sop, "run_checks", return_value=(list(errors), [], [])) as checks, \
                mock.patch.object(sop.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            code = sop.main(argv)
        return code, checks, run

    def test_a_failed_preflight_stops_before_anything_trains(self):
        code, _, run = self._run(["--csv-left-slot", "left"], errors=["Down: label file is missing"])
        self.assertEqual(code, 1)
        run.assert_not_called()

    def test_options_it_does_not_know_go_to_run_experiments(self):
        code, _, run = self._run(["--csv-left-slot", "left,Down=right", "--seeds", "0", "1", "--only", "palm_gru"])
        self.assertEqual(code, 0)
        command = run.call_args[0][0]
        self.assertEqual(command[command.index("--csv-left-slot") + 1], "left,Down=right")
        self.assertEqual(command[command.index("--seeds"):][:3], ["--seeds", "0", "1"])
        self.assertIn("--only", command)

    def test_preflight_is_not_run_twice(self):
        # run_experiments checks on its own unless told not to; sop already did.
        _, _, run = self._run(["--csv-left-slot", "left"])
        self.assertIn("--skip-preflight", run.call_args[0][0])

    def test_skip_preflight_runs_no_checks_but_still_trains(self):
        code, checks, run = self._run(["--csv-left-slot", "left", "--skip-preflight"], errors=["would block"])
        self.assertEqual(code, 0)
        checks.assert_not_called()
        run.assert_called_once()

    def test_the_hand_mapping_must_be_given(self):
        # A wrong mapping fails silently, so there is no default and no guess.
        with self.assertRaises(SystemExit) as raised:
            sop.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_a_failed_training_stage_is_reported_as_failure(self):
        with mock.patch.object(sop, "run_checks", return_value=([], [], [])), \
                mock.patch.object(sop.subprocess, "run", return_value=mock.Mock(returncode=1)):
            self.assertEqual(sop.main(["--csv-left-slot", "left"]), 1)


if __name__ == "__main__":
    unittest.main()
