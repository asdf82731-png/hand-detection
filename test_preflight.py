"""
Tests for preflight.py. It exists because a wrong label file does not stop a
training run, it only makes the hours of training worth less, so each test is
a mistake that was made or could be made and what the check must say about it.
"""
import os
import tempfile
import unittest

from preflight import check_video, csv_label_counts


def _write_csv(path, rows, with_motion_label=False):
    header = "frame_id,hand,motion_label,stroke" if with_motion_label else "frame_id,hand,strike"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")


def _frames(folder, count):
    os.makedirs(folder)
    for number in range(count):
        open(os.path.join(folder, f"frame_{number:04d}.jpg"), "w").close()


def _video(data_dir, name, labels, frames=100, with_motion_label=False):
    """labels: [(label, number of frames)] for the left hand; the right hand rests low."""
    rows, frame = [], 0
    for label, count in labels:
        for _ in range(count):
            name_ = f"frame_{frame:04d}.jpg"
            if with_motion_label:
                motion = "hold" if label in ("none", "down_hold", "up_hold") else "down"
                rows += [(name_, "left", motion, label), (name_, "right", "hold", "down_hold")]
            else:
                rows += [(name_, "left", label), (name_, "right", "down_hold")]
            frame += 1
    _write_csv(os.path.join(data_dir, f"{name}.csv"), rows, with_motion_label)
    _frames(os.path.join(data_dir, name), frames)


class TestPreflightVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_clean_video_gives_no_error_and_no_warning(self):
        _video(self.dir, "Tab", [("down_hold", 30), ("tab_stroke", 12), ("down_hold", 30), ("tab_stroke", 12), ("down_hold", 16)])
        errors, warnings, _ = check_video(self.dir, "Tab")
        self.assertEqual((errors, warnings), ([], []))

    def test_holds_stored_in_motion_label_but_labeled_none_are_an_error(self):
        # The Down.csv of 2026-11-12: motion_label says hold, the label column says none,
        # the loader reads only the label column, and the whole run trained on it.
        _video(self.dir, "Down", [("none", 30), ("down_stroke", 15), ("none", 30), ("down_stroke", 15), ("none", 10)],
               with_motion_label=True)
        errors, _, _ = check_video(self.dir, "Down")
        self.assertEqual(len(errors), 1)
        self.assertIn("motion_label 'hold'", errors[0])

    def test_the_same_video_with_holds_fixed_is_clean(self):
        _video(self.dir, "Down", [("down_hold", 30), ("down_stroke", 15), ("down_hold", 30), ("down_stroke", 15), ("down_hold", 10)],
               with_motion_label=True)
        errors, warnings, _ = check_video(self.dir, "Down")
        self.assertEqual((errors, warnings), ([], []))

    def test_strokes_with_no_holds_at_all_are_a_warning(self):
        # Without a motion_label column the trap cannot be seen directly; no holds in a
        # video that has strokes is the sign of it.
        _write_csv(os.path.join(self.dir, "Full.csv"),
                   [(f"frame_{n:04d}.jpg", "left", "full_stroke" if n % 20 < 10 else "none") for n in range(100)])
        _frames(os.path.join(self.dir, "Full"), 100)
        _, warnings, _ = check_video(self.dir, "Full")
        self.assertTrue(any("no hold rows" in w for w in warnings), warnings)

    def test_a_video_that_is_almost_all_none_is_a_warning(self):
        # The Up.csv: 15 stroke rows among thousands of none rows, the movement never labeled.
        _video(self.dir, "Up", [("none", 190), ("up_stroke", 2), ("down_hold", 8)], frames=200)
        _, warnings, _ = check_video(self.dir, "Up")
        self.assertTrue(any("never have been labeled" in w for w in warnings), warnings)

    def test_labels_past_the_last_frame_are_an_error(self):
        _video(self.dir, "Full", [("up_hold", 30), ("full_stroke", 10), ("up_hold", 30)], frames=50)
        errors, _, _ = check_video(self.dir, "Full")
        self.assertTrue(any("only goes to frame 49" in e for e in errors), errors)

    def test_a_missing_label_file_or_frames_is_an_error_not_a_skip(self):
        # train.py skips a missing video with one printed line, and the model then does not know that action.
        self.assertIn("is missing", check_video(self.dir, "Tab")[0][0])
        _write_csv(os.path.join(self.dir, "Tab.csv"), [("frame_0000.jpg", "left", "down_hold")])
        self.assertIn("neither", check_video(self.dir, "Tab")[0][0])

    def test_an_unknown_label_is_an_error(self):
        _write_csv(os.path.join(self.dir, "Tab.csv"), [("frame_0000.jpg", "left", "tap_stroke")])
        _frames(os.path.join(self.dir, "Tab"), 5)
        self.assertEqual(len(check_video(self.dir, "Tab")[0]), 1)

    def test_label_counts_use_the_names_the_loader_uses(self):
        # The older CSVs write full_strike / tab; the counts must see them as strokes.
        path = os.path.join(self.dir, "x.csv")
        _write_csv(path, [("frame_0000.jpg", "left", "full_strike"), ("frame_0001.jpg", "left", "tab")])
        self.assertEqual(csv_label_counts(path)[0], {"full_stroke": 1, "tab_stroke": 1})


if __name__ == "__main__":
    unittest.main()
