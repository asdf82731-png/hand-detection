"""
Run the retraining comparison in one go: a baseline with the current
defaults, then each option on its own, the baselines (REV-6) and the
detection analysis (REV-9), and one table that compares them.

    python run_experiments.py --data-dir data --csv-left-slot left,Down=right
    python run_experiments.py --data-dir data --csv-left-slot left,Down=right --seeds 0 1 2 3 4

Every run saves its own model in --out-dir (experiments/ by default); the
app's multi_motion_model_7class.pt is never touched. To use one of the new
models in the app, copy it over that file yourself.

A run whose model file already exists is not repeated, so an interrupted
run can be restarted with the same command; delete the file to redo it.
Each train.py run reads the videos and detects the hands again, so one run
takes minutes: 11 experiments x 5 seeds is a long job.

Results go to <out-dir>/results.md. With one seed, a difference between two
rows can be noise; use several seeds and trust only differences larger than
the spread across seeds (TRAINING_DESIGN.md, REV-7).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from train import csv_left_slot_spec

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, extra train.py arguments). "baseline" is the current default model;
# every other row changes one thing so its effect can be read off directly.
EXPERIMENTS = [
    ("baseline", []),
    ("development", ["--embedding", "development"]),      # path development, the thesis's Lie group claim
    ("hand_state", ["--hand-state"]),                      # wrist height and angle (thesis table 4.2)
    ("time_stretch", ["--time-stretch"]),
    ("class_weights", ["--class-weights", "--select-by", "macro_f1"]),  # T1
    ("backward_diff", ["--backward-diff"]),                # F2
    ("palm_scale", ["--palm-scale"]),                      # F4
    ("mirror_hands", ["--mirror-hands"]),                  # F5
    ("lstm_input_128", ["--lstm-input", "128"]),           # M1
    ("bidir_ends", ["--bidir-ends"]),                      # M2
    ("no_output_gelu", ["--no-output-gelu"]),              # M3
]
STROKES = ["full_stroke", "down_stroke", "up_stroke", "tab_stroke"]


def run(command, log_path):
    """Run a command with its output going to log_path; True if it succeeded."""
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
    return result.returncode == 0


def train_command(args, extra, seed, out_path):
    return [sys.executable, os.path.join(HERE, "train.py"), "--data-dir", args.data_dir,
            "--csv-left-slot", args.csv_left_slot, "--seed", str(seed), "--out", out_path, *extra]


def mean_sd(values):
    """'0.812 ± 0.021' over the non-None values ('0.812' for one, '-' for none)."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def results_table(model_rows, baseline_rows):
    """
    model_rows: {experiment: [val_metrics per seed]} (train.py's checkpoint
    "val_metrics"); baseline_rows: {baseline name: [{"accuracy", "macro_f1"} per seed]}.
    -> Markdown: one row per experiment and baseline, mean ± sd over seeds,
    and each experiment's macro-F1 change against "baseline".
    """
    base_f1 = np.mean([m["macro_f1"] for m in model_rows["baseline"]]) if model_rows.get("baseline") else None
    header = ["experiment", "seeds", "accuracy", "macro-F1", "Δ macro-F1"] + [f"F1 {s.split('_')[0]}" for s in STROKES]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, runs in model_rows.items():
        if not runs:
            continue
        f1 = np.mean([m["macro_f1"] for m in runs])
        delta = "-" if base_f1 is None or name == "baseline" else f"{f1 - base_f1:+.3f}"
        cells = [name, str(len(runs)), mean_sd([m["accuracy"] for m in runs]), mean_sd([m["macro_f1"] for m in runs]),
                 delta] + [mean_sd([m["f1"].get(s) for m in runs]) for s in STROKES]
        lines.append("| " + " | ".join(cells) + " |")
    for name, runs in baseline_rows.items():
        cells = [f"({name})", str(len(runs)), mean_sd([r["accuracy"] for r in runs]),
                 mean_sd([r["macro_f1"] for r in runs]), "-"] + ["-"] * len(STROKES)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-left-slot", required=True, help="as for train.py, e.g. left,Down=right")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--only", nargs="+", choices=[name for name, _ in EXPERIMENTS],
                        help="run only these experiments (baseline is always included)")
    parser.add_argument("--skip-baselines", action="store_true", help="do not run baselines.py / detection_by_label.py")
    args = parser.parse_args()
    try:
        csv_left_slot_spec(args.csv_left_slot)  # refuse a bad value before hours of training
    except argparse.ArgumentTypeError as error:
        parser.error(f"--csv-left-slot: {error}")
    os.makedirs(args.out_dir, exist_ok=True)

    experiments = [(n, e) for n, e in EXPERIMENTS if not args.only or n == "baseline" or n in args.only]
    failed, model_rows, baseline_rows = [], {}, {}

    if not args.skip_baselines:
        log = os.path.join(args.out_dir, "detection_by_label.log")
        print(f"[detection] -> {log}")
        if not run([sys.executable, os.path.join(HERE, "detection_by_label.py"), "--data-dir", args.data_dir,
                    "--csv-left-slot", args.csv_left_slot], log):
            failed.append(("detection_by_label", log))

    for seed in args.seeds:
        if not args.skip_baselines:
            out = os.path.join(args.out_dir, f"baselines_seed{seed}.json")
            log = out.replace(".json", ".log")
            if not os.path.exists(out):
                print(f"[baselines seed {seed}] -> {log}")
                if not run([sys.executable, os.path.join(HERE, "baselines.py"), "--data-dir", args.data_dir,
                            "--csv-left-slot", args.csv_left_slot, "--seed", str(seed), "--out", out], log):
                    failed.append((f"baselines seed {seed}", log))
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    for name, scores in json.load(f).items():
                        baseline_rows.setdefault(name, []).append(scores)

        for name, extra in experiments:
            out = os.path.join(args.out_dir, f"{name}_seed{seed}.pt")
            log = out.replace(".pt", ".log")
            if os.path.exists(out):
                print(f"[{name} seed {seed}] already done: {out}")
            else:
                print(f"[{name} seed {seed}] training -> {log}")
                if not run(train_command(args, extra, seed, out), log):
                    failed.append((f"{name} seed {seed}", log))
                    continue
            metrics = torch.load(out, map_location="cpu", weights_only=False).get("val_metrics")
            if metrics is None:
                failed.append((f"{name} seed {seed}: model has no val_metrics (made before this script?)", out))
                continue
            model_rows.setdefault(name, []).append(metrics)

    table = results_table(model_rows, baseline_rows)
    seeds = " ".join(map(str, args.seeds))
    report = (f"# Retraining comparison\n\n--csv-left-slot {args.csv_left_slot}, seeds {seeds}. "
              "Validation windows from train.py's time split; mean ± sd over seeds. "
              "Rows in parentheses are baselines.py. Trust only differences larger than the spread across seeds.\n\n"
              f"{table}\n")
    if failed:
        report += "\n## Failed\n\n" + "\n".join(f"- {what}: see {where}" for what, where in failed) + "\n"
    path = os.path.join(args.out_dir, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"written to {path}")
    if failed:
        sys.exit(f"{len(failed)} run(s) failed; see the logs listed above.")


if __name__ == "__main__":
    main()
