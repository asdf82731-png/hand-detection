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
--out-dir remembers the --data-dir and --csv-left-slot it was started with
(settings.json) and refuses others, so its table never mixes models trained
on different data or hand mappings.
Each train.py run reads the videos and detects the hands again, so one run
takes minutes: 16 experiments x 5 seeds is a long job.

Results go to <out-dir>/results.md. With one seed, a difference between two
rows can be noise; use several seeds and trust only differences larger than
the spread across seeds (TRAINING_DESIGN.md, REV-7).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

from train import VIDEO_NAMES, csv_left_slot_spec, data_sources

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, extra train.py arguments). "baseline" is the current default model;
# every other row changes one thing from the row it is compared with
# (COMPARED_WITH, else "baseline"), so its effect can be read off directly.
EXPERIMENTS = [
    ("baseline", []),
    ("development", ["--embedding", "development"]),      # path development, the thesis's Lie group claim
    ("hand_state", ["--hand-state"]),                      # wrist height and angle (thesis table 4.2)
    ("time_stretch", ["--time-stretch"]),
    ("select_macro_f1", ["--select-by", "macro_f1"]),
    ("class_weights", ["--class-weights", "--select-by", "macro_f1"]),  # T1
    ("backward_diff", ["--backward-diff"]),                # F2
    ("palm_scale", ["--palm-scale"]),                      # F4
    ("mirror_hands", ["--mirror-hands"]),                  # F5
    ("lstm_input_128", ["--lstm-input", "128"]),           # M1
    ("bidir_ends", ["--bidir-ends"]),                      # M2
    ("no_output_gelu", ["--no-output-gelu"]),              # M3
    # palm_scale was the only option clearly above the seed-to-seed spread
    # (2026-11-17), so the options that leaned positive are tried on top of it.
    ("palm_no_output_gelu", ["--no-output-gelu", "--palm-scale"]),
    ("palm_lstm_input_128", ["--lstm-input", "128", "--palm-scale"]),
    # Ablations on top of palm_scale: does the result need the bone path, the graph layer?
    ("palm_no_bones", ["--no-bones", "--palm-scale"]),
    ("palm_no_gcn", ["--no-gcn", "--palm-scale"]),
]
# Class weights are meant to go with macro-F1 selection, and choosing the
# epoch by macro-F1 raises macro-F1 by itself: against "baseline" that gain
# would be credited to the weights.
COMPARED_WITH = {"class_weights": "select_macro_f1", "palm_no_output_gelu": "palm_scale",
                 "palm_lstm_input_128": "palm_scale", "palm_no_bones": "palm_scale", "palm_no_gcn": "palm_scale"}
# baselines.py rows a baselines_seed*.json must hold to count as done: a file
# written before a row existed is redone instead of silently lacking it.
BASELINE_ROWS = ("majority", "rule", "rule + type", "gru", "gru + palm_scale")
STROKES = ["full_stroke", "down_stroke", "up_stroke", "tab_stroke"]


def run(command, log_path):
    """
    Run a command with its output going to log_path; True if it succeeded.
    It runs in the caller's folder, so a relative --data-dir or --out-dir
    means the same place to it as to this script.
    """
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode == 0


def chosen_experiments(only):
    """EXPERIMENTS limited to the names in `only` (all if empty), plus "baseline" and the rows they are compared with."""
    names = set(only or [name for name, _ in EXPERIMENTS]) | {"baseline"}
    names |= {COMPARED_WITH[name] for name in names if name in COMPARED_WITH}
    return [(name, extra) for name, extra in EXPERIMENTS if name in names]


def baselines_done(path):
    """True if path holds every BASELINE_ROWS row."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        found = json.load(f)
    return all(row in found for row in BASELINE_ROWS)


def run_settings(data_dir, spec):
    """
    What this output folder's results are for: the data folder, the hand mapping
    and a hash of each label CSV. The hashes are there because a label file can
    change under the same name (Down.csv was replaced by a corrected one), and
    finished runs are reused, so old models would otherwise sit in the table
    under the new labels.
    """
    hashes = {}
    for name in VIDEO_NAMES:
        csv_path = data_sources(data_dir, name)[1]
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                hashes[name] = hashlib.sha256(f.read()).hexdigest()
    return {"data_dir": os.path.abspath(data_dir), "csv_left_slot": list(spec), "csv_sha256": hashes}


def claim_out_dir(out_dir, settings):
    """
    Make out_dir and record the settings its results are for, or raise
    ValueError if it already holds results for other settings (or from before
    they were recorded): finished runs are reused, so the table would show
    old models under the new settings.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "settings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            found = json.load(f)
        if found != settings:
            raise ValueError(f"{out_dir} holds results for {found}, not {settings}: use another --out-dir or delete it")
    elif any(name.endswith((".pt", ".json")) for name in os.listdir(out_dir)):
        raise ValueError(f"{out_dir} holds results with no record of their settings: use another --out-dir or delete it")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)


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
    and each experiment's macro-F1 change against the row it is compared with
    ("baseline" unless COMPARED_WITH names another, which the cell then says).
    """
    header = ["experiment", "seeds", "accuracy", "macro-F1", "Δ macro-F1"] + [f"F1 {s.split('_')[0]}" for s in STROKES]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, runs in model_rows.items():
        if not runs:
            continue
        reference = COMPARED_WITH.get(name, "baseline")
        if name == "baseline" or not model_rows.get(reference):
            delta = "-"
        else:
            delta = f'{np.mean([m["macro_f1"] for m in runs]) - np.mean([m["macro_f1"] for m in model_rows[reference]]):+.3f}'
            delta += "" if reference == "baseline" else f" vs {reference}"
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
                        help="run only these experiments (baseline and the rows they are compared with are always included)")
    parser.add_argument("--skip-baselines", action="store_true", help="do not run baselines.py / detection_by_label.py")
    args = parser.parse_args()
    try:
        spec = csv_left_slot_spec(args.csv_left_slot)  # refuse a bad value before hours of training
    except argparse.ArgumentTypeError as error:
        parser.error(f"--csv-left-slot: {error}")
    try:
        claim_out_dir(args.out_dir, run_settings(args.data_dir, spec))
    except ValueError as error:
        parser.error(f"--out-dir: {error}")

    experiments = chosen_experiments(args.only)
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
            if not baselines_done(out):
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
