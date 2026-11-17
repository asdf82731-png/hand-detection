"""
Check everything a long training run depends on, before starting it. A run of
run_experiments.py takes hours and a data mistake does not stop it: the 2026-11-16
run trained on a Down.csv whose holds were read as "none" and nobody saw until
the numbers were in. This takes seconds.

    python preflight.py --data-dir data
    python preflight.py --data-dir data --out-dir experiments_1117

Prints one line per finding. An ERROR means the run would train on the wrong
thing or fail (exit status 1); a WARNING is for a person to look at. Nothing
is changed.

Checked: the packages and whether a GPU is used; for each of Full, Down, Up and
Tab that its label CSV and its frames (or video) are there and the CSV's frame
numbers fit them; that the labels load; that no frame labeled "hold" in a
`motion_label` column ended up as "none" (the Down.csv trap); that a video with
strokes also has hold frames; that a video is not almost all "none" (movement
that was never labeled); the stroke order (check_labels.py); and free disk space.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys

from check_labels import check_file
from data_pipeline import load_frame_labels, parse_frame_id
from features import source_frame_count
from train import VIDEO_NAMES, data_sources

STROKE_LABELS = ("full_stroke", "down_stroke", "up_stroke", "tab_stroke")
# Under this share of stroke frames a video is probably mostly unlabeled movement.
MIN_STROKE_SHARE = 0.05
# 16 experiments x 5 seeds, one model file each (17 MB), plus logs and room to spare.
MIN_FREE_GB = 3.0


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def csv_label_counts(path):
    """
    Label CSV -> ({normalized label: rows}, holds read as none). The second is
    the number of rows whose `motion_label` column says hold but whose label is
    none: that column is where an older Down.csv kept its holds, and the loader
    reads only the label column.
    """
    from data_pipeline import normalize_label
    counts, hold_as_none = {}, 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        column = next((c for c in ("strike", "stroke") if c in (reader.fieldnames or [])), None)
        if column is None:
            raise ValueError(f"no label column (expected strike or stroke), got {reader.fieldnames}")
        for row in reader:
            label = normalize_label(row[column])
            counts[label] = counts.get(label, 0) + 1
            if label == "none" and (row.get("motion_label") or "").strip().lower() == "hold":
                hold_as_none += 1
    return counts, hold_as_none


def last_frame_available(source):
    """Highest frame number a frames folder or video holds (a video counts from 0)."""
    if os.path.isdir(source):
        numbers = [parse_frame_id(n) for n in os.listdir(source) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        return max(numbers) if numbers else -1
    return source_frame_count(source) - 1


def check_video(data_dir, name):
    """One of the four videos -> (errors, warnings, info lines)."""
    errors, warnings, info = [], [], []
    source, csv_path = data_sources(data_dir, name)
    if not os.path.exists(csv_path):
        return [f"{name}: label file {csv_path} is missing"], warnings, info
    if not os.path.exists(source):
        return [f"{name}: neither a {name}/ frames folder nor {name}.mp4 in {data_dir}"], warnings, info
    try:
        counts, hold_as_none = csv_label_counts(csv_path)
        frame_labels = load_frame_labels(csv_path)
    except ValueError as error:
        return [f"{name}: {csv_path}: {error}"], warnings, info

    total = sum(counts.values())
    strokes = sum(counts.get(label, 0) for label in STROKE_LABELS)
    holds = counts.get("up_hold", 0) + counts.get("down_hold", 0)
    info.append(f"{name}: {total} rows, {strokes} stroke / {holds} hold / {counts.get('none', 0)} none, sha256 {sha256(csv_path)[:12]}")
    if hold_as_none:
        errors.append(f"{name}: {hold_as_none} rows have motion_label 'hold' but are labeled none, so training reads them as none "
                      "(use the corrected file, e.g. Down_swapped_holdfix.csv, or relabel them down_hold / up_hold)")
    elif strokes and not holds:
        warnings.append(f"{name}: {strokes} stroke rows but no hold rows at all; are the holds labeled as none?")
    if total and strokes / total < MIN_STROKE_SHARE:
        warnings.append(f"{name}: only {strokes} of {total} rows ({strokes / total:.1%}) are strokes; "
                        "movement in this video may never have been labeled and would train as none")

    last_needed, last_have = max(frame_labels), last_frame_available(source)
    if last_needed > last_have:
        errors.append(f"{name}: the CSV labels frame {last_needed} but {source} only goes to frame {last_have}: "
                      "these labels are not for these frames")
    _, order_warnings = check_file(csv_path)
    if order_warnings:
        warnings.append(f"{name}: {order_warnings} stroke-order warning(s); run python check_labels.py {csv_path}")
    return errors, warnings, info


def check_environment(out_dir):
    """Packages, GPU and disk -> (errors, warnings, info lines)."""
    errors, warnings, info = [], [], []
    for module in ("torch", "numpy", "sklearn", "cv2", "mediapipe"):
        try:
            imported = __import__(module)
            info.append(f"{module} {getattr(imported, '__version__', '')}".strip())
        except ImportError:
            errors.append(f"package {module} is not installed (pip install -r requirements.txt)")
    try:
        import torch
        if torch.cuda.is_available():
            info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("no GPU is used (torch.cuda.is_available() is False): training will be much slower")
    except ImportError:
        pass
    folder = os.path.abspath(out_dir)
    while not os.path.exists(folder):
        folder = os.path.dirname(folder)
    free = shutil.disk_usage(folder).free / 1e9
    if free < MIN_FREE_GB:
        errors.append(f"only {free:.1f} GB free where {out_dir} goes; the models and logs need about {MIN_FREE_GB:.0f} GB")
    else:
        info.append(f"{free:.0f} GB free for {out_dir}")
    return errors, warnings, info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="experiments", help="where run_experiments.py will write, for the disk check")
    args = parser.parse_args()

    errors, warnings, info = check_environment(args.out_dir)
    for name in VIDEO_NAMES:
        e, w, i = check_video(args.data_dir, name)
        errors += e
        warnings += w
        info += i
    for line in info:
        print(f"  ok       {line}")
    for line in warnings:
        print(f"  WARNING  {line}")
    for line in errors:
        print(f"  ERROR    {line}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("Do not start the run until the errors are fixed.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
