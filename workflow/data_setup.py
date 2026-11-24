"""
What every stage (check_data, preflight, train, baselines, run_experiments,
evaluate) must agree on about the data: which videos there are, where their
frames and labels are, which hand slot each CSV `hand` value means, and the
content hash that identifies a label file.

These lived in train.py, so a stage that only needed a path (check_data,
preflight) had to import torch through it. train.py still re-exports them.
"""
import argparse
import hashlib
import os

VIDEO_NAMES = ["Full", "Down", "Up", "Tab"]


def file_sha256(path):
    """Content hash of a label CSV; evaluate.py compares it with the evaluation CSVs."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def data_sources(data_dir, name):
    """(frames source, csv path) for one action: a frame folder if present, else the video."""
    frames_dir = os.path.join(data_dir, name)
    source = frames_dir if os.path.isdir(frames_dir) else os.path.join(data_dir, f"{name}.mp4")
    return source, os.path.join(data_dir, f"{name}.csv")


def hand_mapping(csv_left_slot):
    """
    Which tracker slot (features.HANDS) each CSV `hand` value refers to. A
    swapped mapping trains every sample on the other hand's label without any
    error, so it has no default: check_data.py works it out from the data. It
    is saved in the checkpoint so inference reports left/right the same way.
    """
    return {"left": csv_left_slot, "right": "right" if csv_left_slot == "left" else "left"}


def csv_left_slot_spec(text):
    """
    --csv-left-slot value -> (default slot, {video name: slot}): "left" or
    "right", optionally followed by exceptions for single videos, e.g.
    "left,Down=right". One mapping for all videos trains a video whose CSV
    names the hands the other way on swapped labels (flow report D3).
    """
    default, *exceptions = [part.strip() for part in text.split(",")]
    overrides = {}
    for exception in exceptions:
        name, _, slot = exception.partition("=")
        if name not in VIDEO_NAMES or slot not in ("left", "right"):
            raise argparse.ArgumentTypeError(f"{exception!r}: expected <video>=<left|right>, video one of {VIDEO_NAMES}")
        overrides[name] = slot
    if default not in ("left", "right"):
        raise argparse.ArgumentTypeError(f"{default!r}: expected left or right first")
    return default, overrides


def video_hand_mapping(spec, name):
    """The hand_mapping for one video under a csv_left_slot_spec."""
    default, overrides = spec
    return hand_mapping(overrides.get(name, default))
