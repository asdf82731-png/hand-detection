"""
Train the 7-class stroke model with the thesis method (see TRAINING_DESIGN.md).

    python train.py --data-dir data

`data/` holds <Name>.mp4 and <Name>.csv for each of Full/Down/Up/Tab; videos
without a matching CSV are skipped (and reported). Writes
multi_motion_model_7class.pt: model weights, feature scaler and label order.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import LABELS, build_labeled_windows, load_frame_labels, split_train_val, windows_to_samples
from features import FeatureScaler, extract_hand_positions, pack_model_input, window_features
from model import StrokeBiLSTMModel

VIDEO_NAMES = ["Full", "Down", "Up", "Tab"]
WINDOW_SIZE = 30
TRAIN_STEP = 1  # thesis A23: dense windows; the split buffer still keeps train and val apart
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "multi_motion_model_7class.pt"

# Which tracker slot (features.HANDS) each CSV `hand` value refers to. Must be
# confirmed against the real CSVs before training — a swapped mapping trains
# every sample on the other hand's label without any error.
CSV_HAND_TO_SLOT = None


def load_split_windows(data_dir, csv_hand_to_slot):
    """
    Returns (train, val): lists of (joint_window, bone_window, slot, label)
    built from every video in data_dir that has a CSV.
    """
    if csv_hand_to_slot is None:
        raise ValueError("CSV_HAND_TO_SLOT is not set: confirm which tracker slot the CSV's left/right refer to first.")

    train, val = [], []
    for name in VIDEO_NAMES:
        video_path = os.path.join(data_dir, f"{name}.mp4")
        csv_path = os.path.join(data_dir, f"{name}.csv")
        if not (os.path.exists(video_path) and os.path.exists(csv_path)):
            print(f"[SKIPPED] {name}: need both {video_path} and {csv_path}")
            continue

        positions, detection_rate = extract_hand_positions(video_path)
        num_frames = len(positions["left"])
        frame_labels = load_frame_labels(csv_path)
        if max(frame_labels) >= num_frames:
            raise ValueError(
                f"{name}: CSV labels frame {max(frame_labels)} but the video has only {num_frames} frames "
                "(0-indexed) — the CSV does not line up with this video."
            )

        for csv_hand, slot in csv_hand_to_slot.items():
            windows = build_labeled_windows(frame_labels, csv_hand, num_frames, WINDOW_SIZE, TRAIN_STEP)
            for split_windows, target in zip(split_train_val(windows, num_frames, WINDOW_SIZE), (train, val)):
                for window, _, label in windows_to_samples(positions[slot], slot, split_windows, WINDOW_SIZE):
                    target.append((*window_features(window), slot, label))
        print(
            f"[LOADED] {name}: {num_frames} frames, {len(frame_labels)} labeled, "
            f"detected left {detection_rate['left']:.0%} / right {detection_rate['right']:.0%}"
        )
    if not train or not val:
        raise ValueError(f"No usable training data in {data_dir} (train={len(train)}, val={len(val)} windows).")
    return train, val


def to_dataset(windows, scaler):
    inputs = np.stack([pack_model_input(*scaler.transform(joint, bone), slot) for joint, bone, slot, _ in windows])
    labels = np.array([label for *_, label in windows], dtype=np.int64)
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(windows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * windows.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy, true labels, predicted labels)."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)

        outputs = model(windows)
        total_loss += criterion(outputs, labels).item() * windows.size(0)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).tolist())
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return total_loss / len(loader.dataset), accuracy, y_true, y_pred


def fit(train_set, val_set, device, epochs=EPOCHS):
    """Trains and returns (best-validation-accuracy weights, that accuracy)."""
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = StrokeBiLSTMModel(num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"[epoch {epoch:2d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    label_ids = list(range(len(LABELS)))
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    return best_state, best_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows, val_windows = load_split_windows(args.data_dir, CSV_HAND_TO_SLOT)
    print(f"train windows: {len(train_windows)}, val windows: {len(val_windows)}, device: {device}")
    for split_name, windows in (("train", train_windows), ("val", val_windows)):
        counts = np.bincount([label for *_, label in windows], minlength=len(LABELS))
        print(f"{split_name} windows per class: {dict(zip(LABELS, counts.tolist()))}")

    scaler = FeatureScaler.fit([w[0] for w in train_windows], [w[1] for w in train_windows])
    best_state, best_accuracy = fit(to_dataset(train_windows, scaler), to_dataset(val_windows, scaler), device)

    torch.save({"model_state": best_state, "scaler": scaler.state_dict(), "labels": LABELS}, CHECKPOINT_PATH)
    print(f"saved {CHECKPOINT_PATH} (best val_acc={best_accuracy:.3f})")


if __name__ == "__main__":
    main()
