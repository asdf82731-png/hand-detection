"""
Train the 7-class stroke model with the thesis method (see TRAINING_DESIGN.md).

    python check_data.py --data-dir data          (first: checks the data, says which --csv-left-slot to use)
    python train.py --data-dir data --csv-left-slot left

`data/` holds, for each of Full/Down/Up/Tab, the label file <Name>.csv plus
either a folder <Name>/ of frame images (frame_0026.jpg, ...) or a video
<Name>.mp4. An action missing its CSV or its frames is skipped (and reported).
Writes multi_motion_model_7class.pt: model weights, feature scaler, label
order, the hand mapping and the options below.

Options for the app version (all off by default, so a plain run trains the
same model as before; see TRAINING_DESIGN.md):

    --embedding development   speed-invariant Lie group embedding (model.PathDevelopment)
    --hand-state              add wrist height and angle as inputs (features.hand_state)
    --time-stretch            also train on each window played 0.8x and 1.25x as fast
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import LABELS, build_labeled_windows, load_frame_labels, split_train_val
from features import (
    FeatureScaler, extract_hand_tracks, pack_model_input, resample_length, window_features, window_state_features,
)
from model import StrokeBiLSTMModel

VIDEO_NAMES = ["Full", "Down", "Up", "Tab"]
WINDOW_SIZE = 30
TRAIN_STEP = 1  # thesis A23: dense windows; the split buffer still keeps train and val apart
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "multi_motion_model_7class.pt"
# --time-stretch: each training window is also taken from a span this many
# times its length (played slower or faster) and resampled to WINDOW_SIZE.
TIME_STRETCH_FACTORS = (0.8, 1.25)
STATE_DIM = 6  # wrist height and angle, each with velocity and acceleration



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


def _sample(positions, states, first, span, slot, label):
    """
    The window of `span` frames starting at `first`, resampled to WINDOW_SIZE
    frames if span differs -> (joint features, bone features, state features, slot, label).
    """
    window_positions, window_states = positions[first:first + span], states[first:first + span]
    if span != WINDOW_SIZE:
        window_positions = resample_length(window_positions, WINDOW_SIZE)
        window_states = resample_length(window_states, WINDOW_SIZE)
    return (*window_features(window_positions), window_state_features(window_states), slot, label)


def stretched_starts(start):
    """
    (first frame, span) of the time-stretched copies of the window starting at
    `start`: spans of WINDOW_SIZE x TIME_STRETCH_FACTORS frames that end on the
    same frame, so they keep its label. Spans that would start before frame 0
    are skipped.
    """
    end = start + WINDOW_SIZE
    spans = [round(WINDOW_SIZE * factor) for factor in TIME_STRETCH_FACTORS]
    return [(end - span, span) for span in spans if end - span >= 0]


def load_split_windows(data_dir, csv_hand_to_slot, time_stretch=False):
    """
    Returns (train, val): lists of (joint_window, bone_window, state_window, slot, label)
    built from every video in data_dir that has a CSV.

    With time_stretch, every training window is also added as if played at
    each of TIME_STRETCH_FACTORS (see stretched_starts). A longer span reaches
    further back in time, which stays inside the training part because
    training precedes validation in the split. Validation windows are never
    stretched.
    """
    train, val = [], []
    for name in VIDEO_NAMES:
        source, csv_path = data_sources(data_dir, name)
        if not (os.path.exists(source) and os.path.exists(csv_path)):
            print(f"[SKIPPED] {name}: need {csv_path} and either a {name}/ frame folder or {name}.mp4 in {data_dir}")
            continue
        positions, states, detection_rate = extract_hand_tracks(source)
        num_frames = len(positions["left"])
        frame_labels = load_frame_labels(csv_path)
        if max(frame_labels) >= num_frames:
            raise ValueError(
                f"{name}: CSV labels frame {max(frame_labels)} but {source} has only frames 0-{num_frames - 1} "
                "— the CSV does not line up with these frames."
            )
        for csv_hand, slot in csv_hand_to_slot.items():
            windows = build_labeled_windows(frame_labels, csv_hand, num_frames, WINDOW_SIZE, TRAIN_STEP)
            # Split over the labeled frames only: the real frame folders are
            # 2-4x longer than the range the CSV labels.
            split = split_train_val(windows, max(frame_labels) + 1, WINDOW_SIZE, first_frame=min(frame_labels))
            for split_windows, target, stretch in zip(split, (train, val), (time_stretch, False)):
                for start, label in split_windows:
                    target.append(_sample(positions[slot], states[slot], start, WINDOW_SIZE, slot, label))
                    for first, span in (stretched_starts(start) if stretch else []):
                        target.append(_sample(positions[slot], states[slot], first, span, slot, label))
        print(
            f"[LOADED] {name} from {source}: {num_frames} frames, {len(frame_labels)} labeled, "
            f"detected left {detection_rate['left']:.0%} / right {detection_rate['right']:.0%}"
        )
    if not train or not val:
        raise ValueError(f"No usable training data in {data_dir} (train={len(train)}, val={len(val)} windows).")
    return train, val


def to_dataset(windows, scaler, hand_state=False):
    """Samples -> TensorDataset of (model input, [state input,] label)."""
    inputs = np.stack([pack_model_input(*scaler.transform(joint, bone), slot) for joint, bone, _, slot, _ in windows])
    labels = np.array([label for *_, label in windows], dtype=np.int64)
    tensors = [torch.from_numpy(inputs)]
    if hand_state:
        states = np.stack([scaler.transform_state(state) for _, _, state, _, _ in windows]).astype(np.float32)
        tensors.append(torch.from_numpy(states))
    return TensorDataset(*tensors, torch.from_numpy(labels))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for *inputs, labels in loader:
        inputs = [tensor.to(device) for tensor in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy, true labels, predicted labels, class probabilities)."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    for *inputs, labels in loader:
        inputs = [tensor.to(device) for tensor in inputs]
        labels = labels.to(device)

        outputs = model(*inputs)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).tolist())
        y_prob.extend(torch.softmax(outputs, dim=1).tolist())
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return total_loss / len(loader.dataset), accuracy, y_true, y_pred, np.array(y_prob)


def one_vs_rest_auc(y_true, y_prob):
    """AUC-ROC per class, one-vs-rest (thesis 4.4.2). None where the class or its complement is absent."""
    y_true = np.asarray(y_true)
    aucs = {}
    for class_id, name in enumerate(LABELS):
        is_class = y_true == class_id
        aucs[name] = float(roc_auc_score(is_class, y_prob[:, class_id])) if 0 < is_class.sum() < len(y_true) else None
    return aucs


def fit(train_set, val_set, device, epochs=EPOCHS, embedding="cumulative", state_dim=0):
    """Trains and returns (best-validation-accuracy weights, that accuracy)."""
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = StrokeBiLSTMModel(num_classes=len(LABELS), embedding=embedding, state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, _, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"[epoch {epoch:2d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, _, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device)
    label_ids = list(range(len(LABELS)))
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    print("AUC-ROC, one-vs-rest:")
    for name, auc in one_vs_rest_auc(y_true, y_prob).items():
        print(f"  {name:12s} {'not in validation set' if auc is None else f'{auc:.3f}'}")
    return best_state, best_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--csv-left-slot", required=True, choices=["left", "right"],
        help="which detected hand the CSV's 'left' rows describe; check_data.py prints the value to use",
    )
    parser.add_argument("--embedding", default="cumulative", choices=StrokeBiLSTMModel.EMBEDDINGS,
                        help="development = speed-invariant Lie group embedding; cumulative = the original")
    parser.add_argument("--hand-state", action="store_true", help="add wrist height and angle as model inputs")
    parser.add_argument("--time-stretch", action="store_true",
                        help=f"also train on each window played at {TIME_STRETCH_FACTORS}x speed")
    args = parser.parse_args()
    csv_hand_to_slot = hand_mapping(args.csv_left_slot)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows, val_windows = load_split_windows(args.data_dir, csv_hand_to_slot, args.time_stretch)
    print(f"train windows: {len(train_windows)}, val windows: {len(val_windows)}, device: {device}")
    for split_name, windows in (("train", train_windows), ("val", val_windows)):
        counts = np.bincount([label for *_, label in windows], minlength=len(LABELS))
        print(f"{split_name} windows per class: {dict(zip(LABELS, counts.tolist()))}")

    scaler = FeatureScaler.fit(
        [w[0] for w in train_windows], [w[1] for w in train_windows],
        [w[2] for w in train_windows] if args.hand_state else None,
    )
    state_dim = STATE_DIM if args.hand_state else 0
    best_state, best_accuracy = fit(
        to_dataset(train_windows, scaler, args.hand_state), to_dataset(val_windows, scaler, args.hand_state), device,
        embedding=args.embedding, state_dim=state_dim,
    )
    config = {"embedding": args.embedding, "state_dim": state_dim, "time_stretch": args.time_stretch}
    torch.save(
        {"model_state": best_state, "scaler": scaler.state_dict(), "labels": LABELS, "hand_to_slot": csv_hand_to_slot,
         "config": config},
        CHECKPOINT_PATH,
    )
    print(f"saved {CHECKPOINT_PATH} (best val_acc={best_accuracy:.3f})")


if __name__ == "__main__":
    main()
