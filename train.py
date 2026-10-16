"""
Training script for the 7-class retrain (TRAINING_DESIGN.md, Phase 3/4).

This wires model.py + data_pipeline.py together. It does NOT yet read real
video — Phase 1 (obtaining usable copies of the source videos) is still
blocked, so `load_training_samples()` below is the one piece that depends on
real video/frame access and is left as a TODO. Everything else (model,
optimizer, training loop) is complete and exercised by the synthetic smoke
test in `if __name__ == "__main__"`.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MultiMotionDevLSTMModel

NUM_CLASSES = 7


class StrokeWindowDataset(Dataset):
    """Wraps (window, hand, label) samples from data_pipeline.windows_to_samples()."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, _hand, label = self.samples[idx]
        return torch.as_tensor(window, dtype=torch.float32), label


def load_training_samples():
    """
    TODO (blocked on Phase 1): build the full (window, hand, label) sample
    list for all four source videos using data_pipeline.load_frame_labels,
    build_labeled_windows, split_train_val and windows_to_samples, combined
    with per-frame features from pipeline.preprocess_landmarks_per_hand.

    Once Full/Down/Tab/Up.mp4 (or usable compressed copies) are available,
    this should loop over each video, extract per-hand feature blocks per
    frame, load + normalize its CSV labels, build labeled windows per hand,
    split into train/val, and concatenate across all four videos.
    """
    raise NotImplementedError(
        "Real video/CSV materials are not available yet (Phase 1 blocked). "
        "See TRAINING_DESIGN.md section 1 for the exact steps to implement here."
    )


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
    model.eval()
    total_loss = 0.0
    correct = 0
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)

        outputs = model(windows)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * windows.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = load_training_samples()  # raises until Phase 1 unblocks this
    train_loader = DataLoader(StrokeWindowDataset(train_samples), batch_size=16, shuffle=True)

    model = MultiMotionDevLSTMModel(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}")

    torch.save(model.state_dict(), "multi_motion_model_7class.pt")


def _synthetic_smoke_test():
    """Confirms model + training-loop shapes are correct without any real data."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)

    samples = [
        (rng.standard_normal((30, 62, 9)).astype(np.float32), "left", label)
        for label in range(NUM_CLASSES)
        for _ in range(4)
    ]
    loader = DataLoader(StrokeWindowDataset(samples), batch_size=8, shuffle=True)

    model = MultiMotionDevLSTMModel(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss_before = train_one_epoch(model, loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, loader, criterion, device)
    print(f"[smoke test] train_loss={loss_before:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}")
    assert np.isfinite(loss_before)
    assert np.isfinite(val_loss)
    print("[smoke test] OK")


if __name__ == "__main__":
    _synthetic_smoke_test()
