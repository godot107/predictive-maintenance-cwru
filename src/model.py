"""2-D CNN that classifies bearing-fault spectrograms, plus training/CV loops.

Train on the (leakage-free) CWRU dataset and save weights to
``models/bearing_cnn.pth``::

    python src/model.py --epochs 50            # train with early stopping
    python src/model.py --cv 5                 # 5-fold cross-validation
    python src/model.py --split random         # leaky baseline, for comparison

Training targets CUDA automatically when a GPU is available.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import build_cv_folds, build_dataset  # noqa: E402
from downloader import LABEL_NAMES, PROJECT_ROOT  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "bearing_cnn.pth")

NUM_CLASSES = len(LABEL_NAMES)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BearingCNN(nn.Module):
    """Compact 2-D CNN for single-channel spectrogram classification.

    An ``AdaptiveAvgPool2d`` head makes the network agnostic to the exact
    spectrogram height/width, so changing the STFT settings won't break it.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap arrays in a DataLoader, adding the (N, 1, H, W) channel dim."""
    tensor_x = torch.from_numpy(X).float().unsqueeze(1)
    tensor_y = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader."""
    model.eval()
    loss_sum = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_model(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    split_strategy: str = "temporal",
    save_path: str = MODEL_PATH,
) -> BearingCNN:
    """Train ``BearingCNN`` with early stopping on validation loss.

    The best (lowest val-loss) weights are restored and reported on the held-out
    test set, then saved to ``save_path``.
    """
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ds = build_dataset(split_strategy=split_strategy)
    print(ds.summary())

    train_loader = _to_loader(ds.X_train, ds.y_train, batch_size, shuffle=True)
    val_loader = _to_loader(ds.X_val, ds.y_val, batch_size, shuffle=False)
    test_loader = _to_loader(ds.X_test, ds.y_test, batch_size, shuffle=False)

    model = BearingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        flag = ""
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            flag = "  <- best"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}{flag}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    # Restore best weights and report on the untouched test set.
    model.load_state_dict(best_state)
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
    print("-" * 64)
    print(f"Best val loss: {best_val_loss:.4f} | Held-out TEST accuracy: {test_acc:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Weights saved to: {save_path}")
    return model


def cross_validate(
    n_splits: int = 5,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> list[float]:
    """Leakage-free k-fold CV; prints per-fold and mean±std test accuracy."""
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device} | {n_splits}-fold leakage-free cross-validation")
    folds = build_cv_folds(n_splits=n_splits)
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for k, (X_tr, y_tr, X_te, y_te) in enumerate(folds, start=1):
        train_loader = _to_loader(X_tr, y_tr, batch_size, shuffle=True)
        test_loader = _to_loader(X_te, y_te, batch_size, shuffle=False)
        model = BearingCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
        _, acc = _evaluate(model, test_loader, criterion, device)
        accuracies.append(acc)
        print(f"  Fold {k}/{n_splits}: test acc {acc:.3f}  (train n={len(y_tr)}, test n={len(y_te)})")

    arr = np.array(accuracies)
    print("-" * 64)
    print(f"CV accuracy: {arr.mean():.3f} ± {arr.std():.3f}")
    return accuracies


def load_model(path: str = MODEL_PATH, device: torch.device | None = None) -> BearingCNN:
    """Instantiate ``BearingCNN`` and load trained weights."""
    device = device or get_device()
    model = BearingCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model: BearingCNN, spectrogram: np.ndarray, device: torch.device | None = None):
    """Predict the fault class of a single spectrogram (H, W).

    Returns ``(label_index, confidence, probabilities)``.
    """
    device = device or get_device()
    x = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
    idx = int(probs.argmax())
    return idx, float(probs[idx]), probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / cross-validate the bearing-fault CNN.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal",
                        help="temporal = leakage-free (default); random = leaky baseline")
    parser.add_argument("--cv", type=int, metavar="K", default=0,
                        help="run K-fold cross-validation instead of a single train run")
    args = parser.parse_args()

    if args.cv:
        cross_validate(n_splits=args.cv, epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    split_strategy=args.split)
