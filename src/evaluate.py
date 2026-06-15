"""Honest evaluation of the trained bearing-fault CNN.

Loads the saved model, runs it on the **leakage-free held-out test set**, and
reports the metrics that actually matter for a classifier: a confusion matrix
and per-class precision / recall / F1 — not just top-line accuracy.

    python src/evaluate.py                 # evaluate on the temporal (honest) split
    python src/evaluate.py --split random  # evaluate on the leaky baseline split

A confusion-matrix image is written to ``reports/confusion_matrix.png`` for use
in the README / Medium article.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import build_dataset  # noqa: E402
from model import MODEL_PATH, get_device, load_model  # noqa: E402
from downloader import LABEL_NAMES, PROJECT_ROOT  # noqa: E402

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


@torch.no_grad()
def predict_all(model, X: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    """Return predicted label indices for a stack of spectrograms (N, H, W)."""
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().unsqueeze(1).to(device)
        preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


def evaluate(split_strategy: str = "temporal", model_path: str = MODEL_PATH) -> dict:
    """Run the model on the test split and print/save diagnostics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}. Train first: `python src/model.py`.")

    device = get_device()
    print(f"Device: {device} | split='{split_strategy}'")

    ds = build_dataset(split_strategy=split_strategy)
    model = load_model(model_path, device)

    y_true = ds.y_test
    y_pred = predict_all(model, ds.X_test, device)
    accuracy = float((y_true == y_pred).mean())

    print(f"\nHeld-out test accuracy: {accuracy:.3f}  (n={len(y_true)})\n")
    print("Per-class report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    _save_confusion_plot(cm, split_strategy)
    return {"accuracy": accuracy, "confusion_matrix": cm, "y_true": y_true, "y_pred": y_pred}


def _save_confusion_plot(cm: np.ndarray, split_strategy: str) -> str:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Bearing-fault CNN — confusion matrix ({split_strategy} split)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nConfusion-matrix image saved to: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained bearing-fault CNN.")
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    args = parser.parse_args()
    evaluate(split_strategy=args.split)
