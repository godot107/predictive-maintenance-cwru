"""Turn raw CWRU vibration recordings into model-ready spectrograms.

Pipeline
--------
1. Load the Drive-End (DE) channel from each ``.mat`` file.
2. Slice the long 1-D signal into fixed-length windows ("segments").
3. Transform each segment into the frequency / time-frequency domain:
     * FFT magnitude spectrum                (1-D, used in the dashboard)
     * Short-Time Fourier Transform (STFT)   (2-D, fed to the CNN)
4. Log-scale + min-max normalise each spectrogram to [0, 1].
5. Split into train / validation / test.

Splitting strategy (important!)
-------------------------------
Each fault class here comes from a *single continuous recording*. If you segment
with overlap and then split segments **randomly**, near-duplicate windows land on
both sides of the split and the test score is meaninglessly high (data leakage).

The default ``split_strategy="temporal"`` instead cuts each *raw signal* into
contiguous train/val/test spans (with a guard gap) **before** segmenting, so no
window is shared across splits. ``split_strategy="random"`` reproduces the old,
leaky behaviour and is kept only so the notebook/blog can show the contrast.

All transform parameters live here as module constants so the trainer
(``model.py``), evaluator (``evaluate.py``) and dashboard (``app.py``) stay in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.model_selection import train_test_split

# Registry + paths come from the (dependency-free) downloader module.
from downloader import DATA_DIR, FILES, LABEL_NAMES, LOAD_FILES, SAMPLING_RATE

# --- Transform parameters (single source of truth) ---------------------------
SEGMENT_LENGTH = 2048      # samples per window (~0.17 s at 12 kHz)
SEGMENT_OVERLAP = 0.5      # fractional overlap between consecutive windows
FS = SAMPLING_RATE         # sampling frequency (Hz)

# STFT / spectrogram settings. With these values a 2048-sample segment becomes
# a (65, 61) time-frequency image.
STFT_NPERSEG = 128
STFT_NOVERLAP = 96

# Default 3-way split fractions (train is the remainder).
VAL_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42


# --- Feature extraction ------------------------------------------------------
def compute_fft(segment: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """One-sided FFT magnitude spectrum. Returns (frequencies_hz, magnitude)."""
    n = len(segment)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(np.fft.rfft(segment)) * (2.0 / n)
    return freqs, magnitude.astype(np.float32)


def compute_spectrogram(
    segment: np.ndarray,
    fs: int = FS,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
) -> np.ndarray:
    """STFT spectrogram, log-scaled and min-max normalised to [0, 1].

    Returns a 2-D array of shape (n_frequencies, n_times).
    """
    _, _, sxx = scipy_spectrogram(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sxx = np.log1p(sxx)  # compress dynamic range; faults live in small details
    lo, hi = sxx.min(), sxx.max()
    if hi - lo > 1e-12:
        sxx = (sxx - lo) / (hi - lo)
    else:
        sxx = np.zeros_like(sxx)
    return sxx.astype(np.float32)


# Fixed spectrogram shape implied by the constants above (e.g. (65, 61)).
SPECTROGRAM_SHAPE = compute_spectrogram(np.zeros(SEGMENT_LENGTH, dtype=np.float32)).shape


@dataclass
class Dataset:
    """Container returned by :func:`build_dataset`.

    ``X_*`` are spectrograms of shape (N, H, W); ``raw_*`` are the matching raw
    1-D segments (handy for the dashboard); ``y_*`` are integer labels.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    raw_train: np.ndarray
    raw_val: np.ndarray
    raw_test: np.ndarray
    label_names: list[str]
    split_strategy: str

    @property
    def spectrogram_shape(self) -> tuple[int, int]:
        return SPECTROGRAM_SHAPE

    def summary(self) -> str:
        rows = [f"split='{self.split_strategy}'  spectrogram={self.spectrogram_shape}"]
        for split, y in (("train", self.y_train), ("val", self.y_val), ("test", self.y_test)):
            counts = ", ".join(
                f"{self.label_names[i]}={int((y == i).sum())}"
                for i in range(len(self.label_names))
            )
            rows.append(f"  {split:<5} (n={len(y):4d}): {counts}")
        return "\n".join(rows)


# --- Loading -----------------------------------------------------------------
def find_de_key(mat: dict) -> str:
    """Return the MATLAB variable name holding the Drive-End signal.

    CWRU files name this ``X097_DE_time``, ``X105_DE_time``, etc. We match on the
    ``_DE_time`` suffix rather than hard-coding the number.
    """
    for key in mat:
        if key.endswith("_DE_time"):
            return key
    for key in mat:  # fall back to any non-meta key mentioning DE
        if not key.startswith("__") and "DE" in key:
            return key
    raise KeyError("No Drive-End (_DE_time) variable found in .mat file")


def load_signal(mat_path: str) -> np.ndarray:
    """Load and flatten the Drive-End vibration signal from a ``.mat`` file."""
    mat = loadmat(mat_path)
    return np.asarray(mat[find_de_key(mat)], dtype=np.float32).ravel()


# --- Segmentation ------------------------------------------------------------
def segment_signal(
    signal: np.ndarray,
    segment_length: int = SEGMENT_LENGTH,
    overlap: float = SEGMENT_OVERLAP,
) -> np.ndarray:
    """Slice a 1-D signal into overlapping windows -> shape (n_segments, L)."""
    step = max(1, int(segment_length * (1.0 - overlap)))
    starts = range(0, len(signal) - segment_length + 1, step)
    segments = [signal[s : s + segment_length] for s in starts]
    if not segments:
        return np.empty((0, segment_length), dtype=np.float32)
    return np.stack(segments).astype(np.float32)


def temporal_split_signal(
    signal: np.ndarray,
    fractions: tuple[float, float, float],
    gap: int = SEGMENT_LENGTH,
) -> list[np.ndarray]:
    """Cut a 1-D signal into contiguous spans by time, trimming a guard ``gap``.

    Returns ``[train_span, val_span, test_span]``. The gap removes the samples
    straddling each boundary so adjacent (highly autocorrelated) windows don't
    leak between splits.
    """
    n = len(signal)
    bounds = [0] + list(np.cumsum([int(round(f * n)) for f in fractions]))
    bounds[-1] = n  # absorb rounding into the final span
    spans = []
    for i in range(len(fractions)):
        a, b = bounds[i], bounds[i + 1]
        if i > 0:
            a = min(a + gap, b)  # guard gap at the start of every span but the first
        spans.append(signal[a:b])
    return spans


# --- Dataset assembly --------------------------------------------------------
def _balance(per_class: dict[int, np.ndarray], rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Truncate every class to the size of the smallest (random subset)."""
    if not per_class:
        return per_class
    min_n = min(len(v) for v in per_class.values())
    out = {}
    for lbl, segs in per_class.items():
        if len(segs) > min_n:
            pick = np.sort(rng.choice(len(segs), size=min_n, replace=False))
            out[lbl] = segs[pick]
        else:
            out[lbl] = segs
    return out


def _assemble(per_class_raw: dict[int, list[np.ndarray]], balance: bool, rng):
    """Stack per-class raw segments -> (X spectrograms, raw segments, y labels)."""
    per_class = {lbl: np.concatenate(chunks) for lbl, chunks in per_class_raw.items() if len(chunks)}
    per_class = {lbl: v for lbl, v in per_class.items() if len(v)}
    if balance:
        per_class = _balance(per_class, rng)

    raws, labels = [], []
    for lbl in sorted(per_class):
        raws.append(per_class[lbl])
        labels.append(np.full(len(per_class[lbl]), lbl, dtype=np.int64))
    if not raws:
        empty_x = np.empty((0, *SPECTROGRAM_SHAPE), dtype=np.float32)
        return empty_x, np.empty((0, SEGMENT_LENGTH), dtype=np.float32), np.empty((0,), dtype=np.int64)

    raw = np.concatenate(raws)
    y = np.concatenate(labels)
    X = np.stack([compute_spectrogram(s) for s in raw])
    return X, raw, y


def build_dataset(
    data_dir: str = DATA_DIR,
    segment_length: int = SEGMENT_LENGTH,
    overlap: float = SEGMENT_OVERLAP,
    split_strategy: str = "temporal",
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    gap: int | None = None,
    balance: bool = True,
) -> Dataset:
    """Build a leakage-free train/val/test spectrogram dataset.

    ``split_strategy="temporal"`` (default) splits each raw recording by time
    before segmenting; ``"random"`` segments first then splits randomly (leaky
    baseline, for comparison only).
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")
    gap = segment_length if gap is None else gap
    rng = np.random.default_rng(RANDOM_STATE)

    signals = {}
    for entry in FILES:
        path = os.path.join(data_dir, f"{entry['file_id']}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run `python src/downloader.py` first.")
        signals[entry["label"]] = load_signal(path)

    if split_strategy == "temporal":
        train_raw: dict[int, list] = {}
        val_raw: dict[int, list] = {}
        test_raw: dict[int, list] = {}
        fractions = (1.0 - val_size - test_size, val_size, test_size)
        for lbl, sig in signals.items():
            spans = temporal_split_signal(sig, fractions, gap=gap)
            for bucket, span in zip((train_raw, val_raw, test_raw), spans):
                bucket.setdefault(lbl, []).append(segment_signal(span, segment_length, overlap))

        X_train, raw_train, y_train = _assemble(train_raw, balance, rng)
        X_val, raw_val, y_val = _assemble(val_raw, balance, rng)
        X_test, raw_test, y_test = _assemble(test_raw, balance, rng)

    elif split_strategy == "random":
        all_raw = {lbl: [segment_signal(sig, segment_length, overlap)] for lbl, sig in signals.items()}
        X_all, raw_all, y_all = _assemble(all_raw, balance, rng)
        idx = np.arange(len(y_all))
        rel_test = test_size
        rel_val = val_size / (1.0 - test_size)
        tr_val, te = train_test_split(idx, test_size=rel_test, stratify=y_all, random_state=RANDOM_STATE)
        tr, va = train_test_split(tr_val, test_size=rel_val, stratify=y_all[tr_val], random_state=RANDOM_STATE)
        X_train, raw_train, y_train = X_all[tr], raw_all[tr], y_all[tr]
        X_val, raw_val, y_val = X_all[va], raw_all[va], y_all[va]
        X_test, raw_test, y_test = X_all[te], raw_all[te], y_all[te]
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    return Dataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        raw_train=raw_train, raw_val=raw_val, raw_test=raw_test,
        label_names=LABEL_NAMES, split_strategy=split_strategy,
    )


def build_cv_folds(
    data_dir: str = DATA_DIR,
    n_splits: int = 5,
    segment_length: int = SEGMENT_LENGTH,
    balance: bool = True,
):
    """Yield ``n_splits`` leakage-free (X_train, y_train, X_test, y_test) folds.

    Uses **non-overlapping** segments and contiguous, per-class time blocks, so a
    test block never shares samples with its training data. Good for a stable
    accuracy estimate on this small dataset.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    per_class: dict[int, np.ndarray] = {}
    for entry in FILES:
        path = os.path.join(data_dir, f"{entry['file_id']}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run `python src/downloader.py` first.")
        per_class[entry["label"]] = segment_signal(load_signal(path), segment_length, overlap=0.0)

    if balance:
        per_class = _balance(per_class, rng)

    folds = []
    for k in range(n_splits):
        tr_raw: dict[int, list] = {}
        te_raw: dict[int, list] = {}
        for lbl, segs in per_class.items():
            blocks = np.array_split(np.arange(len(segs)), n_splits)
            test_idx = blocks[k]
            train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != k])
            tr_raw[lbl] = [segs[train_idx]]
            te_raw[lbl] = [segs[test_idx]]
        X_tr, _, y_tr = _assemble(tr_raw, balance=False, rng=rng)
        X_te, _, y_te = _assemble(te_raw, balance=False, rng=rng)
        folds.append((X_tr, y_tr, X_te, y_te))
    return folds


def build_cross_load_split(
    train_loads,
    test_loads,
    data_dir: str = DATA_DIR,
    segment_length: int = SEGMENT_LENGTH,
    overlap: float = SEGMENT_OVERLAP,
    balance: bool = True,
):
    """Build a cross-load generalization split from the ``LOAD_FILES`` registry.

    Train segments come from the recordings at ``train_loads`` (motor loads in
    HP); test segments come from the *unseen* ``test_loads``. Because train and
    test are physically different recordings (different RPM/load), no window is
    shared — this is the genuinely hard, deployment-relevant benchmark.

    Returns ``(X_train, y_train, X_test, y_test)`` with spectrograms of shape
    ``(N, H, W)`` and integer labels. Each class is balanced (truncated to the
    smallest class) independently within train and within test.
    """
    train_loads, test_loads = set(train_loads), set(test_loads)
    rng = np.random.default_rng(RANDOM_STATE)

    def collect(loads: set) -> tuple[np.ndarray, np.ndarray]:
        per_class: dict[int, list] = {}
        for entry in LOAD_FILES:
            if entry["load"] not in loads:
                continue
            path = os.path.join(data_dir, f"{entry['file_id']}.mat")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run `python src/downloader.py --all-loads` first."
                )
            segs = segment_signal(load_signal(path), segment_length, overlap)
            per_class.setdefault(entry["label"], []).append(segs)
        X, _, y = _assemble(per_class, balance, rng)
        return X, y

    X_train, y_train = collect(train_loads)
    X_test, y_test = collect(test_loads)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("=== temporal (leakage-free) ===")
    print(build_dataset(split_strategy="temporal").summary())
    print("\n=== random (leaky baseline) ===")
    print(build_dataset(split_strategy="random").summary())
