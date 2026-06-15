"""Engineered vibration features for bearing-fault EDA and baselines.

Two families, both standard in rotating-machinery condition monitoring:

* **Time-domain statistics** — RMS, peak, kurtosis, crest factor, etc. Cheap and
  highly interpretable (kurtosis & crest factor spike on impulsive faults).
* **Frequency / envelope features** — spectral shape plus *envelope-spectrum*
  energy at the bearing's **characteristic fault frequencies**. The CWRU drive-end
  bearing is an **SKF 6205**, whose defect frequencies are known multiples of shaft
  speed, so we can check whether a fault's energy lands exactly where the physics
  predicts (envelope / demodulation analysis — the method vibration engineers use).

These feed `notebooks/01_eda.ipynb` and can later augment the CNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from downloader import SAMPLING_RATE

FS = SAMPLING_RATE

# --- SKF 6205 (CWRU drive-end) characteristic fault-frequency multipliers ------
# Each value multiplies the shaft rotational frequency fr = rpm / 60 (Hz).
# FTF = cage, BPFO = outer race, BSF = ball spin, BPFI = inner race.
CHAR_MULTIPLIERS = {
    "FTF": 0.3983,
    "BPFO": 3.5848,
    "BSF": 4.7135,
    "BPFI": 5.4152,
}

# SKF 6205-2RS geometry (for transparency / reproducibility).
SKF6205_GEOMETRY = {"n_balls": 9, "ball_diameter_in": 0.3126,
                    "pitch_diameter_in": 1.537, "contact_angle_deg": 0.0}

# All four current files are the 0-hp load condition (~1797 rpm).
DEFAULT_RPM = 1797


# --- Time domain -------------------------------------------------------------
def time_domain_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(abs_x.max())
    mean_abs = float(abs_x.mean()) or 1e-12
    sqrt_mean = float(np.mean(np.sqrt(abs_x))) ** 2 or 1e-12
    rms_safe = rms or 1e-12
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(x.max() - x.min()),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),          # excess kurtosis (normal = 0)
        "crest_factor": peak / rms_safe,         # impulsiveness
        "shape_factor": rms / mean_abs,
        "impulse_factor": peak / mean_abs,
        "clearance_factor": peak / sqrt_mean,
    }


# --- Frequency domain --------------------------------------------------------
def frequency_domain_features(x: np.ndarray, fs: int = FS) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    p = spec / (spec.sum() or 1e-12)             # spectrum as a distribution
    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    return {
        "spec_centroid": centroid,
        "spec_spread": spread,
        "spec_skew": float(skew(spec)),
        "spec_kurtosis": float(kurtosis(spec)),
        "spec_energy": float(np.sum(spec ** 2)),
    }


# --- Envelope (demodulation) analysis ----------------------------------------
def characteristic_frequencies(rpm: float = DEFAULT_RPM) -> dict[str, float]:
    """Bearing defect frequencies in Hz for a given shaft speed."""
    fr = rpm / 60.0
    return {name: mult * fr for name, mult in CHAR_MULTIPLIERS.items()}


def envelope_spectrum(x: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert-envelope spectrum: reveals low-frequency impact repetition rates."""
    x = np.asarray(x, dtype=np.float64)
    env = np.abs(hilbert(x))
    env = env - env.mean()
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freqs, mag


def envelope_band_energy(
    x: np.ndarray, fs: int = FS, rpm: float = DEFAULT_RPM, half_bw_hz: float = 5.0
) -> dict[str, float]:
    """Energy in the envelope spectrum within ±half_bw of each defect frequency."""
    freqs, mag = envelope_spectrum(x, fs)
    out = {}
    for name, f0 in characteristic_frequencies(rpm).items():
        band = (freqs >= f0 - half_bw_hz) & (freqs <= f0 + half_bw_hz)
        out[f"env_{name}"] = float(mag[band].sum())
    return out


# --- Combined ----------------------------------------------------------------
def extract_features(
    x: np.ndarray, fs: int = FS, rpm: float = DEFAULT_RPM, include_envelope: bool = True
) -> dict[str, float]:
    feats = {**time_domain_features(x), **frequency_domain_features(x, fs)}
    if include_envelope:
        feats.update(envelope_band_energy(x, fs, rpm))
    return feats


def build_feature_frame(
    raw: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    fs: int = FS,
    rpm: float = DEFAULT_RPM,
) -> pd.DataFrame:
    """One row of engineered features per segment, plus label/fault columns."""
    rows = [extract_features(seg, fs, rpm) for seg in raw]
    df = pd.DataFrame(rows)
    df["label"] = np.asarray(y)
    df["fault"] = [label_names[i] for i in y]
    return df


FEATURE_COLUMNS = list(extract_features(np.random.randn(2048)).keys())


if __name__ == "__main__":
    # Smoke test on a synthetic impulsive signal.
    t = np.linspace(0, 1, FS, endpoint=False)
    clean = np.sin(2 * np.pi * 60 * t)
    impulsive = clean + (np.random.rand(FS) > 0.99) * 8.0
    print("features:", len(FEATURE_COLUMNS))
    print("clean    kurtosis:", round(extract_features(clean)["kurtosis"], 2))
    print("impulsive kurtosis:", round(extract_features(impulsive)["kurtosis"], 2))
    print("char freqs (Hz):", {k: round(v, 1) for k, v in characteristic_frequencies().items()})
