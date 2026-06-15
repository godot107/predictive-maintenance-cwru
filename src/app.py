"""Streamlit dashboard for bearing-fault predictive maintenance.

Pick a raw vibration sample from the held-out test set and watch it travel
through the pipeline:

    raw signal  ->  FFT spectrum  ->  2-D spectrogram  ->  CNN diagnosis

Launch from the project root::

    streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import FS, build_dataset, compute_fft, compute_spectrogram  # noqa: E402
from model import MODEL_PATH, get_device, load_model, predict  # noqa: E402
from downloader import LABEL_NAMES  # noqa: E402

# Friendly status strings shown after inference.
STATUS_MESSAGES = {
    "Normal": "Normal",
    "Inner Race": "Inner Race Fault Detected",
    "Ball": "Ball Fault Detected",
    "Outer Race": "Outer Race Fault Detected",
}

st.set_page_config(page_title="Bearing Predictive Maintenance", page_icon="🛠️", layout="wide")


@st.cache_data(show_spinner="Loading & preprocessing CWRU dataset...")
def get_dataset():
    return build_dataset()


@st.cache_resource(show_spinner="Loading CNN model...")
def get_model():
    device = get_device()
    return load_model(MODEL_PATH, device), device


# --- Header ------------------------------------------------------------------
st.title("🛠️ High-Frequency Sensor Predictive Maintenance")
st.caption(
    "Diagnosing rotating-machinery bearing faults from vibration signals with a "
    "PyTorch 2-D CNN — CWRU Bearing Dataset."
)

# Guard rails: data + model must exist.
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if not os.path.exists(os.path.join(data_dir, "97.mat")):
    st.error("Dataset not found. Run `python src/downloader.py` first.")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run `python src/model.py` first.")
    st.stop()

ds = get_dataset()
model, device = get_model()

# --- Sidebar: sample selection ----------------------------------------------
st.sidebar.header("Choose a test sample")
st.sidebar.write(f"Inference device: **{str(device).upper()}**")

class_filter = st.sidebar.selectbox(
    "Filter by true condition", ["Any"] + LABEL_NAMES
)
if class_filter == "Any":
    candidate_idx = np.arange(len(ds.y_test))
else:
    candidate_idx = np.where(ds.y_test == LABEL_NAMES.index(class_filter))[0]

position = st.sidebar.slider("Sample", 0, len(candidate_idx) - 1, 0)
sample_idx = int(candidate_idx[position])

raw = ds.raw_test[sample_idx]
true_label = LABEL_NAMES[int(ds.y_test[sample_idx])]

# --- Inference ---------------------------------------------------------------
spec = compute_spectrogram(raw)
pred_idx, confidence, probs = predict(model, spec, device)
pred_label = LABEL_NAMES[pred_idx]

c1, c2, c3 = st.columns(3)
c1.metric("Ground Truth", true_label)
c2.metric("CNN Diagnosis", STATUS_MESSAGES[pred_label], f"{confidence * 100:.1f}% confidence")
c3.metric("Result", "✅ Correct" if pred_label == true_label else "❌ Mismatch")

if pred_label == "Normal":
    st.success(f"Diagnosis: **{STATUS_MESSAGES[pred_label]}** — bearing is healthy.")
else:
    st.warning(f"Diagnosis: **{STATUS_MESSAGES[pred_label]}** — schedule maintenance.")

with st.expander("Class probabilities"):
    st.bar_chart({name: float(p) for name, p in zip(LABEL_NAMES, probs)})

st.divider()

# --- Visualisations ----------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1 · Raw Time-Domain Vibration")
    fig, ax = plt.subplots(figsize=(6, 3))
    t = np.arange(len(raw)) / FS
    ax.plot(t, raw, linewidth=0.6, color="#2c7fb8")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title("What the accelerometer actually records")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "The accelerometer streams thousands of samples per second. Faults are "
        "buried in here as tiny periodic impacts — hard to see by eye."
    )

    st.subheader("2 · FFT Frequency Spectrum")
    freqs, mag = compute_fft(raw, FS)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freqs, mag, linewidth=0.7, color="#d95f0e")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Which frequencies are present")
    fig.tight_layout()
    st.pyplot(fig)
    st.info(
        "**🥤 The Smoothie Analogy.** The raw signal is a smoothie — every "
        "frequency blended together. The FFT is the blender run in reverse: it "
        "separates the smoothie back into its ingredients so we can see exactly "
        "which 'flavours' (frequencies of bearing wear) are present and how much."
    )

with col_right:
    st.subheader("3 · 2-D Spectrogram (model input)")
    fig, ax = plt.subplots(figsize=(6, 3))
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.set_title("Frequency content over time")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    st.info(
        "**🎼 The Sheet Music Analogy.** A plain FFT tells you which notes were "
        "played but not *when*. A spectrogram is sheet music: time runs left to "
        "right, pitch (frequency) runs bottom to top, and brightness is loudness. "
        "The CNN reads this 'sheet music' to recognise a fault's signature."
    )

    st.markdown("#### How the diagnosis was made")
    st.markdown(
        "- The raw window is converted to the spectrogram on the left.\n"
        "- That image is fed to a **2-D convolutional neural network**.\n"
        "- The CNN has learned the visual fingerprint of each fault type and "
        "outputs the probabilities shown above."
    )

st.divider()
st.caption(
    "CWRU 12 kHz Drive-End data · PyTorch CNN · "
    f"running inference on **{str(device).upper()}**."
)
