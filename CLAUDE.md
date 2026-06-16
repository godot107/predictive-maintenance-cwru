# Predictive Maintenance — CWRU Bearing Faults

Portfolio project: classify rotating-machinery bearing faults from high-frequency
vibration signals using a PyTorch 2-D CNN over spectrograms. Framed for an
**AI Solutioning Consultant** role pivoting into the **energy / industrial** sector
(predictive maintenance for pumps, compressors, turbines).

## What it does
Raw vibration signal → STFT spectrogram → 2-D CNN → fault class
(Normal / Inner Race / Ball / Outer Race), surfaced in a Streamlit dashboard.

## Layout
- `src/downloader.py` — fetches the CWRU 12k Drive-End `.mat` files. **Single
  source of truth for the dataset registry**: `LOAD_FILES` (all 16 = 4 classes ×
  4 motor loads) is the superset; `FILES` (the load-0 subset), `LABEL_NAMES`,
  `LOADS`, sampling rate are derived from it. `python src/downloader.py` grabs the
  4 load-0 files; `--all-loads` grabs all 16. Stdlib-only so it imports cheaply.
- `src/preprocess.py` — load DE channel, segment, FFT, STFT spectrogram, normalise,
  **leakage-free** splits. **Single source of truth for transform params**
  (`SEGMENT_LENGTH`, `FS`, `STFT_NPERSEG`, `STFT_NOVERLAP`). `build_dataset(split_strategy=
  "temporal"|"random")`, `build_cv_folds()`, and `build_cross_load_split(train_loads,
  test_loads)` (physically different recordings → genuine distribution shift).
- `src/model.py` — `BearingCNN`, training loop with **val + early stopping**,
  `cross_validate()`, `cross_load_validate()` (leave-one-load-out), `load_model`/
  `predict` helpers.
- `src/evaluate.py` — confusion matrix + per-class precision/recall/F1, saves
  `reports/confusion_matrix.png`.
- `src/features.py` — engineered vibration features (RMS, kurtosis, crest factor…)
  + SKF-6205 characteristic frequencies & envelope analysis. Feeds the EDA notebook.
- `notebooks/01_eda.ipynb` — EDA narrative / Medium backbone. Regenerate by
  re-executing: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`.
- `src/app.py` — Streamlit dashboard with the Smoothie (FFT) and Sheet-Music
  (spectrogram) analogies.
- `data/`, `models/` — git-ignored; `reports/` mostly ignored but the README's
  figures are committed (see `.gitignore` negations).
- `requirements-dev.txt` — adds jupyter/seaborn for the notebook (`pip install -r requirements-dev.txt`).

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # project-local env (torch is heavy)
pip install -r requirements.txt
python src/downloader.py          # download dataset (load 0) -> data/
python src/downloader.py --all-loads  # all 16 files for the cross-load benchmark
python src/model.py --epochs 50   # train (leakage-free + early stopping) -> models/bearing_cnn.pth
python src/model.py --cv 5        # leakage-free 5-fold cross-validation
python src/model.py --cross-load  # leave-one-load-out cross-load generalization
python src/evaluate.py            # confusion matrix + per-class F1 -> reports/
streamlit run src/app.py          # dashboard
```

## Key decisions / constraints
- Labels: `0 Normal, 1 Inner Race, 2 Ball, 3 Outer Race` (defined in `downloader.FILES`).
- All scripts resolve `data/`/`models/` relative to the project root, so they work
  regardless of the current working directory.
- `BearingCNN` uses `AdaptiveAvgPool2d`, so changing STFT settings won't break the
  classifier head.
- CUDA is used automatically when available (dev box has a GTX 1660); falls back to CPU.
- CWRU's legacy `csegroups.case.edu` host is retired — downloader tries the live
  `engineering.case.edu` mirror first.
- Classes are balanced by truncation (the Normal recording is ~2× longer than faults).
