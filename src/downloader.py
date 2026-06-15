"""Download the CWRU 12k Drive-End bearing vibration dataset.

This module is the single source of truth for *which* files make up the dataset
(the ``FILES`` registry below). It is intentionally dependency-free (standard
library only) so it can be imported by the rest of the pipeline without pulling
in numpy/scipy.

Case Western Reserve University (CWRU) Bearing Data Center:
    https://engineering.case.edu/bearingdatacenter

We use four 12 kHz Drive-End (DE) recordings, each representing one health state:

    file | fault class | fault diameter | label
    -----+-------------+----------------+------
    97   | Normal      | --             | 0
    105  | Inner Race  | 0.007"         | 1
    118  | Ball        | 0.007"         | 2
    130  | Outer Race  | 0.007"         | 3

Run directly to fetch everything into ``data/``::

    python src/downloader.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error

# --- Project paths (resolved relative to this file, not the current dir) -----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- Dataset registry --------------------------------------------------------
# ``label`` is the integer the model learns; ``fault`` is the human-readable
# name; ``de_key`` is the expected MATLAB variable holding the Drive-End signal.
FILES = [
    {"file_id": "97",  "label": 0, "fault": "Normal",     "de_key": "X097_DE_time"},
    {"file_id": "105", "label": 1, "fault": "Inner Race", "de_key": "X105_DE_time"},
    {"file_id": "118", "label": 2, "fault": "Ball",       "de_key": "X118_DE_time"},
    {"file_id": "130", "label": 3, "fault": "Outer Race", "de_key": "X130_DE_time"},
]

# Ordered by label index -> name. Used everywhere for display.
LABEL_NAMES = [f["fault"] for f in sorted(FILES, key=lambda f: f["label"])]

# Sampling rate of the 12k Drive-End recordings (Hz).
SAMPLING_RATE = 12_000

# Candidate hosts, tried in order. The legacy ``csegroups.case.edu`` host from
# the original CWRU documentation is now retired, so we try the live
# ``engineering.case.edu`` mirror first and fall back to the legacy URL.
URL_TEMPLATES = [
    "https://engineering.case.edu/sites/default/files/{file_id}.mat",
    "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{file_id}.mat",
]

# A browser-like User-Agent avoids the server returning an HTML block page.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _looks_like_mat(path: str) -> bool:
    """Cheap validation that a downloaded file is a real MATLAB v5 .mat file."""
    try:
        if os.path.getsize(path) < 10_000:  # CWRU files are >1 MB
            return False
        with open(path, "rb") as fh:
            return fh.read(6) == b"MATLAB"
    except OSError:
        return False


def _download_one(file_id: str, dest: str) -> bool:
    """Try each candidate URL until one yields a valid .mat file."""
    for template in URL_TEMPLATES:
        url = template.format(file_id=file_id)
        try:
            print(f"  -> trying {url}")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
            if _looks_like_mat(dest):
                print(f"  OK   {len(data) / 1_048_576:.1f} MB")
                return True
            print("  bad file (not a MATLAB .mat), trying next mirror...")
            os.remove(dest)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
    return False


def download_dataset(data_dir: str = DATA_DIR) -> dict[str, str]:
    """Download every file in ``FILES`` into ``data_dir`` (skips existing).

    Returns a mapping of ``file_id -> local path`` for the files now present.
    """
    os.makedirs(data_dir, exist_ok=True)
    paths: dict[str, str] = {}
    failures: list[str] = []

    for entry in FILES:
        file_id = entry["file_id"]
        dest = os.path.join(data_dir, f"{file_id}.mat")
        label = f"{entry['fault']} ({file_id}.mat)"

        if os.path.exists(dest) and _looks_like_mat(dest):
            print(f"[skip] {label} already present")
            paths[file_id] = dest
            continue

        print(f"[get ] {label}")
        if _download_one(file_id, dest):
            paths[file_id] = dest
        else:
            failures.append(label)

    print("-" * 60)
    print(f"Ready: {len(paths)}/{len(FILES)} files in {data_dir}")
    if failures:
        print("Could not download:", ", ".join(failures))
        print(
            "The CWRU servers occasionally rate-limit or move files. You can "
            "download the .mat files manually from "
            "https://engineering.case.edu/bearingdatacenter and drop them in "
            f"{data_dir}/"
        )
    return paths


if __name__ == "__main__":
    result = download_dataset()
    sys.exit(0 if len(result) == len(FILES) else 1)
