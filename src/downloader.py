"""Download the CWRU 12k Drive-End bearing vibration dataset.

This module is the single source of truth for *which* files make up the dataset
(the ``FILES`` registry below). It is intentionally dependency-free (standard
library only) so it can be imported by the rest of the pipeline without pulling
in numpy/scipy.

Case Western Reserve University (CWRU) Bearing Data Center:
    https://engineering.case.edu/bearingdatacenter

We use four 12 kHz Drive-End (DE) recordings, each representing one health state.
At the default motor load (0 HP, ~1797 RPM) those are:

    file | fault class | fault diameter | label
    -----+-------------+----------------+------
    97   | Normal      | --             | 0
    105  | Inner Race  | 0.007"         | 1
    118  | Ball        | 0.007"         | 2
    130  | Outer Race  | 0.007"         | 3

CWRU also recorded each condition at motor loads of 1, 2 and 3 HP (different
RPMs). The full 4-class x 4-load registry lives in ``LOAD_FILES`` and powers the
*cross-load generalization* benchmark (train on some loads, test on an unseen
load). ``FILES`` is the 0 HP subset used by the default single-condition pipeline.

Run directly to fetch the default 4 files into ``data/``::

    python src/downloader.py             # 4 files (load 0 only)
    python src/downloader.py --all-loads # all 16 files (cross-load benchmark)
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
# Full 4-class x 4-load registry (0.007" faults, 12 kHz Drive-End). ``label`` is
# the integer the model learns; ``fault`` is the human-readable name; ``load`` is
# the motor load in HP (0/1/2/3 -> ~1797/1772/1750/1730 RPM). The Drive-End
# MATLAB variable name (``de_key``) is derived: file 97 -> ``X097_DE_time``.
#
#            load 0   load 1   load 2   load 3
#   Normal      97       98       99      100
#   Inner Race 105      106      107      108
#   Ball       118      119      120      121
#   Outer Race 130      131      132      133
_REGISTRY = [
    # (fault, label, [file_id per load 0, 1, 2, 3])
    ("Normal",     0, ["97",  "98",  "99",  "100"]),
    ("Inner Race", 1, ["105", "106", "107", "108"]),
    ("Ball",       2, ["118", "119", "120", "121"]),
    ("Outer Race", 3, ["130", "131", "132", "133"]),
]

LOAD_FILES = [
    {
        "file_id": file_id,
        "label": label,
        "fault": fault,
        "load": load,
        "de_key": f"X{int(file_id):03d}_DE_time",
    }
    for fault, label, ids in _REGISTRY
    for load, file_id in enumerate(ids)
]

# The default single-condition dataset: the 0 HP subset, ordered by label.
FILES = [f for f in LOAD_FILES if f["load"] == 0]

# Ordered by label index -> name. Used everywhere for display.
LABEL_NAMES = [f["fault"] for f in sorted(FILES, key=lambda f: f["label"])]

# Distinct motor loads available in the cross-load registry.
LOADS = sorted({f["load"] for f in LOAD_FILES})

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


def download_dataset(data_dir: str = DATA_DIR, files: list[dict] | None = None) -> dict[str, str]:
    """Download every entry in ``files`` into ``data_dir`` (skips existing).

    ``files`` defaults to ``FILES`` (the 4 load-0 recordings); pass ``LOAD_FILES``
    to fetch all 16 for the cross-load benchmark. Returns a mapping of
    ``file_id -> local path`` for the files now present.
    """
    files = FILES if files is None else files
    os.makedirs(data_dir, exist_ok=True)
    paths: dict[str, str] = {}
    failures: list[str] = []

    for entry in files:
        file_id = entry["file_id"]
        dest = os.path.join(data_dir, f"{file_id}.mat")
        load = entry.get("load")
        suffix = f", load {load} HP" if load is not None else ""
        label = f"{entry['fault']}{suffix} ({file_id}.mat)"

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
    print(f"Ready: {len(paths)}/{len(files)} files in {data_dir}")
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
    import argparse

    parser = argparse.ArgumentParser(description="Download the CWRU bearing dataset.")
    parser.add_argument(
        "--all-loads", action="store_true",
        help="download all 16 files (4 classes x 4 motor loads) for the "
             "cross-load benchmark, not just the 4 load-0 recordings",
    )
    args = parser.parse_args()

    files = LOAD_FILES if args.all_loads else FILES
    result = download_dataset(files=files)
    sys.exit(0 if len(result) == len(files) else 1)
