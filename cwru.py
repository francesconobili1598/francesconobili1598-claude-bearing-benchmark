"""Load CWRU recordings as 12 kHz drive-end acceleration (g), handling the dataset's quirks.

Quirks handled here (documented in the README):
- The normal baseline files (97-100) are recorded at 48 kHz, the fault files at 12 kHz.
  Normal files are decimated by 4 so every contender sees the same sampling rate.
- 99.mat also contains a copy of 98's channels; channels are selected by file id.
- 98.mat and 99.mat have no RPM variable; the approximate speed from the CWRU table is used.
"""

import csv
import os

import numpy as np
import scipy.io as sio
from scipy.signal import decimate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cwru")
FS = 12000
LABELS = ["normal", "inner_race", "outer_race", "ball"]
# CWRU "Bearing Information" page: drive-end bearing 6205-2RS JEM SKF,
# defect frequencies as multiples of running speed.
BEARING = {"name": "SKF 6205-2RS JEM (drive end)", "bpfi": 5.4152, "bpfo": 3.5848, "ftf": 0.39828, "bsf": 4.7135}


def manifest():
    with open(os.path.join(HERE, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def load(row):
    """Return (x, rpm): drive-end acceleration at 12 kHz in g, and shaft speed in rpm."""
    fid = int(row["file_id"])
    m = sio.loadmat(os.path.join(DATA, f"{fid}.mat"))
    x = np.ravel(m[f"X{fid:03d}_DE_time"]).astype(float)
    if row["label"] == "normal":
        x = decimate(x, 4, ftype="fir", zero_phase=True)
    key = f"X{fid:03d}RPM"
    rpm = float(np.ravel(m[key])[0]) if key in m else float(row["approx_rpm"])
    return x, rpm


def segments(x, n=FS, hop=None):
    hop = hop or n
    return [x[i:i + n] for i in range(0, len(x) - n + 1, hop)]


def test_segment(x, n=FS):
    """The one segment every contender is scored on: one second from the middle of the recording."""
    start = (len(x) - n) // 2
    return x[start:start + n], start
