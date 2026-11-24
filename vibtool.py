"""vibtool: bearing vibration analysis from a raw acceleration signal.

Computes time-domain statistics, the bearing defect frequencies for the shaft speed,
an envelope spectrum (demodulation band chosen by kurtosis) and how far each defect
frequency stands above the local noise floor. It reports evidence, not a verdict.

CLI:  python vibtool.py signal.csv --fs 12000 --rpm 1797
      (one value per line, acceleration in g; default bearing = CWRU drive end 6205-2RS)
"""

import argparse
import json

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

DEFAULT_BEARING = {"name": "SKF 6205-2RS JEM (CWRU drive end)", "bpfi": 5.4152, "bpfo": 3.5848,
                   "ftf": 0.39828, "bsf": 4.7135}
BANDS = [(500, 1500), (1000, 2500), (2000, 4000), (3000, 5000), (4000, 5800)]
NAMED = ["BPFO", "BPFI", "2xBSF", "BSF", "FTF", "shaft"]


def time_stats(x):
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {"rms_g": rms, "peak_g": peak, "crest": peak / rms,
            "kurtosis": float(kurtosis(x, fisher=False)), "skew": float(skew(x))}


def amp_spectrum(x, fs):
    n = len(x)
    w = np.hanning(n)
    X = np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()
    return np.fft.rfftfreq(n, 1 / fs), X


def choose_band(x, fs):
    """Pick the band-pass band whose filtered signal is most impulsive (highest kurtosis)."""
    best = None
    for lo, hi in BANDS:
        hi = min(hi, 0.48 * fs)
        if lo >= hi:
            continue
        y = sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), x)
        k = float(kurtosis(y, fisher=False))
        if best is None or k > best[2]:
            best = (lo, hi, k, y)
    return best


def fault_freqs(rpm, b, line_hz=60.0):
    fr = rpm / 60
    return {"shaft": fr, "FTF": b["ftf"] * fr, "BPFO": b["bpfo"] * fr, "BPFI": b["bpfi"] * fr,
            "BSF": b["bsf"] * fr, "2xBSF": 2 * b["bsf"] * fr, "line": line_hz}


def line_strength(f, X, target, tol_frac=0.02, tol_min=1.5, ring=20):
    """Largest line within tolerance of target, in dB over the median of the surrounding ring."""
    tol = max(tol_frac * target, tol_min)
    sel = np.abs(f - target) <= tol
    if not sel.any():
        return None
    i = np.flatnonzero(sel)[np.argmax(X[sel])]
    around = (np.abs(f - target) > tol) & (np.abs(f - target) <= tol + ring)
    floor = float(np.median(X[around])) if around.any() else float(np.median(X))
    return {"freq": float(f[i]), "amp": float(X[i]),
            "db": float(20 * np.log10(X[i] / floor)) if floor > 0 else 0.0}


def nearest_named(freq, ff, tol_frac=0.02, tol_min=1.5):
    """All named frequencies (harmonics 1-4) within tolerance, e.g. '3x BPFO / 2x BPFI'."""
    hits = []
    for name in NAMED + ["line"]:
        for k in (1, 2, 3, 4):
            t = k * ff[name]
            if abs(freq - t) <= max(tol_frac * t, tol_min):
                hits.append(name if k == 1 else f"{k}x {name}")
    return " / ".join(hits)


def top_peaks(f, X, ff, fmin, fmax, n=10):
    floor = median_filter(X, size=41, mode="nearest")
    band = (f >= fmin) & (f <= fmax)
    idx, _ = find_peaks(X * band)
    idx = [i for i in idx if band[i] and floor[i] > 0]
    idx.sort(key=lambda i: X[i] / floor[i], reverse=True)
    return [{"freq": float(f[i]), "amp": float(X[i]), "db": float(20 * np.log10(X[i] / floor[i])),
             "match": nearest_named(f[i], ff)} for i in idx[:n]]


def analyse(x, fs, rpm, bearing=DEFAULT_BEARING):
    x = np.asarray(x, dtype=float)
    ff = fault_freqs(rpm, bearing)
    lo, hi, bk, y = choose_band(x, fs)
    env = np.abs(hilbert(y))
    fe, E = amp_spectrum(env, fs)
    f, X = amp_spectrum(x, fs)
    return {
        "n": len(x), "fs": fs, "rpm": rpm, "bearing": bearing["name"],
        "stats": time_stats(x),
        "fault_freqs": ff,
        "band": {"lo": lo, "hi": hi, "kurtosis": bk},
        "envelope_lines": {name: [line_strength(fe, E, k * ff[name]) for k in (1, 2, 3)] for name in NAMED},
        "bpfi_sidebands": [line_strength(fe, E, ff["BPFI"] + s * ff["shaft"]) for s in (-1, 1)],
        "envelope_peaks": top_peaks(fe, E, ff, 5, 700),
        "spectrum_peaks": top_peaks(f, X, ff, 5, 0.48 * fs),
    }


def report(a):
    s, ff, b = a["stats"], a["fault_freqs"], a["band"]
    L = [f"VIBRATION ANALYSIS REPORT ({a['bearing']})",
         f"signal: {a['n']} samples at {a['fs']} Hz ({a['n'] / a['fs']:.2f} s); shaft {a['rpm']:.0f} rpm = {ff['shaft']:.2f} Hz",
         f"time domain: RMS {s['rms_g']:.3f} g, peak {s['peak_g']:.3f} g, crest factor {s['crest']:.2f}, "
         f"kurtosis {s['kurtosis']:.2f} (about 3 for a healthy, non-impulsive signal), skew {s['skew']:.2f}",
         "defect frequencies (Hz): " + ", ".join(f"{k} {ff[k]:.1f}" for k in ["BPFO", "BPFI", "BSF", "2xBSF", "FTF"]),
         f"envelope analysis: demodulation band {b['lo']:.0f}-{b['hi']:.0f} Hz (band kurtosis {b['kurtosis']:.2f})",
         "envelope spectrum, defect frequencies and harmonics (dB above local floor; above ~6 dB is a clear line):"]
    for name in ["BPFO", "BPFI", "2xBSF", "FTF", "shaft"]:
        cells = []
        for k, v in zip((1, 2, 3), a["envelope_lines"][name]):
            cells.append(f"{k}x {v['freq']:.1f} Hz {v['db']:+.1f} dB" if v else f"{k}x n/a")
        L.append(f"  {name:<6} " + " | ".join(cells))
    sb = a["bpfi_sidebands"]
    L.append(f"  BPFI sidebands at -/+ shaft speed: {sb[0]['db']:+.1f} dB / {sb[1]['db']:+.1f} dB")
    L.append("strongest envelope-spectrum peaks (5-700 Hz):")
    for p in a["envelope_peaks"]:
        L.append(f"  {p['freq']:7.1f} Hz  {p['db']:+5.1f} dB  {p['match']}")
    L.append("strongest raw-spectrum peaks:")
    for p in a["spectrum_peaks"]:
        L.append(f"  {p['freq']:7.1f} Hz  {p['db']:+5.1f} dB  amp {p['amp']:.4f} g  {p['match']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("signal", help="CSV/TXT with one acceleration value (g) per line, or .npy")
    ap.add_argument("--fs", type=float, required=True, help="sampling rate in Hz")
    ap.add_argument("--rpm", type=float, required=True, help="shaft speed in rpm")
    ap.add_argument("--bearing", help='JSON with bpfi, bpfo, ftf, bsf multipliers, e.g. \'{"bpfi":5.4,...}\'')
    ap.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    args = ap.parse_args()
    x = np.load(args.signal) if args.signal.endswith(".npy") else np.loadtxt(args.signal, delimiter=",", ndmin=1)
    bearing = dict(DEFAULT_BEARING, **json.loads(args.bearing)) if args.bearing else DEFAULT_BEARING
    a = analyse(np.ravel(x), args.fs, args.rpm, bearing)
    print(json.dumps(a, indent=1) if args.json else report(a))


if __name__ == "__main__":
    main()
