"""Contender 3: classic machine learning (random forest on standard vibration features).

Scored on the same centre segment of each of the 40 recordings as the Claude contenders.
Three splits, to show how much the split matters on CWRU:
  leaky      random split of 1-second segments: pieces of the same recording land in train and test
  by_load    leave-one-motor-load-out: the test recording's load is never seen in training
             (but the same damaged bearing is, at other loads)
  by_bearing leave-one-fault-size-out: the test bearing is never seen in training (strictest)
"""

import csv
import os

import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import cwru
import vibtool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def features(seg, rpm):
    ff = vibtool.fault_freqs(rpm, cwru.BEARING)
    s = vibtool.time_stats(seg)
    lo, hi, bk, y = vibtool.choose_band(seg, cwru.FS)
    fe, E = vibtool.amp_spectrum(np.abs(vibtool.hilbert(y)), cwru.FS)
    env = [vibtool.line_strength(fe, E, k * ff[n])["db"] for n in ["BPFO", "BPFI", "2xBSF", "FTF"] for k in (1, 2, 3)]
    sb = [vibtool.line_strength(fe, E, ff["BPFI"] + d * ff["shaft"])["db"] for d in (-1, 1)]
    f, X = vibtool.amp_spectrum(seg, cwru.FS)
    edges = np.linspace(0, cwru.FS / 2, 13)
    bands = [np.log10(np.sum(X[(f >= a) & (f < b)] ** 2) + 1e-12) for a, b in zip(edges[:-1], edges[1:])]
    return [s["rms_g"], s["peak_g"], s["crest"], s["kurtosis"], s["skew"], bk, lo] + env + sb + bands


def build():
    recs = []
    for r in cwru.manifest():
        x, rpm = cwru.load(r)
        test_seg, start = cwru.test_segment(x)
        train_segs = cwru.segments(x)
        recs.append({"row": r, "rpm": rpm,
                     "test": features(test_seg, rpm),
                     "train": [features(s, rpm) for s in train_segs]})
    return recs


def fit_predict(train_X, train_y, test_X):
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)


def fold_of(row, split):
    if split == "by_load":
        return row["load_hp"]
    # by_bearing: each fault size is a different physical bearing; spread the 4 normal loads across folds
    if row["label"] == "normal":
        return {"0": "0.007", "1": "0.014", "2": "0.021", "3": "0.021"}[row["load_hp"]]
    return row["fault_size_in"]


def run_split(recs, split):
    preds = {}
    if split == "leaky":
        X = [f for r in recs for f in r["train"]]
        y = [r["row"]["label"] for r in recs for _ in r["train"]]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        acc = float(np.mean(fit_predict(Xtr, ytr, Xte) == np.array(yte)))
        return None, acc
    for fold in sorted({fold_of(r["row"], split) for r in recs}):
        train = [r for r in recs if fold_of(r["row"], split) != fold]
        test = [r for r in recs if fold_of(r["row"], split) == fold]
        X = [f for r in train for f in r["train"]]
        y = [r["row"]["label"] for r in train for _ in r["train"]]
        for r, p in zip(test, fit_predict(X, y, [r["test"] for r in test])):
            preds[r["row"]["file_id"]] = p
    acc = float(np.mean([preds[r["row"]["file_id"]] == r["row"]["label"] for r in recs]))
    return preds, acc


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = build()
    print(f"{len(recs)} recordings, {sum(len(r['train']) for r in recs)} one-second segments")
    rows = []
    for split in ["leaky", "by_load", "by_bearing"]:
        preds, acc = run_split(recs, split)
        print(f"{split:<11} accuracy {acc:.1%}")
        if preds:
            for r in recs:
                rr = r["row"]
                rows.append({"contender": f"ml_{split}", "file_id": rr["file_id"], "source_name": rr["source_name"],
                             "label": rr["label"], "pred": preds[rr["file_id"]],
                             "correct": int(preds[rr["file_id"]] == rr["label"])})
    with open(os.path.join(OUT, "ml_predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in ["by_load", "by_bearing"]:
        sub = [r for r in rows if r["contender"] == f"ml_{split}"]
        per = {lab: f"{sum(r['correct'] for r in sub if r['label'] == lab)}/{sum(1 for r in sub if r['label'] == lab)}"
               for lab in cwru.LABELS}
        print(f"  {split} per class: {per}")
        print("  misses:", [(r["source_name"], r["pred"]) for r in sub if not r["correct"]])


if __name__ == "__main__":
    main()
