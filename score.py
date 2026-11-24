"""Scoreboard: combine every contender's predictions into one table and one chart.

Reads   results/ml_predictions.csv, results/claude_predictions.jsonl
Writes  results/scoreboard.csv, results/scoreboard.md, results/accuracy.png

Works on a partial Claude run: each contender is scored on what has finished, and n is shown.
The leaky random-split number is computed once by baseline_ml and cached in results/ml_summary.json.
"""

import csv
import json
import math
import os

import cwru

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CONTENDERS = [  # fixed order: the story reads top to bottom
    ("ml_leaky", "Classic ML, random split", "1-s pieces of the same recording in train and test (leaky)"),
    ("ml_by_load", "Classic ML, unseen motor load", "same damaged bearing seen in training at other loads"),
    ("ml_by_bearing", "Classic ML, unseen bearing", "test bearing never seen in training"),
    ("claude_raw", "Claude, raw numbers", "12,000 samples pasted in the prompt, no tools, no training"),
    ("claude_tool", "Claude + vibtool report", "one-page envelope analysis report, no tools, no training"),
]


def leaky_summary():
    path = os.path.join(RES, "ml_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import baseline_ml  # slow path: rebuilds features once
    recs = baseline_ml.build()
    _, acc = baseline_ml.run_split(recs, "leaky")
    n = sum(len(r["train"]) for r in recs)
    s = {"leaky_accuracy": acc, "leaky_n_test": math.ceil(0.3 * n), "segments": n}
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
    return s


def load_predictions():
    rows = {}
    with open(os.path.join(RES, "ml_predictions.csv")) as f:
        for r in csv.DictReader(f):
            r["correct"] = int(r["correct"])
            rows[(r["contender"], r["file_id"])] = r
    path = os.path.join(RES, "claude_predictions.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                rows[(j["contender"], j["file_id"])] = j  # last write wins
    return list(rows.values())


def prompt_tokens(u):
    u = u or {}
    return sum(u.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def summarise(preds, leaky):
    out = []
    for key, name, note in CONTENDERS:
        if key == "ml_leaky":
            n = leaky["leaky_n_test"]
            out.append({"key": key, "name": name, "note": note, "n": n,
                        "correct": round(leaky["leaky_accuracy"] * n), "accuracy": leaky["leaky_accuracy"]})
            continue
        sub = [p for p in preds if p["contender"] == key]
        if not sub:
            continue
        s = {"key": key, "name": name, "note": note, "n": len(sub),
             "correct": sum(p["correct"] for p in sub), "accuracy": sum(p["correct"] for p in sub) / len(sub)}
        for lab in cwru.LABELS:
            c = [p for p in sub if p["label"] == lab]
            s[lab] = f"{sum(p['correct'] for p in c)}/{len(c)}"
        if key.startswith("claude"):
            conf = lambda ok: [float(p["confidence"]) for p in sub
                               if p["correct"] == ok and isinstance(p.get("confidence"), (int, float))]
            right, wrong = conf(1), conf(0)
            s["unparsed"] = sum(p["pred"] == "unparsed" for p in sub)
            s["conf_right"] = sum(right) / len(right) if right else None
            s["conf_wrong"] = sum(wrong) / len(wrong) if wrong else None
            s["prompt_tokens"] = sum(prompt_tokens(p.get("usage")) for p in sub) / len(sub)
            s["cost_usd"] = sum(p.get("cost_usd") or 0 for p in sub)
            s["models"] = ", ".join(sorted({m for p in sub for m in p.get("models", [])}))
        out.append(s)
    return out


def write_tables(summary, preds):
    fields = ["key", "name", "n", "correct", "accuracy"] + cwru.LABELS + \
             ["unparsed", "conf_right", "conf_wrong", "prompt_tokens", "cost_usd", "models", "note"]
    with open(os.path.join(RES, "scoreboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    pct = lambda v: f"{v:.0%}" if isinstance(v, float) else ""
    num = lambda v, fmt: format(v, fmt) if isinstance(v, (int, float)) else ""
    L = ["# Bearing benchmark scoreboard", "",
         "CWRU drive-end bearing, 40 recordings (4 healthy, 12 inner race, 12 outer race, 12 ball faults), "
         "scored on the same 1-second centre segment of each recording.", "",
         "| Contender | Accuracy | Correct | Normal | Inner race | Outer race | Ball | Setup |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['name']} | {pct(s['accuracy'])} | {s['correct']}/{s['n']} | {s.get('normal', '')} | "
                 f"{s.get('inner_race', '')} | {s.get('outer_race', '')} | {s.get('ball', '')} | {s['note']} |")
    claude = [s for s in summary if s["key"].startswith("claude")]
    if claude:
        L += ["", "## Claude runs", "",
              "| Contender | Model | Avg prompt tokens | Confidence when right | Confidence when wrong | Unparsed | Cost (USD, API list price) |",
              "|---|---|---|---|---|---|---|"]
        for s in claude:
            L.append(f"| {s['name']} | {s['models']} | {num(s['prompt_tokens'], ',.0f')} | {num(s['conf_right'], '.2f')} | "
                     f"{num(s['conf_wrong'], '.2f')} | {s['unparsed']} | {num(s['cost_usd'], '.2f')} |")
    keys = [k for k, _, _ in CONTENDERS if k != "ml_leaky" and any(p["contender"] == k for p in preds)]
    by = {(p["contender"], p["file_id"]): p for p in preds}
    L += ["", "## Every recording", "", "Misses in bold.", "",
          "| Recording | Truth | " + " | ".join(dict((k, n) for k, n, _ in CONTENDERS)[k] for k in keys) + " |",
          "|---|---|" + "---|" * len(keys)]
    for r in cwru.manifest():
        cells = []
        for k in keys:
            p = by.get((k, r["file_id"]))
            cells.append("" if p is None else (p["pred"] if p["correct"] else f"**{p['pred']}**"))
        L.append(f"| {r['source_name']} | {r['label']} | " + " | ".join(cells) + " |")
    with open(os.path.join(RES, "scoreboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    ACCENT, CONTEXT = "#2a78d6", "#b5b4ad"  # emphasis pattern: Claude in the accent, baselines in gray
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})

    rows = list(reversed(summary))  # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(12, 1.15 * len(rows) + 2.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.52
    for i, s in enumerate(rows):
        w = s["accuracy"] * 100
        color = ACCENT if s["key"].startswith("claude") else CONTEXT
        r = min(1.2, w / 2)  # rounded data end, square at the baseline
        hatch = "//" if s["key"] == "ml_leaky" else None  # texture marks the invalid (leaky) setup
        ax.add_patch(FancyBboxPatch((0, i - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=h / 6, fc=color, ec=SURFACE, lw=0, hatch=hatch, zorder=2))
        ax.add_patch(Rectangle((0, i - h / 2), min(w, 6), h, fc=color, ec=SURFACE, lw=0, hatch=hatch, zorder=2))
        label = f"{w:.0f}%  ({s['correct']}/{s['n']})"
        ax.text(w + 1.2, i, label, va="center", ha="left", color=INK, fontsize=15, fontweight="bold", zorder=3, clip_on=False)
        ax.text(-1.5, i + 0.08, s["name"], va="center", ha="right", color=INK, fontsize=15, zorder=3)
        ax.text(-1.5, i - 0.27, s["note"], va="center", ha="right", color=MUTED, fontsize=10.5, zorder=3)
    ax.axvline(25, color=AXIS, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(25.6, len(rows) - 0.45, "chance (4 classes)", color=MUTED, fontsize=11, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=12)
    ax.set_yticks([])
    for x in (50, 75, 100):
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", length=0, pad=6)
    fig.text(0.02, 0.965, "Can Claude diagnose a bearing from vibration data?", color=INK, fontsize=21,
             fontweight="bold", va="top")
    fig.text(0.02, 0.905, "Accuracy on 40 recordings from the CWRU bearing dataset: healthy, inner race, "
             "outer race, ball fault", color=INK2, fontsize=13, va="top")
    fig.text(0.02, 0.025, "Same 1-second segment per recording for every contender. Random split scored on "
             "random 1-s segments. Code and method: see repo.", color=MUTED, fontsize=10, va="bottom")
    fig.subplots_adjust(left=0.40, right=0.83, top=0.80, bottom=0.14)
    fig.savefig(os.path.join(RES, "accuracy.png"), facecolor=SURFACE)
    plt.close(fig)


def main():
    leaky = leaky_summary()
    preds = load_predictions()
    summary = summarise(preds, leaky)
    write_tables(summary, preds)
    chart(summary)
    for s in summary:
        print(f"{s['name']:<32} {s['accuracy']:>5.0%}  {s['correct']}/{s['n']}")
    print("wrote results/scoreboard.md, results/scoreboard.csv, results/accuracy.png")


if __name__ == "__main__":
    main()
