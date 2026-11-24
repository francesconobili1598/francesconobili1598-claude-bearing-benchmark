"""Contenders 1 and 2: Claude, run headless through Claude Code with every tool disabled.

  claude_raw   the 12,000 raw samples of the centre segment, as numbers in the prompt
  claude_tool  the vibtool.py analysis report of the same segment (no raw numbers)

Both prompts carry the same context (machine, speed, bearing defect frequencies, label options).
No file names or labels reach the model. Results append to results/claude_predictions.jsonl,
so an interrupted run resumes where it stopped.

Usage: python run_claude.py [--contender raw|tool|both] [--model opus] [--workers 3] [--limit N]
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import tempfile

import cwru
import vibtool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claude_predictions.jsonl")
SYSTEM = "You are an expert vibration analyst. Follow the requested output format exactly."


def context(rpm):
    ff = vibtool.fault_freqs(rpm, cwru.BEARING)
    return (
        "You are diagnosing a rolling-element bearing from one vibration measurement.\n\n"
        "Machine: 2 hp induction motor test rig on a 60 Hz supply. Accelerometer on the drive-end bearing "
        "housing, acceleration in g.\n"
        f"Sampling rate: 12,000 Hz. Duration: 1.00 s (12,000 samples). Shaft speed: {rpm:.0f} rpm ({ff['shaft']:.2f} Hz).\n"
        "Bearing: SKF 6205-2RS deep groove ball bearing. Defect frequencies as multiples of shaft speed: "
        "inner race (BPFI) 5.4152, outer race (BPFO) 3.5848, ball spin (BSF) 4.7135, cage (FTF) 0.39828. "
        f"At this speed: BPFI {ff['BPFI']:.1f} Hz, BPFO {ff['BPFO']:.1f} Hz, BSF {ff['BSF']:.1f} Hz, FTF {ff['FTF']:.1f} Hz.\n\n"
        "Classify the bearing as exactly one of: normal, inner_race, outer_race, ball.\n\n"
    )


ANSWER = ('Answer with only a JSON object and nothing else: '
          '{"label": "normal|inner_race|outer_race|ball", "confidence": <0.0-1.0>, "reason": "<one sentence>"}')


def prompt(contender, seg, rpm):
    if contender == "raw":
        body = ("The signal, 12,000 values in time order, comma-separated:\n"
                + ",".join(f"{v:.4f}" for v in seg) + "\n\n")
    else:
        body = ("The output of a standard vibration analysis of this signal (time-domain statistics and "
                "envelope spectrum):\n\n" + vibtool.report(vibtool.analyse(seg, cwru.FS, rpm)) + "\n\n")
    return context(rpm) + body + ANSWER


def call_claude(text, model):
    exe = shutil.which("claude") or shutil.which("claude.exe")
    # --tools "" drops the built-in tools; --strict-mcp-config drops every MCP server (including
    # claude.ai connectors); --disable-slash-commands drops skills. Verified with check_isolation.py.
    cmd = [exe, "-p", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
           "--setting-sources", "", "--no-session-persistence",
           "--model", model, "--output-format", "json", "--system-prompt", SYSTEM]
    with tempfile.TemporaryDirectory() as tmp:  # neutral working directory: no project files around
        p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", cwd=tmp, timeout=900)
    d = json.loads(p.stdout)
    if d.get("is_error"):
        raise RuntimeError(d.get("result"))
    return d


def parse(result):
    m = re.search(r"\{.*\}", result, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    label = str(j.get("label", "")).strip().lower()
    return (label if label in cwru.LABELS else "unparsed"), j.get("confidence"), j.get("reason", result[:300])


def done_keys():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {(j["contender"], j["file_id"]) for j in map(json.loads, f)}


def job(contender, row, model):
    x, rpm = cwru.load(row)
    seg, start = cwru.test_segment(x)
    d = call_claude(prompt(contender, seg, rpm), model)
    label, conf, reason = parse(d.get("result", ""))
    return {"contender": f"claude_{contender}", "file_id": row["file_id"], "source_name": row["source_name"],
            "label": row["label"], "pred": label, "correct": int(label == row["label"]),
            "confidence": conf, "reason": reason, "segment_start": start,
            "models": list((d.get("modelUsage") or {}).keys()), "usage": d.get("usage"),
            "cost_usd": d.get("total_cost_usd"), "raw_result": d.get("result")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contender", choices=["raw", "tool", "both"], default="both")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N recordings (smoke test)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = cwru.manifest()[: args.limit or None]
    contenders = ["tool", "raw"] if args.contender == "both" else [args.contender]
    done = done_keys()
    todo = [(c, r) for c in contenders for r in rows if (f"claude_{c}", r["file_id"]) not in done]
    print(f"{len(todo)} calls to make ({len(done)} already done)")
    with cf.ThreadPoolExecutor(args.workers) as ex, open(OUT, "a", encoding="utf-8") as out:
        futs = {ex.submit(job, c, r, args.model): (c, r["source_name"]) for c, r in todo}
        for fut in cf.as_completed(futs):
            c, name = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"FAILED {c} {name}: {e}")
                continue
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(f"{res['contender']:<12} {name:<11} truth {res['label']:<10} pred {res['pred']:<10} "
                  f"{'OK' if res['correct'] else 'MISS'}  conf {res['confidence']}")


if __name__ == "__main__":
    main()
