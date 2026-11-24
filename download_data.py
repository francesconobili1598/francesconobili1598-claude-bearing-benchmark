"""Download the CWRU bearing files listed in manifest.csv into data/cwru/.

Source: Case Western Reserve University Bearing Data Center
https://engineering.case.edu/bearingdatacenter
The files are not redistributed in this repo; this script fetches them from the source.
"""

import csv
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "cwru")
URL = "https://engineering.case.edu/sites/default/files/{}.mat"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"), newline="")))
    total = 0
    for r in rows:
        dst = os.path.join(OUT, f"{r['file_id']}.mat")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            total += os.path.getsize(dst)
            continue
        for attempt in range(3):
            try:
                req = urllib.request.Request(URL.format(r["file_id"]), headers={"User-Agent": "bearing-benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                break
            except Exception as e:  # network hiccup: retry
                print(f"  retry {r['file_id']}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            sys.exit(f"failed to download {r['file_id']}.mat")
        if not data.startswith(b"MATLAB 5.0 MAT-file"):
            sys.exit(f"{r['file_id']}.mat is not a MATLAB file, stopping")
        with open(dst, "wb") as f:
            f.write(data)
        total += len(data)
        print(f"{r['file_id']}.mat  {r['source_name']:<11} {len(data) / 1e6:5.1f} MB")
    print(f"done: {len(rows)} files, {total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
