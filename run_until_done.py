"""
run_until_done.py — auto-retry runner for moedmpnn_regr.py

Keeps re-launching moedmpnn_regr.py until results_moedmpnn_regr.json has
all 3 datasets (ESOL, FreeSolv, Lipo) tagged with the fixed backbone.
Relies on the script's own resume logic to skip already-completed datasets,
so each retry only re-attempts whatever failed last time.

Usage:
    conda activate moe_admet
    cd D:\\molprop_project\\attentivefp-multitask-admet
    python run_until_done.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = "moedmpnn_regr.py"
RESULTS_FILE = Path("results_moedmpnn_regr.json")
REQUIRED_DATASETS = ["ESOL", "FreeSolv", "Lipo"]
BACKBONE_TAG = "true_dmpnn_nnconv_grucell"
MAX_ATTEMPTS = 6


def check_done():
    if not RESULTS_FILE.exists():
        return False, []
    try:
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] couldn't parse {RESULTS_FILE}: {e}")
        return False, []
    missing = [
        d for d in REQUIRED_DATASETS
        if d not in data or data[d].get("backbone") != BACKBONE_TAG
    ]
    return len(missing) == 0, missing


def main():
    for attempt in range(1, MAX_ATTEMPTS + 1):
        done, missing = check_done()
        if done:
            print(f"\nAll datasets complete with fixed backbone: {REQUIRED_DATASETS}")
            return
        print(f"\n=== Attempt {attempt}/{MAX_ATTEMPTS} — missing: {missing} ===")

        result = subprocess.run(
            [sys.executable, SCRIPT],
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            print(f"  Run finished cleanly (exit code 0).")
        else:
            print(f"  Run exited with code {result.returncode} (likely crashed, e.g. OOM).")

        done, missing = check_done()
        if done:
            print(f"\nAll datasets complete with fixed backbone: {REQUIRED_DATASETS}")
            return

        print(f"  Still missing: {missing}. Retrying in 10s...")
        time.sleep(10)

    print(f"\nGave up after {MAX_ATTEMPTS} attempts. Still missing: {check_done()[1]}")
    print("Manual intervention needed — check the last error above.")


if __name__ == "__main__":
    main()
