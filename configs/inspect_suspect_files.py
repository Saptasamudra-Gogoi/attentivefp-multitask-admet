"""
inspect_suspect_files.py

For review items 2.1, 2.2, 2.8: several result files exist in more than one
location/version (root-level vs attentivefp-multitask-admet/, or base vs
"_fixed"). This script loads each candidate JSON, prints its FreeSolv /
Caco-2 / dense-ablation values with the file's mtime, so we can see directly
whether "3.432" is a re-run number or a copy-forward, and where the Table 1
vs Table 6 discrepancy comes from.

Usage:
    conda activate moe_admet
    cd D:\\molprop_project
    python inspect_suspect_files.py
"""

import json
import os
from pathlib import Path

ROOT = Path(r"D:\molprop_project")

# Candidate files to inspect, grouped by review item.
# Paths are relative to ROOT; script skips any that don't exist.
CANDIDATES = {
    "2.8 — DMPNN / MoE-DMPNN regression (FreeSolv)": [
        "results_dmpnn_regr.json",
        "results_moedmpnn_regr.json",
        "attentivefp-multitask-admet/results_dmpnn_regr.json",
        "attentivefp-multitask-admet/results_moedmpnn_regr.json",
    ],
    "2.8 — DMPNN / MoE-DMPNN classification (sanity check, same fix)": [
        "results_dmpnn_classif.json",
        "results_moedmpnn_classif.json",
        "attentivefp-multitask-admet/results_dmpnn_classif.json",
        "attentivefp-multitask-admet/results_moedmpnn_classif.json",
    ],
    "2.1 — dense/parameter-matched ablation (Table 6 source)": [
        "random_ablation_results.json",
        "random_ablation_results_fixed.json",
        "attentivefp-multitask-admet/random_ablation_results.json",
        "attentivefp-multitask-admet/random_ablation_results_fixed.json",
        "ablation_routing_results.json",
        "attentivefp-multitask-admet/ablation_routing_results.json",
        "ablation_optuna_results.json",
        "attentivefp-multitask-admet/ablation_optuna_results.json",
        "cross_arch_results.json",
        "attentivefp-multitask-admet/cross_arch_results.json",
    ],
    "2.1/2.7 — Table 1 final results": [
        "final_results_table.csv",
        "attentivefp-multitask-admet/final_results_table.csv",
    ],
}


def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def find_freesolv_like(obj, path=""):
    """Recursively search a JSON structure for keys mentioning freesolv/caco/dense/gcn
    and return flattened key->value pairs for anything numeric or short."""
    hits = []
    KEYS_OF_INTEREST = ["freesolv", "caco", "dense", "gcn", "mae", "rmse", "esol", "lipo"]
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_lower = str(k).lower()
            new_path = f"{path}.{k}" if path else str(k)
            if isinstance(v, (dict, list)):
                hits.extend(find_freesolv_like(v, new_path))
            else:
                if any(kw in k_lower for kw in KEYS_OF_INTEREST) or any(kw in path.lower() for kw in KEYS_OF_INTEREST):
                    hits.append((new_path, v))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(find_freesolv_like(item, f"{path}[{i}]"))
    return hits


def print_file_report(rel_path):
    full_path = ROOT / rel_path
    if not full_path.exists():
        print(f"  [MISSING] {rel_path}")
        return

    mtime = os.path.getmtime(full_path)
    from datetime import datetime
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    size = full_path.stat().st_size

    print(f"  [{mtime_str}] ({size}B) {rel_path}")

    if full_path.suffix == ".json":
        data = load_json_safe(full_path)
        if "__error__" in data:
            print(f"      ERROR reading JSON: {data['__error__']}")
            return
        hits = find_freesolv_like(data)
        if hits:
            for k, v in hits[:30]:
                print(f"      {k} = {v}")
            if len(hits) > 30:
                print(f"      ... and {len(hits) - 30} more matches")
        else:
            # fallback: print top-level keys so we can see structure
            if isinstance(data, dict):
                print(f"      (no keyword match; top-level keys: {list(data.keys())[:20]})")
            elif isinstance(data, list):
                print(f"      (no keyword match; list of {len(data)} items, first item: {data[0] if data else 'EMPTY'})")
    elif full_path.suffix == ".csv":
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"      ({len(lines)} lines) preview:")
        for line in lines[:15]:
            print(f"      {line.rstrip()}")
        if len(lines) > 15:
            print(f"      ... and {len(lines) - 15} more lines")


def main():
    for label, files in CANDIDATES.items():
        print("=" * 70)
        print(label)
        print("=" * 70)
        for rel_path in files:
            print_file_report(rel_path)
            print()
        print()


if __name__ == "__main__":
    main()
