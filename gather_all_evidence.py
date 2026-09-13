"""
gather_all_evidence.py

Final evidence pass before writing anything to Prof. Li. Covers every
still-open review item:

  2.1  find the script/log that produced Table 1's MoE-GCN numbers
       (to compare its hyperparams against ablation_optuna_results.json)
  2.2  find where "0.461" (Caco-2 MAE, pre-routing) actually comes from
  2.3  dump every LogP / aromatic-ring eta2 value we have, across datasets,
       so the cross-dataset average can be recomputed by hand
  2.5  check whether expanded_descriptor_results.json actually has 8
       descriptors x 4 datasets, or fewer
  2.6  list every per-dataset TDC result file/value found
  2.7  check whether AttentiveFP/GROVER numbers have any std-dev source
       anywhere, or are hardcoded literature values with no run log

Usage:
    conda activate moe_admet
    cd D:\\molprop_project
    python gather_all_evidence.py > evidence_report.txt
    (also prints to console; redirect to a file so nothing gets lost)
"""

import json
import os
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\molprop_project")


def ts(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "N/A"


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_files(root, name_substrings, exts=None):
    """Walk root, return files whose lowercased name contains any of the
    substrings and (optionally) matches an extension."""
    hits = []
    skip = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "venv", "env"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fname in filenames:
            fname_l = fname.lower()
            if exts and Path(fname).suffix.lower() not in exts:
                continue
            if any(s in fname_l for s in name_substrings):
                hits.append(Path(dirpath) / fname)
    return hits


# ---------------------------------------------------------------
# 2.1 — find whatever produced Table 1's MoE-GCN numbers
# ---------------------------------------------------------------
section("2.1 — Scripts that could have produced Table 1's MoE-GCN column")
candidates = find_files(ROOT, ["moegcn", "moe_gcn"], exts={".py"})
for c in candidates:
    print(f"  [{ts(c)}] {c}")
print("\n  --> Open each of these and check num_experts/top_k/hidden defaults")
print("      against ablation_optuna_results.json's ESOL.moe.best_params")
print("      (hidden=256, num_layers=3, num_experts=4, top_k=3).")
print("      If defaults differ, that confirms independent HPO runs.")

# ---------------------------------------------------------------
# 2.2 — hunt for "0.461" or any Caco-2 MAE near it
# ---------------------------------------------------------------
section("2.2 — Every Caco-2 MAE-like value found in JSON results")
caco_files = find_files(ROOT, ["caco"], exts={".json"})
for f in caco_files:
    data = load_json(f)
    if "__error__" in data:
        continue
    flat = json.dumps(data)
    if "caco" in str(f).lower():
        print(f"  [{ts(f)}] {f}")
        # print any numeric value between 0.3 and 0.6 (plausible MAE range)
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            elif isinstance(obj, (int, float)):
                if 0.30 <= obj <= 0.65:
                    print(f"      {path} = {obj}")
        walk(data)
        print()

# ---------------------------------------------------------------
# 2.3 — dump every LogP / aromatic ring eta2 value
# ---------------------------------------------------------------
section("2.3 — LogP / aromatic-ring eta2 values across all descriptor files")
desc_files = find_files(ROOT, ["expanded_descriptor", "expert_specialization"], exts={".json"})
for f in desc_files:
    data = load_json(f)
    if "__error__" in data:
        print(f"  [{ts(f)}] {f}  ERROR: {data['__error__']}")
        continue
    flat = json.dumps(data).lower()
    if "logp" in flat or "aromatic" in flat or "eta" in flat:
        print(f"  [{ts(f)}] {f}")
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    kl = str(k).lower()
                    newpath = f"{path}.{k}" if path else k
                    if isinstance(v, (dict, list)):
                        walk(v, newpath)
                    elif "logp" in kl or "aromatic" in kl or "eta" in path.lower():
                        print(f"      {newpath} = {v}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
        walk(data)
        print()

# ---------------------------------------------------------------
# 2.5 — descriptor coverage check: how many descriptors x datasets?
# ---------------------------------------------------------------
section("2.5 — Descriptor coverage in expanded_descriptor_results.json")
target = ROOT / "attentivefp-multitask-admet" / "expanded_descriptor_results.json"
if not target.exists():
    target = ROOT / "expanded_descriptor_results.json"
if target.exists():
    data = load_json(target)
    if "__error__" not in data:
        print(f"  File: {target}  [{ts(target)}]")
        if isinstance(data, dict):
            print(f"  Top-level keys (datasets?): {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"    {k}: descriptor keys = {list(v.keys())}")
    else:
        print(f"  ERROR reading: {data['__error__']}")
else:
    print("  ** FILE NOT FOUND **")

# ---------------------------------------------------------------
# 2.6 — every per-dataset TDC result located
# ---------------------------------------------------------------
section("2.6 — TDC per-dataset result files and their dataset coverage")
tdc_files = find_files(ROOT, ["results_gcn_tdc", "results_moegcn_tdc", "results_tdc"], exts={".json"})
for f in tdc_files:
    data = load_json(f)
    print(f"  [{ts(f)}] {f}")
    if "__error__" in data:
        print(f"      ERROR: {data['__error__']}")
        continue
    if isinstance(data, dict):
        print(f"      Datasets covered ({len(data)}): {list(data.keys())}")
    print()

# ---------------------------------------------------------------
# 2.7 — AttentiveFP / GROVER source check
# ---------------------------------------------------------------
section("2.7 — AttentiveFP / GROVER results: any run log, or hardcoded?")
af_files = find_files(ROOT, ["attentivefp"], exts={".py", ".json"})
grover_files = find_files(ROOT, ["grover"], exts={".py", ".json"})
print("  AttentiveFP-related files:")
for f in af_files:
    print(f"    [{ts(f)}] {f}")
print("\n  GROVER-related files:")
for f in grover_files:
    print(f"    [{ts(f)}] {f}")
gr = ROOT / "attentivefp-multitask-admet" / "grover_results.json"
if not gr.exists():
    gr = ROOT / "grover_results.json"
if gr.exists():
    data = load_json(gr)
    print(f"\n  Contents of {gr}:")
    print(f"  {json.dumps(data, indent=2)[:2000]}")

print("\n\nDONE. Review each section above before drafting anything to Prof. Li.")
