"""
gather_evidence_round2.py

Fixes the path-matching bug from round 1 (2.3/2.5 came back empty) and
closes out 2.1 (hyperparameter defaults) and 2.2 (hunt for 0.461 inside
the plain-GCN TDC baseline specifically).

Usage:
    conda activate moe_admet
    cd D:\\molprop_project
    python gather_evidence_round2.py > evidence_report_round2.txt
    type evidence_report_round2.txt
"""

import json
import os
import re
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


def walk_all_numeric(obj, path=""):
    """Yield (path, value) for every numeric leaf, no filtering."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            newpath = f"{path}.{k}" if path else str(k)
            out.extend(walk_all_numeric(v, newpath))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(walk_all_numeric(v, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)):
        out.append((path, obj))
    return out


# ---------------------------------------------------------------
# 2.3 (fixed) — every eta2 / LogP / aromatic value, properly extracted
# ---------------------------------------------------------------
section("2.3 (FIXED) — every eta2-like numeric value, LogP + aromatic focus")
desc_files = list(ROOT.rglob("expert_specialization_*.json")) + list(ROOT.rglob("expanded_descriptor_results.json"))
seen = set()
for f in desc_files:
    if f.name in seen and "attentivefp-multitask-admet" in str(f):
        continue  # skip exact duplicate copies inside repo folder for brevity
    seen.add(f.name)
    data = load_json(f)
    if "__error__" in data:
        print(f"  [{ts(f)}] {f}  ERROR: {data['__error__']}")
        continue
    all_nums = walk_all_numeric(data)
    relevant = [(p, v) for p, v in all_nums if any(kw in p.lower() for kw in ["logp", "aromatic", "arring", "eta2", "eta_2"])]
    if relevant:
        print(f"  [{ts(f)}] {f}")
        for p, v in relevant:
            print(f"      {p} = {v}")
        print()

# ---------------------------------------------------------------
# 2.5 (fixed) — descriptor coverage, handling list-or-dict shape
# ---------------------------------------------------------------
section("2.5 (FIXED) — expanded_descriptor_results.json structure + coverage")
target = ROOT / "attentivefp-multitask-admet" / "expanded_descriptor_results.json"
data = load_json(target)
if "__error__" in data:
    print(f"  ERROR: {data['__error__']}")
else:
    print(f"  Top-level type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"  Top-level keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"    '{k}' -> sub-keys: {list(v.keys())}")
            elif isinstance(v, list):
                print(f"    '{k}' -> list of {len(v)} items; first item: {v[0] if v else 'EMPTY'}")
    elif isinstance(data, list):
        print(f"  List of {len(data)} items")
        if data:
            print(f"  First item: {json.dumps(data[0], indent=2)[:1000]}")

# ---------------------------------------------------------------
# 2.2 (continued) — check plain GCN TDC baseline for Caco-2, hunt "0.461"
# ---------------------------------------------------------------
section("2.2 (continued) — plain GCN (non-MoE) Caco-2 value + literal 0.461 search")
gcn_tdc = ROOT / "results_gcn_tdc.json"
data = load_json(gcn_tdc)
if "__error__" not in data and "caco2_wang" in data:
    print(f"  results_gcn_tdc.json -> caco2_wang entry:")
    print(f"  {json.dumps(data['caco2_wang'], indent=2)}")
else:
    print(f"  caco2_wang not found in results_gcn_tdc.json top-level, or error: {data.get('__error__')}")

print("\n  Searching ALL json/txt/csv/md files under project root for literal '0.461' or '0.46' substring...")
hits = []
skip = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "venv", "env"}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for fname in filenames:
        if Path(fname).suffix.lower() not in {".json", ".txt", ".csv", ".md"}:
            continue
        fpath = Path(dirpath) / fname
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            if "0.461" in content or re.search(r"0\.46\d", content):
                for m in re.finditer(r".{30}0\.46\d.{30}", content):
                    hits.append((fpath, m.group()))
        except Exception:
            continue
for fpath, snippet in hits[:40]:
    print(f"  [{ts(fpath)}] {fpath}")
    print(f"      ...{snippet}...")
if not hits:
    print("  ** NO FILE CONTAINS 0.46x ANYWHERE IN THE PROJECT **")

# ---------------------------------------------------------------
# 2.1 (continued) — dump argparse/default hyperparams from moegcn scripts
# ---------------------------------------------------------------
section("2.1 (continued) — hyperparameter defaults in MoE-GCN scripts")
scripts_to_check = [
    ROOT / "moegcn_regr.py",
    ROOT / "pharma_moe_gcn.py",
    ROOT / "run_moegcn_tdc_benchmark.py",
]
for s in scripts_to_check:
    if not s.exists():
        print(f"  [MISSING] {s}")
        continue
    print(f"\n  --- {s} [{ts(s)}] ---")
    with open(s, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # grep lines mentioning num_experts, top_k, hidden, default=
    for lineno, line in enumerate(content.splitlines(), 1):
        if re.search(r"(num_experts|top_k|hidden|default\s*=)", line, re.IGNORECASE):
            print(f"      L{lineno}: {line.strip()}")

print("\n\nDONE — round 2.")
