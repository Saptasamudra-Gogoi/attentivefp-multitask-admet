"""
scan_results.py
---------------
Run from your project root (where your results JSON files live).
Tells you:
  1. Which JSON result files exist
  2. Which models are present in each file
  3. Which datasets are covered
  4. Whether plain GCN TDC baselines exist
  5. A summary table of what's missing

Usage:
    python scan_results.py
    python scan_results.py --results_dir path/to/your/results
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── TDC datasets we expect ────────────────────────────────────────────────────
TDC_DATASETS = {
    "hia_hou", "pgp_broccatelli", "bioavailability_ma", "bbb_martins",
    "cyp2c9_veith", "cyp3a4_veith", "cyp2d6_veith", "cyp2d6_substrate",
    "cyp2c9_substrate", "cyp3a4_substrate", "herg", "ames", "dili",
    "caco2_wang", "lipophilicity_az", "solubility_aqsoldb", "ppbr_az",
    "vdss_lombardo", "half_life_obach", "clearance_microsome",
    "clearance_hepatocyte", "ld50_zhu"
}

# ── MoleculeNet datasets we expect ────────────────────────────────────────────
MOLECULENET_DATASETS = {
    "bbbp", "bace", "tox21", "toxcast", "sider", "clintox", "hiv",
    "esol", "freesolv", "lipo"
}

# ── Models we care about ──────────────────────────────────────────────────────
MODELS_OF_INTEREST = {
    "gcn", "plain_gcn", "baseline_gcn",           # plain GCN variants
    "moe_gcn", "moe-gcn",                          # MoE-GCN
    "dmpnn", "plain_dmpnn",                        # plain DMPNN
    "moe_dmpnn", "moe-dmpnn",                      # MoE-DMPNN
    "gin", "plain_gin",                            # plain GIN
    "moe_gin", "moe-gin",                          # MoE-GIN
    "attentivefp", "attfp",                        # AttentiveFP
    "grover",                                      # GROVER
}

def find_json_files(root: Path):
    """Recursively find all JSON files."""
    return sorted(root.rglob("*.json"))

def extract_info(data, filepath):
    """
    Try multiple common result file formats:
      - {dataset: {model: {metric: value}}}
      - {model: {dataset: {metric: value}}}
      - {dataset: {metric: value, model: name}}
      - flat list of records
    Returns list of (dataset, model, metric, value) tuples.
    """
    records = []

    if isinstance(data, dict):
        # Format 1: top-level keys are dataset names
        for k, v in data.items():
            k_lower = k.lower()
            if k_lower in TDC_DATASETS | MOLECULENET_DATASETS:
                if isinstance(v, dict):
                    for model_or_metric, val in v.items():
                        if isinstance(val, dict):
                            # {dataset: {model: {metric: value}}}
                            for metric, score in val.items():
                                records.append((k_lower, model_or_metric.lower(), metric, score))
                        elif isinstance(val, (int, float)):
                            # {dataset: {metric: value}} — model unknown
                            records.append((k_lower, "unknown", model_or_metric, val))

        # Format 2: top-level keys are model names
        for k, v in data.items():
            k_lower = k.lower()
            if any(m in k_lower for m in MODELS_OF_INTEREST):
                if isinstance(v, dict):
                    for dataset, scores in v.items():
                        dataset_lower = dataset.lower()
                        if dataset_lower in TDC_DATASETS | MOLECULENET_DATASETS:
                            if isinstance(scores, dict):
                                for metric, val in scores.items():
                                    records.append((dataset_lower, k_lower, metric, val))
                            elif isinstance(scores, (int, float)):
                                records.append((dataset_lower, k_lower, "score", scores))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                dataset = str(item.get("dataset", item.get("task", ""))).lower()
                model   = str(item.get("model", item.get("backbone", "unknown"))).lower()
                metric  = item.get("metric", "score")
                value   = item.get("value", item.get("score", item.get("result", None)))
                if dataset and value is not None:
                    records.append((dataset, model, metric, value))

    return records

def is_plain_gcn(model_name: str) -> bool:
    name = model_name.lower()
    return ("gcn" in name) and ("moe" not in name) and ("plain" in name or name == "gcn" or "baseline" in name)

def is_moe_gcn(model_name: str) -> bool:
    name = model_name.lower()
    return "gcn" in name and "moe" in name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=".", help="Root directory to scan for JSON files")
    args = parser.parse_args()

    root = Path(args.results_dir)
    print(f"\n{'='*60}")
    print(f"  Scanning: {root.resolve()}")
    print(f"{'='*60}\n")

    json_files = find_json_files(root)
    if not json_files:
        print("❌  No JSON files found. Are you running this from your project root?")
        return

    print(f"Found {len(json_files)} JSON file(s):\n")
    for f in json_files:
        print(f"  📄 {f.relative_to(root)}")
    print()

    # ── Parse all records ────────────────────────────────────────────────────
    all_records = []  # (dataset, model, metric, value, filepath)
    parse_errors = []

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            records = extract_info(data, jf)
            for r in records:
                all_records.append((*r, str(jf.relative_to(root))))
        except Exception as e:
            parse_errors.append((jf, str(e)))

    if parse_errors:
        print("⚠️  Parse errors:")
        for jf, err in parse_errors:
            print(f"   {jf.relative_to(root)}: {err}")
        print()

    if not all_records:
        print("⚠️  Could not parse any structured records from the JSON files.")
        print("    Falling back to raw key inspection...\n")
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                print(f"  📄 {jf.relative_to(root)}")
                if isinstance(data, dict):
                    print(f"     Top-level keys ({len(data)}): {list(data.keys())[:15]}")
                elif isinstance(data, list):
                    print(f"     List of {len(data)} items. First item keys: {list(data[0].keys()) if data else 'empty'}")
                print()
            except Exception as e:
                print(f"     Error: {e}\n")
        return

    # ── Build coverage maps ──────────────────────────────────────────────────
    # dataset -> set of models with results
    dataset_models = defaultdict(set)
    for dataset, model, metric, value, filepath in all_records:
        dataset_models[dataset].add(model)

    tdc_covered    = {d for d in dataset_models if d in TDC_DATASETS}
    molnet_covered = {d for d in dataset_models if d in MOLECULENET_DATASETS}
    tdc_missing    = TDC_DATASETS - tdc_covered
    molnet_missing = MOLECULENET_DATASETS - molnet_covered

    # ── Plain GCN TDC check ──────────────────────────────────────────────────
    tdc_with_plain_gcn = set()
    tdc_with_moe_gcn   = set()
    for dataset, model, metric, value, filepath in all_records:
        if dataset in TDC_DATASETS:
            if is_plain_gcn(model):
                tdc_with_plain_gcn.add(dataset)
            if is_moe_gcn(model):
                tdc_with_moe_gcn.add(dataset)

    # ── Report ───────────────────────────────────────────────────────────────
    print("━"*60)
    print("  MOLECULENET COVERAGE")
    print("━"*60)
    for d in sorted(MOLECULENET_DATASETS):
        models = dataset_models.get(d, set())
        status = "✅" if models else "❌"
        print(f"  {status} {d:<20} models: {', '.join(sorted(models)) or 'none found'}")

    print()
    print("━"*60)
    print("  TDC COVERAGE")
    print("━"*60)
    for d in sorted(TDC_DATASETS):
        models = dataset_models.get(d, set())
        has_plain = any(is_plain_gcn(m) for m in models)
        has_moe   = any(is_moe_gcn(m) for m in models)
        plain_tag = "🟢 plain GCN" if has_plain else "🔴 NO plain GCN"
        moe_tag   = "MoE-GCN ✓" if has_moe else "MoE-GCN ✗"
        print(f"  {'✅' if models else '❌'} {d:<30} {plain_tag} | {moe_tag}")

    print()
    print("━"*60)
    print("  SUMMARY")
    print("━"*60)
    print(f"  MoleculeNet datasets found : {len(molnet_covered)}/10")
    print(f"  TDC datasets found         : {len(tdc_covered)}/22")
    print(f"  TDC with plain GCN baseline: {len(tdc_with_plain_gcn)}/22")
    print(f"  TDC with MoE-GCN           : {len(tdc_with_moe_gcn)}/22")

    if tdc_missing:
        print(f"\n  ❌ TDC datasets missing entirely:")
        for d in sorted(tdc_missing):
            print(f"     - {d}")

    needs_plain_gcn = TDC_DATASETS - tdc_with_plain_gcn
    if needs_plain_gcn:
        print(f"\n  🔴 TDC datasets needing plain GCN run ({len(needs_plain_gcn)}):")
        for d in sorted(needs_plain_gcn):
            print(f"     - {d}")
    else:
        print("\n  ✅ All TDC datasets have plain GCN baselines!")

    print()
    print("━"*60)
    print("  ALL MODELS DETECTED ACROSS ALL FILES")
    print("━"*60)
    all_models = {m for _, m, _, _, _ in all_records}
    for m in sorted(all_models):
        count = sum(1 for _, model, _, _, _ in all_records if model == m)
        tag = ""
        if is_plain_gcn(m): tag = "  ← plain GCN ✅"
        if is_moe_gcn(m):   tag = "  ← MoE-GCN ✅"
        print(f"  {m:<30} ({count} records){tag}")

    print()

if __name__ == "__main__":
    main()
