"""
filter_scan.py

Run after scan_project_folder.py. Searches project_scan_results.csv for files
relevant to specific review tasks and prints them grouped, with full paths
and sizes so we can spot pooled embeddings vs routing outputs, find the
DMPNN FreeSolv source, etc.

Usage:
    conda activate moe_admet
    cd D:\\molprop_project
    python filter_scan.py
"""

import csv
import sys
from pathlib import Path

CSV_PATH = "project_scan_results.csv"

# Each group: label -> list of substrings (any match on lowercased path triggers it)
GROUPS = {
    "POOLED EMBEDDINGS (for 2.4 k-means control)": [
        "pooled", "graph_embed", "mol_embed", "representation", "features_",
        "embed_" , "_embed.", "hidden_repr", "readout",
    ],
    "ROUTING outputs (NOT the same as pooled embeddings)": [
        "routing_", "gate_weight", "router",
    ],
    "DMPNN specific (for 2.8 FreeSolv 3.432 check)": [
        "dmpnn",
    ],
    "FreeSolv specific (for 2.2 / 2.8)": [
        "freesolv",
    ],
    "Caco-2 specific (for 2.2)": [
        "caco",
    ],
    "Table 6 / dense ablation (for 2.1)": [
        "dense_ablation", "param_matched", "parameter_matched", "ablation_routing",
        "ablation_optuna", "random_ablation", "cross_arch",
    ],
    "Table 1 baseline results (for 2.1 / 2.7)": [
        "baseline_g", "final_results_table", "grover_results",
    ],
    "TDC 9/22-dataset results (for 2.6)": [
        "9datasets", "9_datasets", "22datasets", "22_datasets", "tdc",
    ],
    "Descriptor / eta2 / ANOVA (for 2.3, 2.5)": [
        "descriptor", "eta2", "eta_2", "anova", "kruskal", "expert_specialization",
    ],
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    if not Path(CSV_PATH).exists():
        print(f"ERROR: {CSV_PATH} not found in current directory. Run scan_project_folder.py first.")
        sys.exit(1)

    rows = load_rows(CSV_PATH)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}\n")

    matched_any = set()

    for label, keywords in GROUPS.items():
        matches = []
        for r in rows:
            path_lower = r["path"].lower()
            if any(kw in path_lower for kw in keywords):
                matches.append(r)
                matched_any.add(r["path"])

        print("=" * 70)
        print(f"{label}  ({len(matches)} files)")
        print("=" * 70)
        if not matches:
            print("  ** NONE FOUND ** — this data may not exist yet, or is named differently.")
        else:
            # sort newest first
            matches_sorted = sorted(matches, key=lambda r: r["modified"], reverse=True)
            for r in matches_sorted[:25]:
                print(f"  [{r['modified']}] {r['size_human']:>8}  {r['path']}")
            if len(matches) > 25:
                print(f"  ... and {len(matches) - 25} more")
        print()

    print("=" * 70)
    print(f"Total distinct files matched across all groups: {len(matched_any)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
