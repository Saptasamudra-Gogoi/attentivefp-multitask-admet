"""
scan_project_folder.py

Scans D:\\molprop_project for files relevant to the MoE-ADMET review tasks:
  - saved embeddings (for the 2.4 k-means null control)
  - run logs / training logs (for 2.1, 2.2, 2.8 forensics)
  - checkpoints (to check which model version produced which number)
  - result / metrics files (csv, json) that might hold Table 1/5/6 source data

Usage (Anaconda Prompt, env: moe_admet):
    conda activate moe_admet
    python scan_project_folder.py

Edit ROOT_DIR below if your project root differs.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# ---------------- CONFIG ----------------
ROOT_DIR = r"D:\molprop_project"

# Extensions we care about, grouped by purpose
CATEGORIES = {
    "embeddings": [".npy", ".npz", ".pt", ".pth", ".pkl", ".h5"],
    "logs": [".log", ".txt", ".out"],
    "checkpoints": [".ckpt", ".pth", ".pt"],
    "results_tables": [".csv", ".json", ".tsv", ".xlsx"],
}

# Keywords in filename/path that raise relevance (case-insensitive)
KEYWORDS = [
    "embed", "pool", "cluster", "kmeans", "k-means",
    "moe", "gcn", "dmpnn", "attentivefp", "gcnconv", "nnconv",
    "table", "results", "metrics", "eta2", "anova", "kruskal",
    "esol", "freesolv", "lipophilicity", "caco", "caco2", "caco-2",
    "ld50", "tdc", "moleculenet",
    "ablation", "dense", "sparse", "router", "routing",
    "seed", "run", "log", "checkpoint", "expert",
]

# Skip huge irrelevant folders to keep the scan fast
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "venv", "env"}

OUTPUT_JSON = "project_scan_results.json"
OUTPUT_CSV = "project_scan_results.csv"
# -----------------------------------------


def categorize(ext):
    cats = []
    for cat, exts in CATEGORIES.items():
        if ext.lower() in exts:
            cats.append(cat)
    return cats


def is_relevant(filename_lower):
    return any(kw in filename_lower for kw in KEYWORDS)


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def scan(root_dir):
    root = Path(root_dir)
    if not root.exists():
        print(f"ERROR: {root_dir} does not exist. Edit ROOT_DIR at the top of this script.")
        sys.exit(1)

    records = []
    total_files = 0
    skipped_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # prune skip dirs in-place so os.walk doesn't descend into them
        before = len(dirnames)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        skipped_dirs += before - len(dirnames)

        for fname in filenames:
            total_files += 1
            fpath = Path(dirpath) / fname
            ext = fpath.suffix
            cats = categorize(ext)
            fname_lower = fname.lower()
            path_lower = str(fpath).lower()
            relevant = is_relevant(fname_lower) or is_relevant(path_lower)

            if not cats and not relevant:
                continue  # skip totally irrelevant files silently

            try:
                stat = fpath.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                size = -1
                mtime = "N/A"

            records.append({
                "path": str(fpath),
                "filename": fname,
                "extension": ext,
                "categories": cats,
                "keyword_match": relevant,
                "size_bytes": size,
                "size_human": human_size(size) if size >= 0 else "N/A",
                "modified": mtime,
            })

    return records, total_files, skipped_dirs


def print_summary(records, total_files, skipped_dirs):
    print("=" * 70)
    print(f"SCAN COMPLETE — {ROOT_DIR}")
    print("=" * 70)
    print(f"Total files walked : {total_files}")
    print(f"Flagged as relevant: {len(records)}")
    print()

    # Group by category for a quick summary
    by_cat = {}
    for r in records:
        if not r["categories"]:
            by_cat.setdefault("keyword_only", []).append(r)
        for c in r["categories"]:
            by_cat.setdefault(c, []).append(r)

    for cat, items in sorted(by_cat.items()):
        print(f"--- {cat} ({len(items)} files) ---")
        # show newest 15 first, likely most relevant to recent runs
        items_sorted = sorted(items, key=lambda r: r["modified"], reverse=True)
        for r in items_sorted[:15]:
            print(f"  [{r['modified']}] {r['size_human']:>8}  {r['path']}")
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more (see {OUTPUT_CSV})")
        print()

    if skipped_dirs:
        print(f"(Skipped {skipped_dirs} directories like .git/__pycache__/venv)")
        print()


def save_outputs(records):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "filename", "extension", "categories",
            "keyword_match", "size_bytes", "size_human", "modified"
        ])
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["categories"] = ";".join(row["categories"])
            writer.writerow(row)

    print(f"Full results saved to:\n  {os.path.abspath(OUTPUT_JSON)}\n  {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    records, total_files, skipped_dirs = scan(ROOT_DIR)
    print_summary(records, total_files, skipped_dirs)
    save_outputs(records)
