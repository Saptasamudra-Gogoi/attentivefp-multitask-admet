"""
Full per-dataset TDC table, MoE-GCN vs GCN (review item 2.6).
Run from D:\\molprop_project\\ with moe_admet activated.

Sources (per your project's file-provenance rules):
  results_tdc.json      -> MoE model results
  results_gcn_tdc.json  -> plain GCN baseline

Outputs a markdown table + CSV covering all 22 TDC datasets so the P=0.025
pooled statistic in Table 2 is traceable to per-dataset numbers.
"""
import json
import pandas as pd

MOE_FILE = "results_tdc.json"
GCN_FILE = "results_gcn_tdc.json"

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_rows(data, label):
    rows = []
    # adjust the traversal below to match your actual JSON schema if it differs
    for dataset, entry in data.items():
        if not isinstance(entry, dict):
            continue
        mean = entry.get("mean", entry.get("auc_mean", entry.get("rmse_mean")))
        std = entry.get("std", entry.get("auc_std", entry.get("rmse_std")))
        seeds = entry.get("seeds", entry.get("n_seeds"))
        n_seeds = len(seeds) if isinstance(seeds, list) else seeds
        rows.append({"dataset": dataset, "model": label, "mean": mean, "std": std, "n_seeds": n_seeds})
    return rows

def main():
    moe = load(MOE_FILE)
    gcn = load(GCN_FILE)

    rows = extract_rows(moe, "MoE-GCN") + extract_rows(gcn, "GCN")
    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows extracted -- check the JSON schema against extract_rows().")
        return

    pivot = df.pivot_table(index="dataset", columns="model", values=["mean", "std", "n_seeds"])
    pivot.to_csv("tdc_full_22_dataset_table.csv")
    print(pivot)
    print(f"\nDatasets found: {df['dataset'].nunique()} (expected 22 -- check for missing ones below)")

    found = set(df["dataset"].unique())
    print("Datasets present:", sorted(found))
    print("\nSaved: tdc_full_22_dataset_table.csv")
    print("Paste this into the paper as the missing per-dataset TDC table for 2.6.")

if __name__ == "__main__":
    main()
