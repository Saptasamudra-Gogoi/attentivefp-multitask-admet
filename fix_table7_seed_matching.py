"""
Fix Table 7's seed-count mismatch (GCN n=3, MoE-GCN n=5): recompute MoE-GCN
using only its first 3 seeds, so both models are compared at matched n=3.
No new training needed -- results_moegcn_tdc_v2.json already stores all 5
individual seed values per dataset.

Run from D:\\molprop_project\\ with moe_admet activated.
"""
import json
import numpy as np

with open("results_moegcn_tdc_v2.json") as f:
    moe = json.load(f)
with open("results_gcn_tdc.json") as f:
    gcn = json.load(f)

print(f"{'Dataset':<35} {'GCN (n=3)':<20} {'MoE-GCN (n=3, matched)':<25} {'MoE-GCN (n=5, full)'}")
print("-" * 105)

rows = []
for ds, entry in moe.items():
    seeds_full = entry.get("seeds", [])
    if len(seeds_full) < 3:
        print(f"[skip] {ds}: fewer than 3 seeds available")
        continue
    seeds_3 = seeds_full[:3]
    mean_3, std_3 = np.mean(seeds_3), np.std(seeds_3, ddof=1)
    mean_5, std_5 = entry["mean"], entry["std"]

    gcn_entry = gcn.get(ds, {})
    gcn_mean, gcn_std = gcn_entry.get("mean", "?"), gcn_entry.get("std", "?")

    print(f"{ds:<35} {gcn_mean:.4f} ± {gcn_std:.4f}   {mean_3:.4f} ± {std_3:.4f}          {mean_5:.4f} ± {std_5:.4f}")
    rows.append({
        "dataset": ds, "gcn_mean": gcn_mean, "gcn_std": gcn_std,
        "moe_mean_n3": mean_3, "moe_std_n3": std_3,
        "moe_mean_n5": mean_5, "moe_std_n5": std_5,
    })

with open("table7_matched_seeds.json", "w") as f:
    json.dump(rows, f, indent=2)

print("\nSaved: table7_matched_seeds.json")
print("Use the 'MoE-GCN (n=3, matched)' column to replace Table 7's current n=5 column")
print("for an apples-to-apples comparison with GCN's n=3 -- OR run 2 more GCN seeds")
print("for the more rigorous n=5 vs n=5 comparison, per Li's preference if time permits.")
