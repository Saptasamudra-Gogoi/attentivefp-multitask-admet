"""
make_merged_performance_figure.py
====================================
P0-8 completion. Merges Figs 4-6 into a single 3-panel performance
figure, and fixes Fig 5's caption disclaimer by making the pooled
sign-test scope explicit in the panel itself rather than apologizing
for a scope mismatch in prose.

Panel A: MoleculeNet regression, GCN vs MoE-GCN, with real error bars
         from the matched-seed rerun (run_gcn_matched.py / P0-2).
Panel B: EC-MPNN transfer (plain vs +MoE) on the same 3 datasets.
Panel C: Parameter-matched ablation (MoE-GCN / Dense-uniform /
         Dense-wide / Plain-GCN) from the corrected ablation_routing
         protocol (P0-5b), 5 datasets.

Panel A's title states plainly that the annotated pooled sign test
(P=0.019) covers all 12 regression datasets, of which these 3 are
shown -- stated once, in the panel, not apologized for in the caption.

Run: python make_merged_performance_figure.py
Output: fig_merged_performance.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.35)

# ============================================================
# Panel A: MoleculeNet regression, GCN vs MoE-GCN (matched protocol)
# ============================================================
ax_a = fig.add_subplot(gs[0])

datasets_a = ["ESOL", "FreeSolv", "Lipophilicity"]
# GCN: matched-HPO 5-seed values from run_gcn_matched.py
gcn_mean = [1.1200, 3.6538, 0.7446]
gcn_std  = [0.0363, 0.5441, 0.0066]
# MoE-GCN: 5-seed values from results_moegcn_regr.json
moe_mean = [1.0665, 3.5906, 0.7219]
moe_std  = [0.0329, 0.3793, 0.0061]

x = np.arange(len(datasets_a))
w = 0.35
ax_a.bar(x - w/2, gcn_mean, w, yerr=gcn_std, label="GCN", color="#9E9E9E",
          capsize=4, edgecolor="white", linewidth=0.5)
ax_a.bar(x + w/2, moe_mean, w, yerr=moe_std, label="MoE-GCN", color="#2196F3",
          capsize=4, edgecolor="white", linewidth=0.5)
ax_a.set_xticks(x)
ax_a.set_xticklabels(datasets_a, rotation=15, ha="right", fontsize=9)
ax_a.set_ylabel("RMSE (lower is better)", fontsize=10)
ax_a.set_title("A. MoleculeNet regression\n"
               "(3 of 12 datasets in the pooled sign test, P=0.019)",
               fontsize=9.5, fontweight="bold")
ax_a.legend(fontsize=8)
ax_a.grid(axis="y", alpha=0.25)

# ============================================================
# Panel B: EC-MPNN transfer
# ============================================================
ax_b = fig.add_subplot(gs[1])

datasets_b = ["ESOL", "FreeSolv", "Lipophilicity"]
ecmpnn_plain = [1.163, 3.089, 0.701]
ecmpnn_moe   = [1.105, 2.865, 0.712]

x = np.arange(len(datasets_b))
colors_b = ["#4CAF50" if m < p else "#F44336" for p, m in zip(ecmpnn_plain, ecmpnn_moe)]
ax_b.bar(x - w/2, ecmpnn_plain, w, label="EC-MPNN", color="#9E9E9E",
          edgecolor="white", linewidth=0.5)
ax_b.bar(x + w/2, ecmpnn_moe, w, label="EC-MPNN + MoE", color="#FF9800",
          edgecolor="white", linewidth=0.5)
ax_b.set_xticks(x)
ax_b.set_xticklabels(datasets_b, rotation=15, ha="right", fontsize=9)
ax_b.set_ylabel("RMSE (lower is better)", fontsize=10)
ax_b.set_title("B. Cross-architecture transfer\n(EC-MPNN backbone)",
               fontsize=9.5, fontweight="bold")
ax_b.legend(fontsize=8)
ax_b.grid(axis="y", alpha=0.25)

# ============================================================
# Panel C: Parameter-matched ablation, 4 configurations, 5 datasets
# ============================================================
ax_c = fig.add_subplot(gs[2])

datasets_c = ["ESOL", "FreeSolv", "Lipo", "Caco-2", "Solub."]
moe_c    = [1.118, 3.067, 0.783, 0.540, 1.136]
dense_u  = [1.123, 3.043, 0.764, 0.547, 1.137]
dense_w  = [1.108, 3.087, 0.809, 0.525, 1.112]
plain_c  = [1.102, 3.119, 0.752, 0.557, 1.267]

x = np.arange(len(datasets_c))
w4 = 0.2
ax_c.bar(x - 1.5*w4, moe_c,   w4, label="MoE-GCN",      color="#2196F3", edgecolor="white", linewidth=0.4)
ax_c.bar(x - 0.5*w4, dense_u, w4, label="Dense-uniform", color="#FF9800", edgecolor="white", linewidth=0.4)
ax_c.bar(x + 0.5*w4, dense_w, w4, label="Dense-wide",    color="#9C27B0", edgecolor="white", linewidth=0.4)
ax_c.bar(x + 1.5*w4, plain_c, w4, label="Plain-GCN",     color="#4CAF50", edgecolor="white", linewidth=0.4)
ax_c.set_xticks(x)
ax_c.set_xticklabels(datasets_c, rotation=15, ha="right", fontsize=9)
ax_c.set_ylabel("RMSE (lower is better)", fontsize=10)
ax_c.set_title("C. Parameter-matched ablation\n(no configuration dominates)",
               fontsize=9.5, fontweight="bold")
ax_c.legend(fontsize=7.5, ncol=2)
ax_c.grid(axis="y", alpha=0.25)

plt.savefig("fig_merged_performance.png", dpi=300, bbox_inches="tight")
print("Saved fig_merged_performance.png (3 panels: MoleculeNet regression, "
      "EC-MPNN transfer, parameter-matched ablation)")
print("This replaces the previous separate Figs 4, 5, and 6.")
