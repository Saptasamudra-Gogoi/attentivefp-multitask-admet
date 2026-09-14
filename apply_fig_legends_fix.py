import shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_fig_legends_fix.bak")
content = open(path, encoding="utf-8").read()

old = ("- **Fig. 4 | EC-MPNN cross-architecture transferability: a partial "
       "result.** RMSE for the edge-conditioned MPNN backbone (plain) "
       "versus EC-MPNN + MoE on ESOL, FreeSolv, and Lipophilicity; MoE "
       "improves ESOL and FreeSolv but not Lipophilicity.\n"
       "- **Fig. 5 | Regression performance gain over plain GCN.** "
       "Per-dataset percentage RMSE improvement, MoE-GCN versus GCN, "
       "shown for the three MoleculeNet regression datasets (ESOL, "
       "FreeSolv, Lipophilicity); the annotated pooled significance test "
       "(exact sign test, *P* = 0.019) covers all 12 regression datasets "
       "(MoleculeNet + TDC), not only the three bars shown here (data: "
       "Tables 1\u20132, 7).\n"
       "- **Fig. 6 | Parameter-matched ablation.** RMSE for MoE-GCN, "
       "Dense-uniform, and Dense-wide across five regression datasets, "
       "with pooled significance tests annotated (data: Table 6).")

new = ("- **Fig. 4 | Merged performance summary (replaces the previous "
       "separate Figs. 4\u20136).** Three panels sharing one figure: (A) "
       "MoleculeNet regression, GCN versus MoE-GCN under the matched "
       "seed/HPO protocol (data: Table 1); the panel states directly that "
       "these three datasets are 3 of the 12 datasets covered by the "
       "pooled sign test (*P* = 0.019, Table 2), so the scope of the "
       "annotated statistic is stated once, in the panel itself, rather "
       "than requiring a caption disclaimer about a mismatch between what "
       "is drawn and what is tested. (B) Cross-architecture transfer, EC-"
       "MPNN plain versus EC-MPNN + MoE on the same three datasets; MoE "
       "improves ESOL and FreeSolv but not Lipophilicity (data: Methods). "
       "(C) Parameter-matched ablation \u2014 MoE-GCN, Dense-uniform, "
       "Dense-wide, and Plain-GCN, all four under the identical fixed "
       "architecture and training protocol \u2014 across five regression "
       "datasets; no configuration dominates, and Plain-GCN wins outright "
       "on two of five (data: Table 6).")

c = content.count(old)
print(f"Found {c} occurrence(s)")
assert c == 1, f"Expected 1, found {c}"
content = content.replace(old, new)

# Also update Fig 3's legend to note the three-way (not two-way) k-means comparison
old2 = ("- **Fig. 3 | Quantified expert specialization across four ADMET "
        "datasets.** (A) \u03b7\u00b2 heatmap\u2014the proportion of "
        "descriptor variance explained by expert assignment\u2014across "
        "eight descriptors and four datasets (data: Table 5). (B) "
        "Per-expert descriptor profiles on the solubility dataset (data: "
        "Table 4).")
new2 = ("- **Fig. 3 | Quantified expert specialization across four ADMET "
        "datasets, and its representation-level controls.** (A) "
        "\u03b7\u00b2 heatmap\u2014the proportion of descriptor variance "
        "explained by expert assignment\u2014across eight descriptors and "
        "four datasets (data: Table 5). (B) Per-expert descriptor profiles "
        "on the solubility dataset (data: Table 4). (C) Paired scatter, "
        "router \u03b7\u00b2 against k-means \u03b7\u00b2 on the same "
        "representation, for both the plain-GCN control (14 of 16 points "
        "at or above the identity line) and the Dense-uniform control (7 "
        "of 16), LogP and aromatic-ring count, 8 datasets (data: "
        "`kmeans_control_plain_gcn.json`, `kmeans_control_dense_uniform.json`, "
        "code repository; rendered in `fig_kmeans_vs_router.png`).")

c2 = content.count(old2)
print(f"Found {c2} occurrence(s) for Fig 3")
assert c2 == 1, f"Expected 1, found {c2}"
content = content.replace(old2, new2)

open(path, "w", encoding="utf-8").write(content)
print("Both applied")
