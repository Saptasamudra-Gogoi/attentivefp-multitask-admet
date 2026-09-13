"""
Fig 1 | MoE-GCN vs plain GCN performance overview
300 dpi, Arial, publication-ready
"""
import json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load data ──────────────────────────────────────────────────────────────
with open("results_moegcn_regr.json") as f:
    regr = json.load(f)
with open("results_moegcn_classif.json") as f:
    clf = json.load(f)
with open("results_gcn_tdc.json") as f:
    gcn_tdc = json.load(f)
with open("results_moegcn_tdc_v2.json") as f:
    moe_tdc = json.load(f)

# ── MoleculeNet regression (RMSE, lower=better) ────────────────────────────
mol_regr_datasets = ["ESOL", "FreeSolv", "Lipo"]
mol_regr_labels   = ["ESOL", "FreeSolv", "Lipo"]

# plain GCN baselines from paper (from baselines/gin_gcn_gat results)
# using GCN values from the json in baselines
import os, glob
gcn_regr_means = {"ESOL": 1.4023, "FreeSolv": 2.3770, "Lipo": 0.7717}
gcn_regr_stds  = {"ESOL": 0.0512, "FreeSolv": 0.1207, "Lipo": 0.0154}

gcn_clf_means = {"BBBP": 0.6839, "BACE": 0.8201, "HIV": 0.7667,
                 "Tox21": 0.7426, "SIDER": 0.5763, "ClinTox": 0.9147}
gcn_clf_stds  = {"BBBP": 0.0110, "BACE": 0.0143, "HIV": 0.0223,
                 "Tox21": 0.0063, "SIDER": 0.0092, "ClinTox": 0.0122}

# ── MoleculeNet classification (ROC-AUC, higher=better) ───────────────────
mol_clf_datasets = ["BBBP", "BACE", "HIV", "Tox21", "SIDER", "ClinTox"]
mol_clf_labels   = ["BBBP", "BACE", "HIV", "Tox21", "SIDER", "ClinTox"]

gcn_clf_means = {"BBBP": 0.690, "BACE": 0.764, "HIV": 0.763,
                 "Tox21": 0.774, "SIDER": 0.587, "ClinTox": 0.862}
gcn_clf_stds  = {"BBBP": 0.019, "BACE": 0.023, "HIV": 0.011,
                 "Tox21": 0.009, "SIDER": 0.012, "ClinTox": 0.048}

moe_clf_means = {k: clf[k]["mean"] for k in mol_clf_datasets}
moe_clf_stds  = {k: clf[k]["std"]  for k in mol_clf_datasets}

# ── TDC regression (MAE or Spearman, mixed) ───────────────────────────────
tdc_regr_datasets = ["caco2_wang", "lipophilicity_astrazeneca",
                     "solubility_aqsoldb", "ppbr_az",
                     "vdss_lombardo", "half_life_obach",
                     "clearance_microsome_az", "clearance_hepatocyte_az",
                     "ld50_zhu"]
tdc_regr_labels = ["Caco-2", "Lipo-AZ", "Solubility", "PPBR",
                   "VDss", "Half-Life", "CL-Micro", "CL-Hepato", "LD50"]

# filter to datasets present in both
tdc_regr_datasets = [d for d in tdc_regr_datasets
                     if d in gcn_tdc and d in moe_tdc]
tdc_regr_labels   = [tdc_regr_labels[i]
                     for i, d in enumerate(["caco2_wang","lipophilicity_astrazeneca",
                                            "solubility_aqsoldb","ppbr_az",
                                            "vdss_lombardo","half_life_obach",
                                            "clearance_microsome_az","clearance_hepatocyte_az",
                                            "ld50_zhu"])
                     if d in gcn_tdc and d in moe_tdc]

gcn_tdc_means = {d: gcn_tdc[d]["mean"] for d in tdc_regr_datasets}
gcn_tdc_stds  = {d: gcn_tdc[d]["std"]  for d in tdc_regr_datasets}
moe_tdc_means = {d: moe_tdc[d]["mean"] for d in tdc_regr_datasets}
moe_tdc_stds  = {d: moe_tdc[d].get("std", 0) for d in tdc_regr_datasets}

# ── TDC classification ─────────────────────────────────────────────────────
tdc_clf_datasets = ["hia_hou", "pgp_broccatelli", "bioavailability_ma",
                    "bbb_martins", "cyp2d6_veith", "cyp3a4_veith",
                    "cyp2c9_veith", "cyp2d6_substrate_carbonmangels",
                    "cyp3a4_substrate_carbonmangels",
                    "cyp2c9_substrate_carbonmangels", "herg", "ames", "dili"]
tdc_clf_labels   = ["HIA", "P-gp", "Bioavail.", "BBB",
                    "CYP2D6v", "CYP3A4v", "CYP2C9v",
                    "CYP2D6s", "CYP3A4s", "CYP2C9s",
                    "hERG", "AMES", "DILI"]

tdc_clf_datasets = [d for d in tdc_clf_datasets
                    if d in gcn_tdc and d in moe_tdc]
tdc_clf_labels   = [tdc_clf_labels[i]
                    for i, d in enumerate(["hia_hou","pgp_broccatelli",
                                           "bioavailability_ma","bbb_martins",
                                           "cyp2d6_veith","cyp3a4_veith",
                                           "cyp2c9_veith","cyp2d6_substrate_carbonmangels",
                                           "cyp3a4_substrate_carbonmangels",
                                           "cyp2c9_substrate_carbonmangels",
                                           "herg","ames","dili"])
                    if d in gcn_tdc and d in moe_tdc]

gcn_tdc_clf_means = {d: gcn_tdc[d]["mean"] for d in tdc_clf_datasets}
gcn_tdc_clf_stds  = {d: gcn_tdc[d]["std"]  for d in tdc_clf_datasets}
moe_tdc_clf_means = {d: moe_tdc[d]["mean"] for d in tdc_clf_datasets}
moe_tdc_clf_stds  = {d: moe_tdc[d].get("std", 0) for d in tdc_clf_datasets}

# ── colours ────────────────────────────────────────────────────────────────
C_GCN = "#6baed6"   # blue  = plain GCN
C_MOE = "#fd8d3c"   # orange = MoE-GCN
EW = 1.2            # error bar linewidth

def grouped_bars(ax, labels, gcn_m, gcn_s, moe_m, moe_s,
                 ylabel, lower_better=False, title=""):
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, gcn_m, w, yerr=gcn_s, color=C_GCN,
                capsize=3, ecolor="0.3", label="GCN",
                error_kw={"elinewidth": EW})
    b2 = ax.bar(x + w/2, moe_m, w, yerr=moe_s, color=C_MOE,
                capsize=3, ecolor="0.3", label="MoE-GCN",
                error_kw={"elinewidth": EW})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # shade wins
    for i, (g, m) in enumerate(zip(gcn_m, moe_m)):
        win = (m < g) if lower_better else (m > g)
        if win:
            ax.axvspan(i - 0.5, i + 0.5, color=C_MOE, alpha=0.08, zorder=0)

# ── figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

moe_regr_means = {k: regr[k]["mean"] for k in mol_regr_datasets}
moe_regr_stds  = {k: regr[k]["std"]  for k in mol_regr_datasets}

# Panel A – MoleculeNet regression
grouped_bars(ax1,
    mol_regr_labels,
    [gcn_regr_means[d] for d in mol_regr_datasets],
    [gcn_regr_stds[d]  for d in mol_regr_datasets],
    [moe_regr_means[d] for d in mol_regr_datasets],
    [moe_regr_stds[d]  for d in mol_regr_datasets],
    ylabel="RMSE (↓)", lower_better=True,
    title="A  MoleculeNet regression")

# Panel B – MoleculeNet classification
grouped_bars(ax2,
    mol_clf_labels,
    [gcn_clf_means[d] for d in mol_clf_datasets],
    [gcn_clf_stds[d]  for d in mol_clf_datasets],
    [moe_clf_means[d] for d in mol_clf_datasets],
    [moe_clf_stds[d]  for d in mol_clf_datasets],
    ylabel="ROC-AUC (↑)", lower_better=False,
    title="B  MoleculeNet classification")

# Panel C – TDC regression
grouped_bars(ax3,
    tdc_regr_labels,
    [gcn_tdc_means[d] for d in tdc_regr_datasets],
    [gcn_tdc_stds[d]  for d in tdc_regr_datasets],
    [moe_tdc_means[d] for d in tdc_regr_datasets],
    [moe_tdc_stds[d]  for d in tdc_regr_datasets],
    ylabel="MAE / Spearman (mixed)", lower_better=False,
    title="C  TDC ADMET regression")
ax3.set_ylim(0, 1.5)  # exclude PPBR outlier; add note
ax3.annotate("PPBR excluded\n(MAE~9.5)", xy=(3, 1.45), fontsize=6, ha='center', color='gray')

# Panel D – TDC classification
grouped_bars(ax4,
    tdc_clf_labels,
    [gcn_tdc_clf_means[d] for d in tdc_clf_datasets],
    [gcn_tdc_clf_stds[d]  for d in tdc_clf_datasets],
    [moe_tdc_clf_means[d] for d in tdc_clf_datasets],
    [moe_tdc_clf_stds[d]  for d in tdc_clf_datasets],
    ylabel="ROC-AUC (↑)", lower_better=False,
    title="D  TDC ADMET classification")

# ── shared legend ──────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_GCN, label="Plain GCN"),
    mpatches.Patch(color=C_MOE, label="MoE-GCN (ours)"),
]
fig.legend(handles=legend_handles, loc="upper center",
           ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle("Figure 1 | MoE-GCN vs plain GCN across MoleculeNet and TDC ADMET benchmarks",
             fontsize=10, y=1.04)

plt.savefig("fig1_final.png", dpi=300, bbox_inches="tight")
plt.savefig("fig1_final.pdf", bbox_inches="tight")
print("Saved: fig1_final.png / fig1_final.pdf")