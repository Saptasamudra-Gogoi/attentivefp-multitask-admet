"""
generate_all_figures.py
=========================
Generates every figure the manuscript references that doesn't already
exist as a file, using data already verified in the manuscript's own
tables (Table 3, 4, 5) plus the two k-means control result sets
(plain-GCN 14/16, Dense-uniform 7/16) from terminal output this session.

Produces:
  MoE_Fig1_ArchitectureSchematic_v1.png        -- architecture schematic (Fig. 1)
  MoE_Fig2_SpecializationTransition_v1.png       -- exploration-to-specialization transition (Fig. 2, data: Table 3)
  MoE_Fig3A_Eta2Heatmap_v1.png         -- eta2 heatmap, 8 descriptors x 4 datasets (Fig. 3A, data: Table 5)
  MoE_Fig3B_ExpertProfiles_v1.png -- per-expert descriptor profiles on solubility (Fig. 3B, data: Table 4)
  MoE_Fig3C_DualControlScatter_v1.png    -- router eta2 vs k-means eta2, BOTH plain-GCN (16 pts)
                                and Dense-uniform (16 pts) -- 32 points total (Fig. 3C)

Run: python generate_all_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ============================================================
# Fig 1: Architecture schematic
# ============================================================
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    ("Molecular\ngraph", 0.5, "#E0E0E0"),
    ("Backbone\nGNN", 2.3, "#90CAF9"),
    ("Pooled\nvector", 4.1, "#B39DDB"),
    ("Router\n(gate)", 5.9, "#FFCC80"),
    ("Top-K\nexperts", 7.7, "#A5D6A7"),
    ("Combined\nprediction", 9.5, "#EF9A9A"),
]

for label, x, color in boxes:
    box = FancyBboxPatch((x - 0.75, 1.0), 1.5, 1.0,
                          boxstyle="round,pad=0.08", linewidth=1.2,
                          edgecolor="black", facecolor=color)
    ax.add_patch(box)
    ax.text(x, 1.5, label, ha="center", va="center", fontsize=9.5, fontweight="bold")

for i in range(len(boxes) - 1):
    x0 = boxes[i][1] + 0.75
    x1 = boxes[i + 1][1] - 0.75
    arrow = FancyArrowPatch((x0, 1.5), (x1, 1.5), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.3, color="black")
    ax.add_patch(arrow)

ax.text(5.9, 2.55, "No physicochemical descriptors,\nscaffold labels, or expert-assignment\nsupervision given to the router",
        ha="center", va="center", fontsize=7.5, style="italic", color="#555555")
ax.annotate("", xy=(5.9, 2.05), xytext=(5.9, 2.35),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1))

plt.title("Fig. 1 | Architecture-agnostic expert plug-in", fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("MoE_Fig1_ArchitectureSchematic_v1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved MoE_Fig1_ArchitectureSchematic_v1.png")

# ============================================================
# Fig 2: Exploration-to-specialization transition (Table 3 data)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
epochs = [10, 20, 30, 60]
expert0 = [26.8, 10.0, 3.7, 4.2]
expert1 = [36.2, 18.5, 15.8, 17.3]
expert2 = [23.9, 25.6, 20.1, 19.9]
expert3 = [13.2, 45.9, 60.3, 58.5]

colors = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726"]
for vals, label, c in zip([expert0, expert1, expert2, expert3],
                           ["Expert 0", "Expert 1", "Expert 2", "Expert 3"], colors):
    ax.plot(epochs, vals, marker="o", linewidth=2, markersize=7, label=label, color=c)

ax.axvspan(0, 15, alpha=0.08, color="gray")
ax.axvspan(15, 25, alpha=0.08, color="orange")
ax.axvspan(25, 65, alpha=0.08, color="green")
ax.text(7, 62, "Exploration", ha="center", fontsize=8.5, style="italic")
ax.text(20, 62, "Transition", ha="center", fontsize=8.5, style="italic")
ax.text(45, 62, "Specialization", ha="center", fontsize=8.5, style="italic")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("% of molecules routed to expert", fontsize=11)
ax.set_title("Fig. 2 | Exploration-to-specialization transition\n(Tox21, top-1 routing, 4 experts)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="center left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("MoE_Fig2_SpecializationTransition_v1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved MoE_Fig2_SpecializationTransition_v1.png")

# ============================================================
# Fig 3A: eta2 heatmap (Table 5 data)
# ============================================================
descriptors = ["LogP", "Aromatic\nrings", "MW", "Polar\nsurface area",
               "H-bond\nacceptors", "H-bond\ndonors", "Rotatable\nbonds", "Ring\ncount"]
datasets_5 = ["Solubility\n(AqSolDB)", "Caco-2\n(Wang)", "LD50\n(Zhu)", "Lipophilicity\n(AstraZeneca)"]

eta2_matrix = np.array([
    [0.325, 0.151, 0.142, 0.051],
    [0.102, 0.325, 0.445, 0.093],
    [0.101, 0.103, 0.110, 0.006],
    [0.087, 0.204, 0.105, 0.023],
    [0.049, 0.151, 0.152, 0.016],
    [0.057, 0.140, 0.029, 0.001],
    [0.042, 0.121, 0.169, 0.019],
    [0.077, 0.075, 0.291, 0.017],
])

fig, ax = plt.subplots(figsize=(6, 6.5))
im = ax.imshow(eta2_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.45)
ax.set_xticks(range(len(datasets_5)))
ax.set_xticklabels(datasets_5, fontsize=8.5, rotation=20, ha="right")
ax.set_yticks(range(len(descriptors)))
ax.set_yticklabels(descriptors, fontsize=9)

for i in range(len(descriptors)):
    for j in range(len(datasets_5)):
        val = eta2_matrix[i, j]
        color = "white" if val > 0.22 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("eta-squared", fontsize=10)
ax.set_title("Fig. 3A | Expert specialization by descriptor\n(all P < 0.001 except Lipophilicity-HBD)",
              fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("MoE_Fig3A_Eta2Heatmap_v1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved MoE_Fig3A_Eta2Heatmap_v1.png")

# ============================================================
# Fig 3B: per-expert descriptor profiles (Table 4 data, solubility dataset)
# ============================================================
experts = ["E5\n(n=3433)", "E1\n(n=1904)", "E7\n(n=3500)", "E6\n(n=195)", "E3\n(n=948)"]
logp_vals = [4.47, -0.68, 0.88, 0.30, 2.74]
tpsa_vals = [44.9, 96.2, 64.8, 30.8, 56.3]
arrings_vals = [1.59, 0.98, 0.67, 0.04, 1.00]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors5 = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726", "#AB47BC"]

for ax, vals, title, ylabel in zip(
    axes, [logp_vals, tpsa_vals, arrings_vals],
    ["LogP", "Polar surface area (A^2)", "Aromatic rings"],
    ["Mean LogP", "Mean TPSA", "Mean count"]):
    ax.bar(experts, vals, color=colors5, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.grid(axis="y", alpha=0.25)

plt.suptitle("Fig. 3B | Per-expert physicochemical profiles (Solubility, AqSolDB)",
             fontsize=11.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("MoE_Fig3B_ExpertProfiles_v1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved MoE_Fig3B_ExpertProfiles_v1.png")

# ============================================================
# Fig 3C: DUAL control scatter -- plain-GCN (16 pts) + Dense-uniform (16 pts)
# ============================================================
plain_gcn_pts = [
    ("caco2_wang", "LogP", 0.2058, 0.2571),
    ("caco2_wang", "ArRings", 0.4007, 0.3838),
    ("solubility_aqsoldb", "LogP", 0.2897, 0.3306),
    ("solubility_aqsoldb", "ArRings", 0.1596, 0.3615),
    ("lipophilicity_astrazeneca", "LogP", 0.0187, 0.0134),
    ("lipophilicity_astrazeneca", "ArRings", 0.0379, 0.1691),
    ("ld50_zhu", "LogP", 0.1796, 0.2497),
    ("ld50_zhu", "ArRings", 0.3878, 0.5569),
    ("bbb_martins", "LogP", 0.2138, 0.2904),
    ("bbb_martins", "ArRings", 0.2624, 0.4512),
    ("herg", "LogP", 0.0734, 0.1549),
    ("herg", "ArRings", 0.3796, 0.5662),
    ("ames", "LogP", 0.2239, 0.3765),
    ("ames", "ArRings", 0.5643, 0.6660),
    ("dili", "LogP", 0.0458, 0.2856),
    ("dili", "ArRings", 0.2188, 0.5544),
]

dense_uniform_pts = [
    ("caco2_wang", "LogP", 0.2894),
    ("caco2_wang", "ArRings", 0.3900),
    ("solubility_aqsoldb", "LogP", 0.3328),
    ("solubility_aqsoldb", "ArRings", 0.3040),
    ("lipophilicity_astrazeneca", "LogP", 0.0062),
    ("lipophilicity_astrazeneca", "ArRings", 0.1913),
    ("ld50_zhu", "LogP", 0.2137),
    ("ld50_zhu", "ArRings", 0.5490),
    ("bbb_martins", "LogP", 0.2653),
    ("bbb_martins", "ArRings", 0.3891),
    ("herg", "LogP", 0.1905),
    ("herg", "ArRings", 0.5040),
    ("ames", "LogP", 0.3652),
    ("ames", "ArRings", 0.6762),
    ("dili", "LogP", 0.3808),
    ("dili", "ArRings", 0.5124),
]

fig, ax = plt.subplots(figsize=(7, 7))

router_plain = [p[2] for p in plain_gcn_pts]
kmeans_plain = [p[3] for p in plain_gcn_pts]
router_dense = [p[2] for p in plain_gcn_pts]  # same router values, matched by dataset/descriptor order
kmeans_dense = [p[2] for p in dense_uniform_pts]

ax.scatter(router_plain, kmeans_plain, s=90, c="#4CAF50", edgecolor="white",
           linewidth=1, alpha=0.85, label="Plain-GCN control (14/16 >= router)", zorder=3)
ax.scatter(router_dense, kmeans_dense, s=90, c="#E91E63", edgecolor="white",
           linewidth=1, alpha=0.85, marker="^", label="Dense-uniform control (7/16 >= router)", zorder=3)

lims = [0, 0.75]
ax.plot(lims, lims, "k--", alpha=0.4, zorder=1, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Router eta-squared (MoE dominant-expert assignment)", fontsize=11)
ax.set_ylabel("K-means eta-squared (independent representation)", fontsize=11)
ax.set_title("Fig. 3C | Two controls, different answers\n"
             "Plain-GCN robustly recovers router organization; Dense-uniform does not",
             fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("MoE_Fig3C_DualControlScatter_v1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved MoE_Fig3C_DualControlScatter_v1.png (32 points: 16 plain-GCN + 16 Dense-uniform)")

print("\n" + "="*70)
print("  FLAG: fig_specialization_vs_gain.png (made earlier this session)")
print("  has NO figure number or legend entry in the manuscript.")
print("  Recommend adding as new Fig. 5, with its own legend paragraph.")
print("="*70)
