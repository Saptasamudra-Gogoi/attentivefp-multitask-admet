"""
Fig 4 | Per-dataset gain + Wilcoxon pooled test
"""
import json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

with open("statistical_significance_results.json") as f:
    d = json.load(f)

per     = d["per_dataset"]
overall = d["overall"]

labels = [v["display"] for v in per.values()]
gains  = [v["gain_pct"] for v in per.values()]
sig05  = [v["significant_p05"] for v in per.values()]
sig01  = [v["significant_p01"] for v in per.values()]

order  = np.argsort(gains)[::-1]
labels = [labels[i] for i in order]
gains  = [gains[i]  for i in order]
sig05  = [sig05[i]  for i in order]
sig01  = [sig01[i]  for i in order]

colors = []
for g, s5, s1 in zip(gains, sig05, sig01):
    if s1:      colors.append("#d73027")
    elif s5:    colors.append("#fc8d59")
    elif g > 0: colors.append("#91bfdb")
    else:       colors.append("#e0e0e0")

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(range(len(labels)), gains, color=colors,
        edgecolor="0.3", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("MoE-GCN gain over plain GCN (%)", fontsize=9)
ax.set_title("Figure 4 | Per-dataset performance gain (MoE-GCN vs GCN)",
             fontsize=10, fontweight="bold", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=8)

for i, (g, s5, s1) in enumerate(zip(gains, sig05, sig01)):
    if s1:
        ax.text(g + 0.3, i, "**", va="center", fontsize=9, color="#d73027")
    elif s5:
        ax.text(g + 0.3, i, "*",  va="center", fontsize=9, color="#fc8d59")

ax.text(0.98, 0.96,
        f"Pooled Wilcoxon P = 0.025\n"
        f"Positive: {overall['positive_datasets']}/14 datasets\n"
        f"* p<0.05  ** p<0.01",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.8))

legend_elements = [
    Patch(facecolor="#d73027", label="p < 0.01"),
    Patch(facecolor="#fc8d59", label="p < 0.05"),
    Patch(facecolor="#91bfdb", label="gain, n.s."),
    Patch(facecolor="#e0e0e0", label="loss, n.s."),
]
ax.legend(handles=legend_elements, fontsize=7, frameon=False,
          loc="lower left", bbox_to_anchor=(0.01, 0.01))

plt.tight_layout()
plt.savefig("fig4_final.png", dpi=300, bbox_inches="tight")
plt.savefig("fig4_final.pdf", bbox_inches="tight")
print("Saved: fig4_final.png / fig4_final.pdf")