"""
Fig 2 | Expert specialization: eta2 and MI across physicochemical descriptors
"""
import json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt

with open("expert_specialization_SUMMARY.json") as f:
    d = json.load(f)

per      = d["per_dataset"]
datasets  = [ds for ds in per if per[ds]["stats"]]
ds_labels = ["Solubility", "Caco-2", "Lipophilicity", "LD50"]
descriptors = list(per[datasets[0]]["stats"].keys())

eta2_matrix = np.array([[per[ds]["stats"][desc]["eta2"] for ds in datasets]
                         for desc in descriptors])
mi_matrix   = np.array([[per[ds]["stats"][desc]["MI"]   for ds in datasets]
                         for desc in descriptors])

mean_eta2 = eta2_matrix.mean(axis=1)
std_eta2  = eta2_matrix.std(axis=1)
order     = np.argsort(mean_eta2)[::-1]

fig = plt.figure(figsize=(14, 5))
gs  = fig.add_gridspec(1, 3, wspace=0.45)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

CMAP = "YlOrRd"

def heatmap(ax, matrix, row_labels, col_labels, title, vmax=None):
    vm = vmax or matrix.max()
    im = ax.imshow(matrix, aspect="auto", cmap=CMAP, vmin=0, vmax=vm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if matrix[i,j] < vm*0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

heatmap(ax1, eta2_matrix, descriptors, ds_labels,
        "A  η² (ANOVA effect size)", vmax=0.45)
heatmap(ax2, mi_matrix, descriptors, ds_labels,
        "B  Mutual information", vmax=0.45)

sorted_desc = [descriptors[i] for i in order]
sorted_eta2 = mean_eta2[order]
sorted_std  = std_eta2[order]
colors = ["#d73027" if desc == "LogP" else
          "#fdae61" if sorted_eta2[i] > 0.15 else
          "#74add1" for i, desc in enumerate(sorted_desc)]

ax3.barh(range(len(sorted_desc)), sorted_eta2, xerr=sorted_std,
         color=colors, capsize=3,
         error_kw={"elinewidth": 1.0, "ecolor": "0.3"})
ax3.set_yticks(range(len(sorted_desc)))
ax3.set_yticklabels(sorted_desc, fontsize=8)
ax3.set_xlabel("Mean η² across datasets", fontsize=8)
ax3.set_title("C  Descriptor importance ranking", fontsize=9,
              fontweight="bold", pad=4)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.axvline(0.10, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax3.text(0.101, -0.7, "η²=0.10", fontsize=6, color="gray")

fig.suptitle("Figure 2 | Physicochemical specialization of MoE routing",
             fontsize=10, y=1.02)

plt.savefig("fig2_final.png", dpi=300, bbox_inches="tight")
plt.savefig("fig2_final.pdf", bbox_inches="tight")
print("Saved: fig2_final.png / fig2_final.pdf")