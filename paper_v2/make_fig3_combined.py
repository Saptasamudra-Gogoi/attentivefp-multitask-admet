"""Combined Fig. 3 from saved results only. Run from the repo root:  python make_fig3_combined.py
A: eta^2 mean +- s.d. over 5 seeds   (paper_v2/results/eta2_across_seeds.json)
B: per-expert descriptor means, AqSolDB seed 0  (paper_v2/results/profiles_solubility_aqsoldb_seed0.json)
No embedded title; 17 cm width.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = os.path.join("paper_v2", "results")
E = json.load(open(os.path.join(R, "eta2_across_seeds.json")))
P = json.load(open(os.path.join(R, "profiles_solubility_aqsoldb_seed0.json")))
OUT = os.path.join("paper_v2", "figures")
INK, SEC = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

ds = [("solubility_aqsoldb", "Solubility\n(AqSolDB)"), ("caco2_wang", "Caco-2\n(Wang)"),
      ("ld50_zhu", "LD50\n(Zhu)"), ("lipophilicity_astrazeneca", "Lipophilicity\n(AstraZeneca)")]
rows = [("LogP", "LogP"), ("ArRings", "Aromatic rings"), ("MW", "MW"), ("TPSA", "Polar surface area"),
        ("HBA", "H-bond acceptors"), ("HBD", "H-bond donors"), ("RotBonds", "Rotatable bonds"), ("RingCount", "Ring count")]
M = np.array([[E[d]["eta2"][r]["mean"] for d, _ in ds] for r, _ in rows])
S = np.array([[E[d]["eta2"][r]["sd"] for d, _ in ds] for r, _ in rows])
print("eta2 mean +- sd (rows: " + ", ".join(l for _, l in rows) + ")")
for i, (_, l) in enumerate(rows):
    print(f"{l:20s}", "  ".join(f"{M[i,j]:.2f}±{S[i,j]:.2f}" for j in range(4)))

fig = plt.figure(figsize=(6.7, 8.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.035], height_ratios=[8, 5.4], hspace=0.28, wspace=0.05)
axA, caxA = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axB, caxB = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# ---- A
cmap = LinearSegmentedColormap.from_list("seq", ["#f0f6fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
im = axA.imshow(M, cmap=cmap, vmin=0, vmax=0.45, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        c = "#ffffff" if M[i, j] > 0.24 else INK
        axA.text(j, i - 0.12, f"{M[i,j]:.2f}", ha="center", va="center", color=c, fontsize=7.5, fontweight="bold")
        axA.text(j, i + 0.24, f"± {S[i,j]:.2f}", ha="center", va="center", color=c, fontsize=5.8)
axA.set_xticks(range(4)); axA.set_xticklabels([l for _, l in ds])
axA.set_yticks(range(len(rows))); axA.set_yticklabels([l for _, l in rows])
axA.tick_params(length=0); [s.set_visible(False) for s in axA.spines.values()]
axA.set_title("η² of dominant-expert assignment (mean ± s.d., 5 seeds)", loc="left", color=INK)
cb = fig.colorbar(im, cax=caxA); cb.set_label("η² (mean over seeds)"); cb.outline.set_visible(False)
axA.text(-0.30, 1.06, "A", transform=axA.transAxes, fontsize=11, fontweight="bold")

# ---- B
cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "ArRings"]
P = sorted(P, key=lambda r: -r["n"])
Z = np.array([[r[f"z_{c}"] for c in cols] for r in P])
V = np.array([[r[c] for c in cols] for r in P])
im2 = axB.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        axB.text(j, i, f"{V[i,j]:.1f}", ha="center", va="center", fontsize=7,
                 color="white" if abs(Z[i, j]) > 1.3 else INK)
axB.set_xticks(range(len(cols))); axB.set_xticklabels(["ArRings" if c == "ArRings" else c for c in cols])
axB.set_yticks(range(len(P))); axB.set_yticklabels([f"E{r['expert']} (n = {r['n']:,})" for r in P])
axB.tick_params(length=0); [s.set_visible(False) for s in axB.spines.values()]
axB.set_title("Per-expert descriptor means, AqSolDB, seed 0 (text: raw mean; colour: z-score)", loc="left", color=INK)
cb2 = fig.colorbar(im2, cax=caxB); cb2.set_label("z-score of expert mean vs all molecules"); cb2.outline.set_visible(False)
axB.text(-0.30, 1.10, "B", transform=axB.transAxes, fontsize=11, fontweight="bold")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig3_specialization.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig3_specialization.pdf"), bbox_inches="tight")
print("saved Fig3_specialization.png/.pdf")
