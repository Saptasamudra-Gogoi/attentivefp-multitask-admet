"""
make_fig4.py - Figure 4 (three panels), generated from the tables in the manuscript so the figure
cannot disagree with them. Panels:
  A  MoleculeNet regression, GCN vs MoE-GCN (Table 1), RMSE relative to GCN
  B  EC-MPNN with and without MoE (Results text), RMSE relative to plain EC-MPNN
  C  Dense-baseline ablation (Table 3), RMSE relative to MoE-GCN
Layout for 17 cm print width (6.7 in), no embedded title (the caption carries it).
Run from paper_v2:   python make_fig4.py manuscript_v2.33_DRAFT.md
Writes figures/Fig4_performance_summary.png and .pdf
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MS = sys.argv[1] if len(sys.argv) > 1 else "manuscript_v2.33_DRAFT.md"
text = open(MS, encoding="utf-8").read()
lines = text.split("\n")


def block(start, end):
    a = next(i for i, l in enumerate(lines) if l.startswith(start))
    b = next(i for i, l in enumerate(lines) if l.startswith(end))
    return lines[a:b]


def cells(l):
    return [c.strip().replace("*", "") for c in l.strip().strip("|").split("|")]


NUM = re.compile(r"([0-9]+\.[0-9]+)\s*±\s*([0-9]+\.[0-9]+)")


def ms(c):
    m = NUM.search(c)
    return (float(m.group(1)), float(m.group(2))) if m else None


# ---- Panel A: Table 1
A = {}
for l in block("**Table 1 |", "**Table 1b |"):
    if l.startswith("|"):
        c = cells(l)
        if c[0] in ("ESOL", "FreeSolv", "Lipophilicity") and len(c) >= 4:
            A[c[0]] = (ms(c[2]), ms(c[3]))
assert len(A) == 3 and all(v[0] and v[1] for v in A.values()), f"Table 1 parse failed: {A}"

# ---- Panel B: EC-MPNN values quoted in Results (guarded: must still be in the text)
for s in ("1.105 versus 1.163", "2.865 versus 3.089", "0.712 versus 0.701"):
    assert s in text, f"EC-MPNN value {s!r} no longer in the manuscript text"
B = {"ESOL": (1.163, 1.105), "FreeSolv": (3.089, 2.865), "Lipophilicity": (0.701, 0.712)}

# ---- Panel C: Table 3
C = {}
for l in block("**Table 3 |", "**Table 4 |"):
    if l.startswith("|"):
        c = cells(l)
        if len(c) == 5 and ms(c[1]):
            C[c[0]] = [ms(x) for x in c[1:]]
assert len(C) == 5 and all(all(v) for v in C.values()), f"Table 3 parse failed: {list(C)}"

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6})
fig = plt.figure(figsize=(6.7, 6.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.32)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
col = {"gcn": "#9aa5b1", "moe": "#c0392b", "du": "#2e86ab", "dw": "#7fb069", "pl": "#e0a458"}
w = 0.36
LAB = dict(fontsize=5.5, rotation=90, color="white", ha="center", va="bottom")

# A
for j, n in enumerate(A):
    (g, gs_), (m, mss) = A[n]
    axA.bar(j - w / 2, 1.0, w, yerr=gs_ / g, color=col["gcn"], capsize=2, label="GCN" if j == 0 else None)
    axA.bar(j + w / 2, m / g, w, yerr=mss / g, color=col["moe"], capsize=2, label="MoE-GCN" if j == 0 else None)
    axA.text(j - w / 2, 0.02, f"{g:.3f}", **LAB)
    axA.text(j + w / 2, 0.02, f"{m:.3f}", **LAB)
axA.set_xticks(np.arange(3), list(A))
axA.set_ylim(0, 1.35)
axA.set_ylabel("RMSE relative to GCN\n(lower is better)")
axA.set_title("A  MoleculeNet regression", loc="left", fontweight="bold")
axA.legend(frameon=False, loc="upper right")
axA.set_xlabel("Error bars: s.d. over five seeds (Table 1)", fontsize=6)

# B
for j, n in enumerate(B):
    p, m = B[n]
    axB.bar(j - w / 2, 1.0, w, color=col["gcn"], label="EC-MPNN" if j == 0 else None)
    axB.bar(j + w / 2, m / p, w, color=col["moe"], label="EC-MPNN + MoE" if j == 0 else None)
    axB.text(j - w / 2, 0.02, f"{p:.3f}", **LAB)
    axB.text(j + w / 2, 0.02, f"{m:.3f}", **LAB)
axB.set_xticks(np.arange(3), list(B))
axB.set_ylim(0, 1.35)
axB.set_ylabel("RMSE relative to plain EC-MPNN")
axB.set_title("B  Cross-architecture transfer", loc="left", fontweight="bold")
axB.legend(frameon=False, loc="upper right")
axB.set_xlabel("No s.d. reported for this comparison", fontsize=6)

# C
labels = ["MoE-GCN", "Dense-uniform", "Dense-wide", "Plain-GCN"]
ckeys = ["moe", "du", "dw", "pl"]
w2 = 0.2
for j, n in enumerate(C):
    base = C[n][0][0]
    for k in range(4):
        mean, sd = C[n][k]
        hatch = "///" if (k == 3 and n in ("Caco-2", "Solubility (AqSolDB)")) else None
        axC.bar(j + (k - 1.5) * w2, mean / base, w2, yerr=sd / base, color=col[ckeys[k]], capsize=1.5,
                hatch=hatch, edgecolor="white" if hatch else None, label=labels[k] if j == 0 else None)
axC.axhline(1.0, color="k", lw=0.6, ls=":")
axC.set_xticks(np.arange(len(C)), [n.replace(" (AqSolDB)", "") for n in C])
axC.set_ylabel("RMSE relative to MoE-GCN")
axC.set_title("C  Dense-baseline ablation", loc="left", fontweight="bold")
axC.set_ylim(0, 1.5)
axC.legend(frameon=False, ncol=4, loc="upper center")
axC.set_xlabel("Hatched: Plain-GCN split not confirmed to match the other arms; not interpreted", fontsize=6)

for a in (axA, axB, axC):
    a.spines[["top", "right"]].set_visible(False)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Fig4_performance_summary.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/Fig4_performance_summary.pdf", bbox_inches="tight")
print("wrote figures/Fig4_performance_summary.png and .pdf")
