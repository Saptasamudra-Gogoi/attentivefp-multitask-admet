"""
make_fig4.py - Figure 4 (three panels), generated from the tables in the manuscript so the figure
cannot disagree with them. Panels:
  A  MoleculeNet regression, GCN vs MoE-GCN (Table 1), RMSE relative to GCN
  B  EC-MPNN with and without MoE (Results text), RMSE relative to plain EC-MPNN
  C  Dense-baseline ablation (Table 3), RMSE relative to MoE-GCN
All panels share a zero baseline. Run from paper_v2:   python make_fig4.py manuscript_v2.33_DRAFT.md
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

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.5]})
col = {"gcn": "#9aa5b1", "moe": "#c0392b", "du": "#2e86ab", "dw": "#7fb069", "pl": "#e0a458"}
w = 0.36

# A
names = list(A)
for j, n in enumerate(names):
    (g, gs), (m, mss) = A[n]
    ax[0].bar(j - w / 2, 1.0, w, yerr=gs / g, color=col["gcn"], capsize=3, label="GCN" if j == 0 else None)
    ax[0].bar(j + w / 2, m / g, w, yerr=mss / g, color=col["moe"], capsize=3, label="MoE-GCN" if j == 0 else None)
    ax[0].text(j - w / 2, 0.02, f"{g:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
    ax[0].text(j + w / 2, 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
ax[0].set_xticks(np.arange(3), names)
ax[0].set_ylim(0, 1.35)
ax[0].set_ylabel("RMSE relative to GCN (lower is better)")
ax[0].set_title("A  MoleculeNet regression", loc="left", fontweight="bold", pad=22)
ax[0].text(0, 1.03, "3 of the 12 regression datasets in the pooled sign test (P = 0.019, Table 4)",
           transform=ax[0].transAxes, fontsize=7.5, va="bottom")
ax[0].legend(frameon=False, loc="upper right")
ax[0].set_xlabel("Error bars: s.d. over five seeds (Table 1)", fontsize=7)

# B
for j, n in enumerate(B):
    p, m = B[n]
    ax[1].bar(j - w / 2, 1.0, w, color=col["gcn"], label="EC-MPNN" if j == 0 else None)
    ax[1].bar(j + w / 2, m / p, w, color=col["moe"], label="EC-MPNN + MoE" if j == 0 else None)
    ax[1].text(j - w / 2, 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
    ax[1].text(j + w / 2, 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
ax[1].set_xticks(np.arange(3), list(B))
ax[1].set_ylim(0, 1.35)
ax[1].set_ylabel("RMSE relative to plain EC-MPNN")
ax[1].set_title("B  Cross-architecture transfer", loc="left", fontweight="bold", pad=22)
ax[1].legend(frameon=False, loc="upper right")
ax[1].set_xlabel("No s.d. reported for this comparison", fontsize=7)

# C
labels = ["MoE-GCN", "Dense-uniform", "Dense-wide", "Plain-GCN"]
ckeys = ["moe", "du", "dw", "pl"]
w2 = 0.2
for j, n in enumerate(C):
    base = C[n][0][0]
    for k in range(4):
        mean, sd = C[n][k]
        hatch = "///" if (k == 3 and n in ("Caco-2", "Solubility (AqSolDB)")) else None
        ax[2].bar(j + (k - 1.5) * w2, mean / base, w2, yerr=sd / base, color=col[ckeys[k]], capsize=2,
                  hatch=hatch, edgecolor="white" if hatch else None, label=labels[k] if j == 0 else None)
ax[2].axhline(1.0, color="k", lw=0.6, ls=":")
ax[2].set_xticks(np.arange(len(C)), [n.replace(" (AqSolDB)", "") for n in C], fontsize=8)
ax[2].set_ylabel("RMSE relative to MoE-GCN")
ax[2].set_title("C  Dense-baseline ablation", loc="left", fontweight="bold", pad=22)
ax[2].set_ylim(0, 1.5)
ax[2].legend(frameon=False, ncol=2, fontsize=8, loc="upper center")
ax[2].set_xlabel("Hatched: Plain-GCN split not confirmed to match the other arms; not interpreted", fontsize=7)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Figure 4 | Performance summary", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Fig4_performance_summary.png", dpi=300)
fig.savefig("figures/Fig4_performance_summary.pdf")
print("wrote figures/Fig4_performance_summary.png and .pdf")
