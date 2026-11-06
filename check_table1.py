"""
check_table1.py - recompute Table 1 regression rows (MoE-GCN vs GCN) from the saved result files.
Run from the repo ROOT:   python check_table1.py
Compare with Table 1 in the manuscript: GCN 1.120+/-0.036, 3.654+/-0.544, 0.745+/-0.007;
MoE-GCN 1.067+/-0.046, 3.591+/-0.527, 0.722+/-0.008; one-sided paired t-test P = 0.017, 0.31, <0.001.
"""
import json

import numpy as np
from scipy import stats

moe = json.load(open("results_moegcn_regr.json"))
gcn = json.load(open("results_gcn_matched.json"))
print(f"{'dataset':10s} {'model':5s} {'n':>2s} {'mean':>7s} {'sd(ddof0)':>10s} {'sd(ddof1)':>10s}")
for k in ("ESOL", "FreeSolv", "Lipo"):
    g = np.array(gcn[k]["seeds"], float)
    m = np.array(moe[k]["seeds"], float)
    for name, a in (("GCN", g), ("MoE", m)):
        print(f"{k:10s} {name:5s} {len(a):2d} {a.mean():7.4f} {a.std():10.4f} {a.std(ddof=1):10.4f}")
    if len(g) == len(m):
        t = stats.ttest_rel(g, m, alternative="greater")
        d = g - m
        print(f"{'':10s} paired (GCN - MoE) mean {d.mean():+.4f}, one-sided P = {t.pvalue:.4f}, "
              f"MoE lower in {(d > 0).sum()}/{len(d)} seeds")
    else:
        print(f"{'':10s} seed counts differ ({len(g)} vs {len(m)}): cannot pair")
    print(f"{'':10s} relative reduction: {100 * (g.mean() - m.mean()) / g.mean():+.1f}%\n")
