"""
s3_recompute.py - one collapse rule, applied everywhere, computed from the saved files.

Rule (decided from active_per_seed, before looking at any P-value):
  COLLAPSED       : no seed has more than 2 active experts   -> excluded from primary analysis
  LIMITED ROUTING : at least one seed has fewer than 3 active experts (but not collapsed)
PPBR is already absent from tableS3.csv (single expert) and counts as collapsed.

Run from paper_v2:   python s3_recompute.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

d = pd.read_csv("results/tableS3.csv")
J = json.load(open("results/tableS1_S3.json"))["datasets"]
act = {k: v["active_per_seed"] for k, v in J.items()}

names = sorted(d.dataset.unique())
collapsed = [k for k in names if max(act[k]) <= 2]
limited = [k for k in names if k not in collapsed and min(act[k]) < 3]

print("Dataset classification (active experts per seed):")
for k in names:
    tag = "COLLAPSED" if k in collapsed else ("limited" if k in limited else "")
    print(f"  {k:34s} min {min(act[k]):2d} max {max(act[k]):2d}  {tag}")
print("  (PPBR: single expert, absent from this file, collapsed)\n")

SETS = {
    "ALL (as in v2.31 main analysis)": names,
    "PRIMARY (collapsed removed)": [k for k in names if k not in collapsed],
    "SENSITIVITY (collapsed + limited removed)": [k for k in names if k not in collapsed and k not in limited],
}

rng = np.random.RandomState(0)

def pooled_rho(df):
    r, _ = spearmanr(df.abs_endpoint_corr, df.eta2_mean_over_seeds)
    return r

for label, keep in SETS.items():
    s = d[d.dataset.isin(keep)]
    top = s.loc[s.groupby("dataset").eta2_mean_over_seeds.idxmax()]
    n_ar = int((top.descriptor == "ArRings").sum())
    means = s.groupby("descriptor").eta2_mean_over_seeds.mean()
    rho = pooled_rho(s)
    boots = []
    groups = {k: s[s.dataset == k] for k in keep}
    for _ in range(2000):
        pick = rng.choice(keep, size=len(keep), replace=True)
        b = pd.concat([groups[k] for k in pick])
        if b.abs_endpoint_corr.nunique() > 1 and b.eta2_mean_over_seeds.nunique() > 1:
            boots.append(pooled_rho(b))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"== {label}: {len(keep)} datasets, {len(s)} pairs")
    print(f"   ArRings largest eta2 on {n_ar} of {len(keep)}")
    print(f"   mean eta2: ArRings {means['ArRings']:.3f}  LogP {means['LogP']:.3f}  RingCount {means['RingCount']:.3f}")
    print(f"   pooled Spearman rho = {rho:.2f} (95% CI {lo:.2f} to {hi:.2f}, dataset bootstrap, 2000)\n")

print("Leave-descriptor-out pooled rho (point estimates):")
for label, keep in SETS.items():
    s = d[d.dataset.isin(keep)]
    out = []
    for drop in (["ArRings"], ["RingCount"], ["ArRings", "RingCount"]):
        t = s[~s.descriptor.isin(drop)]
        out.append(f"drop {'+'.join(drop)}: {pooled_rho(t):.2f}")
    print(f"  {label}: " + "   ".join(out))
print()

print("Regression sign test, 11 datasets (Lipophilicity once): 9 wins, losses = half-life, hepatocyte")
print("  excluding PPBR + microsome (both wins): 7 of 9, one-sided P = "
      f"{binomtest(7, 9, 0.5, alternative='greater').pvalue:.3f}")
print("  excluding PPBR + microsome + hepatocyte (collapse rule above): 7 of 8, one-sided P = "
      f"{binomtest(7, 8, 0.5, alternative='greater').pvalue:.3f}")
print("  Report BOTH rows. The second applies the single rule, but its P-value was seen")
print("  before the rule was written down, so do not describe it as pre-specified.")
