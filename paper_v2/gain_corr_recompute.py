"""
gain_corr_recompute.py - specialization strength vs MoE-GCN gain, on 21 / 19 / 18 datasets.

Gain per dataset (percent, positive = MoE better), from Table 2 of the manuscript (parsed from the
file, not retyped):  MAE rows: (GCN - MoE) / GCN ;  AUROC / Spearman rows: (MoE - GCN) / GCN.
Specialization per dataset: largest single-descriptor eta2 (mean over seeds), and the mean eta2
over the 8 descriptors, from results/tableS3.csv.

Same collapse rule as s3_recompute.py. Run from paper_v2:
    python gain_corr_recompute.py manuscript_v2.31_DRAFT.md
CHECK: the 21-dataset row should land near the paper's rho = -0.17 (CI -0.60 to 0.31, P = 0.48) and
-0.15 (-0.61 to 0.33). If it does, the gain definition is confirmed.
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MS = sys.argv[1] if len(sys.argv) > 1 else "manuscript_v2.31_DRAFT.md"

NAME = {
    "AMES": "ames", "BBB (Martins)": "bbb_martins", "Bioavailability (Ma)": "bioavailability_ma",
    "Caco-2 (Wang)": "caco2_wang", "Clearance, hepatocyte (AZ)": "clearance_hepatocyte_az",
    "Clearance, microsome (AZ)": "clearance_microsome_az",
    "CYP2C9 substrate (Carbon-Mangels)": "cyp2c9_substrate_carbonmangels",
    "CYP2C9 (Veith)": "cyp2c9_veith",
    "CYP2D6 substrate (Carbon-Mangels)": "cyp2d6_substrate_carbonmangels",
    "CYP2D6 (Veith)": "cyp2d6_veith",
    "CYP3A4 substrate (Carbon-Mangels)": "cyp3a4_substrate_carbonmangels",
    "CYP3A4 (Veith)": "cyp3a4_veith", "DILI": "dili", "Half-life (Obach)": "half_life_obach",
    "hERG": "herg", "HIA (Hou)": "hia_hou", "LD50 (Zhu)": "ld50_zhu",
    "Lipophilicity (AstraZeneca)": "lipophilicity_astrazeneca",
    "Pgp inhibition (Broccatelli)": "pgp_broccatelli", "PPBR (AZ)": "ppbr_az",
    "Solubility (AqSolDB)": "solubility_aqsoldb", "VDss (Lombardo)": "vdss_lombardo",
}

lines = open(MS, encoding="utf-8").read().splitlines()
a = next(i for i, l in enumerate(lines) if l.startswith("**Table 2 |"))
b = next(i for i, l in enumerate(lines) if l.startswith("**Table 3 |"))
gain = {}
for l in lines[a:b]:
    if not l.startswith("|") or l.startswith("|---"):
        continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) != 4 or c[0] == "Dataset":
        continue
    key = NAME.get(c[0])
    if key is None:
        print("UNMAPPED row:", c[0])
        continue
    g = float(re.match(r"[0-9.]+", c[2]).group())
    m = float(re.match(r"[0-9.]+", c[3]).group())
    gain[key] = 100 * ((g - m) / g if "MAE" in c[1] else (m - g) / g)

d = pd.read_csv("results/tableS3.csv")
J = json.load(open("results/tableS1_S3.json"))["datasets"]
act = {k: v["active_per_seed"] for k, v in J.items()}
names = sorted(d.dataset.unique())
collapsed = [k for k in names if max(act[k]) <= 2]
limited = [k for k in names if k not in collapsed and min(act[k]) < 3]

topeta = d.groupby("dataset").eta2_mean_over_seeds.max()
meaneta = d.groupby("dataset").eta2_mean_over_seeds.mean()
missing = [k for k in names if k not in gain]
if missing:
    sys.exit(f"No gain parsed for: {missing}")

print("Gains parsed from Table 2 (percent, + = MoE better):")
for k in names:
    print(f"  {k:34s} gain {gain[k]:+7.2f}   top eta2 {topeta[k]:.3f}   mean eta2 {meaneta[k]:.3f}")
print()

SETS = {
    "ALL 21 (as in v2.31)": names,
    "PRIMARY 19 (collapsed removed)": [k for k in names if k not in collapsed],
    "SENSITIVITY 18 (collapsed + limited removed)": [k for k in names if k not in collapsed and k not in limited],
}
rng = np.random.RandomState(0)


def run(keep, eta):
    x = np.array([eta[k] for k in keep])
    y = np.array([gain[k] for k in keep])
    rho, p = spearmanr(x, y)
    boots = []
    for _ in range(5000):
        i = rng.randint(0, len(keep), len(keep))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, lo, hi, p


for label, keep in SETS.items():
    r1 = run(keep, topeta)
    r2 = run(keep, meaneta)
    print(f"== {label}: {len(keep)} datasets")
    print(f"   largest-descriptor eta2 vs gain: rho = {r1[0]:+.2f} (95% CI {r1[1]:+.2f} to {r1[2]:+.2f}, P = {r1[3]:.2f})")
    print(f"   mean eta2 (8 descriptors) vs gain: rho = {r2[0]:+.2f} (95% CI {r2[1]:+.2f} to {r2[2]:+.2f})\n")
