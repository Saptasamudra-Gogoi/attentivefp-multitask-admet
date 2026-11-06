"""
check_tables23.py - compare Table 2 (MoE-GCN column, optionally GCN) and Table 3 in the manuscript with
the saved result files. Prints only mismatches and a summary.

Run from the repo ROOT:
    python check_tables23.py                       # Table 3 + Table 2 MoE-GCN column
    python check_tables23.py results_gcn_tdc.json  # also the Table 2 GCN column (file name = yours)

Tolerance: a table value is the file value rounded to 3 decimals, so |file - table| <= 0.0005 is a match.
s.d. matches if it agrees with the file's own 'std' or with the s.d. recomputed from 'seeds'
(ddof 0 or 1); the convention that matched is reported.
"""
import json
import re
import sys

import numpy as np

MS = "paper_v2/manuscript_v2.33_DRAFT.md"
lines = open(MS, encoding="utf-8").read().split("\n")
TOL = 0.00051
NUM = re.compile(r"([0-9]+\.[0-9]+)\s*±\s*([0-9]+\.[0-9]+)")


def block(a, b):
    i = next(k for k, l in enumerate(lines) if l.startswith(a))
    j = next(k for k, l in enumerate(lines) if l.startswith(b))
    return lines[i:j]


def cells(l):
    return [c.strip().replace("*", "") for c in l.strip().strip("|").split("|")]


def ms(c):
    m = NUM.search(c)
    return (float(m.group(1)), float(m.group(2))) if m else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sd_ok(sd_t, entry, seeds=None):
    s = np.array(seeds if seeds is not None else entry.get("seeds", []), float)
    cand = {}
    if seeds is None and "std" in entry:
        cand["file std"] = entry["std"]
    if len(s) > 1:
        cand["ddof0"] = s.std()
        cand["ddof1"] = s.std(ddof=1)
    for name, v in cand.items():
        if abs(v - sd_t) <= TOL:
            return name
    return None


checked = 0
bad = []

# ------------------------------------------------------------------ Table 3
abl = json.load(open("ablation_routing_results.json"))
ARMS = ["moe", "dense-uniform", "dense-wide", "plain"]


def find_ds(name, keys):
    n = norm(name)
    for rule in (lambda kn: kn == n, lambda kn: n.startswith(kn) and len(kn) >= 4,
                 lambda kn: kn.startswith(n), lambda kn: "aqsol" in n and "aqsol" in kn):
        for k in keys:
            if rule(norm(k)):
                return k
    return None


print("Table 3 vs ablation_routing_results.json")
for l in block("**Table 3 |", "**Table 4 |"):
    if not l.startswith("|"):
        continue
    c = cells(l)
    if len(c) != 5 or not ms(c[1]):
        continue
    key = find_ds(c[0], list(abl))
    if key is None:
        bad.append(f"Table 3: no JSON key for dataset {c[0]!r}; keys are {list(abl)}")
        continue
    for k in range(4):
        arm = next((a for a in abl[key] if a.lower().startswith(ARMS[k])), None)
        if arm is None:
            bad.append(f"Table 3 {c[0]}: no arm starting with {ARMS[k]!r}; arms {list(abl[key])}")
            continue
        e = abl[key][arm]
        mt, st = ms(c[1 + k])
        checked += 2
        if abs(e["mean"] - mt) > TOL:
            bad.append(f"Table 3 {c[0]} / {arm}: mean table {mt} vs file {e['mean']}")
        if sd_ok(st, e) is None:
            bad.append(f"Table 3 {c[0]} / {arm}: s.d. table {st} vs file std {e.get('std')}")

# ------------------------------------------------------------------ Table 2
NAME = {
    "AMES": "ames", "BBB (Martins)": "bbb_martins", "Bioavailability (Ma)": "bioavailability_ma",
    "Caco-2 (Wang)": "caco2_wang", "Clearance, hepatocyte (AZ)": "clearance_hepatocyte_az",
    "Clearance, microsome (AZ)": "clearance_microsome_az",
    "CYP2C9 substrate (Carbon-Mangels)": "cyp2c9_substrate_carbonmangels", "CYP2C9 (Veith)": "cyp2c9_veith",
    "CYP2D6 substrate (Carbon-Mangels)": "cyp2d6_substrate_carbonmangels", "CYP2D6 (Veith)": "cyp2d6_veith",
    "CYP3A4 substrate (Carbon-Mangels)": "cyp3a4_substrate_carbonmangels", "CYP3A4 (Veith)": "cyp3a4_veith",
    "DILI": "dili", "Half-life (Obach)": "half_life_obach", "hERG": "herg", "HIA (Hou)": "hia_hou",
    "LD50 (Zhu)": "ld50_zhu", "Lipophilicity (AstraZeneca)": "lipophilicity_astrazeneca",
    "Pgp inhibition (Broccatelli)": "pgp_broccatelli", "PPBR (AZ)": "ppbr_az",
    "Solubility (AqSolDB)": "solubility_aqsoldb", "VDss (Lombardo)": "vdss_lombardo",
}
files = [("MoE-GCN", "results_moegcn_tdc_v2.json", 3, 3)]
if len(sys.argv) > 1:
    files.append(("GCN", sys.argv[1], 2, None))
for label, path, col, nseed in files:
    d = json.load(open(path))
    print(f"Table 2 {label} vs {path}")
    for l in block("**Table 2 |", "**Table 3 |"):
        if not l.startswith("|"):
            continue
        c = cells(l)
        if len(c) != 4 or c[0] not in NAME or not ms(c[col]):
            continue
        key = NAME[c[0]]
        if key not in d:
            bad.append(f"Table 2 {label}: {key} missing from {path}")
            continue
        e = d[key]
        seeds = e["seeds"][:nseed] if nseed else e["seeds"]
        mean_f = float(np.mean(seeds))
        mt, st = ms(c[col])
        checked += 2
        if abs(mean_f - mt) > TOL:
            bad.append(f"Table 2 {label} {c[0]}: mean table {mt} vs file (first {len(seeds)} seeds) {mean_f:.4f}")
        if sd_ok(st, e, seeds=seeds) is None:
            bad.append(f"Table 2 {label} {c[0]}: s.d. table {st} vs ddof0 {np.std(seeds):.4f} / "
                       f"ddof1 {np.std(seeds, ddof=1):.4f}")

print(f"\n{checked} cells checked, {len(bad)} problems")
for b in bad:
    print("  ", b)
