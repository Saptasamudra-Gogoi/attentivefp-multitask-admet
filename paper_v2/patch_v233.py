"""
patch_v233.py - v2.32 -> v2.33: removes the withdrawn-results banner, makes Table 1 regression-only,
inserts the new classification table (Table 1b), and fixes the Table 1 cross-reference.
Never modifies v2.32. Run from paper_v2:   python patch_v233.py
"""
import re

SRC = "manuscript_v2.32_DRAFT.md"
DST = "manuscript_v2.33_DRAFT.md"
lines = open(SRC, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
tbl = open("table1_classification_v2.32.md", encoding="utf-8").read().replace("\r\n", "\n")
tbl = tbl.replace(" (replaces the classification rows of Table 1)", "")
done, fails = [], []


def asc(s):
    return s.encode("ascii", "replace").decode()


# ---- 1. banner: leading blockquote lines
n = 0
while lines and lines[0].startswith(">"):
    lines.pop(0)
    n += 1
while lines and lines[0].strip() == "":
    lines.pop(0)
(done if n else fails).append(f"banner: removed {n} leading blockquote lines")

# ---- 2. Table 1: regression-only caption, drop classification rows, insert Table 1b
CAPTION = (
    "**Table 1 | The expert plug-in improves regression over a plain graph convolutional backbone but "
    "trails a stronger attention backbone.** RMSE (↓, lower error is better) on the three MoleculeNet "
    "regression datasets. MoE-GCN and GCN are both mean ± s.d. over the same five seeds on one "
    "deterministic Bemis–Murcko scaffold split (Methods; the standard deviations exclude split variance), "
    "with a matched Optuna hyperparameter-search budget. AttentiveFP values are our own scaffold-split "
    "runs (five seeds), reported as mean ± s.d. GCN, plain graph convolutional backbone; MoE-GCN, the same "
    "backbone with the expert plug-in; AttentiveFP, attention-based backbone; GROVER, model pretrained on "
    "10⁷ molecules (literature reference), values taken from the GROVER<sub>large</sub> configuration "
    "reported by Rong et al. [8] under their original random-split protocol, not independently reproduced "
    "here. **†** marks every literature-sourced, cross-protocol cell (all GROVER cells): none carries a "
    "standard deviation in this table, and none was evaluated under our scaffold-split protocol, so these "
    "values are reference points only. **‡** marks FreeSolv's Δ, which is not statistically significant "
    "at the per-dataset level (paired one-sided *t*-test *P* = 0.31; see Results) even though MoE-GCN's "
    "lower mean still counts as a win for the pooled sign test in Table 4. Δ is the relative percentage "
    "RMSE reduction of MoE-GCN relative to GCN. Classification results, obtained under a different split "
    "protocol, are in Table 1b."
)
a = [i for i, l in enumerate(lines) if l.startswith("**Table 1 |")]
b = [i for i, l in enumerate(lines) if l.startswith("**Table 2 |")]
if len(a) != 1 or len(b) != 1 or a[0] >= b[0]:
    fails.append(f"table 1: anchors found {len(a)} / {len(b)}")
else:
    a, b = a[0], b[0]
    CLS = {"BBBP", "BACE", "Tox21", "ToxCast", "SIDER", "ClinTox", "HIV"}
    kept, dropped = [], 0
    for l in lines[a:b]:
        if l.startswith("|") and l.strip("|").split("|")[0].strip() in CLS:
            dropped += 1
            continue
        kept.append(l)
    kept[0] = CAPTION
    while kept and kept[-1].strip() == "":
        kept.pop()
    kept += ["", tbl.rstrip("\n"), ""]
    lines[a:b] = kept
    (done if dropped == 7 else fails).append(f"table 1: dropped {dropped} classification rows (expected 7)")

text = "\n".join(lines)

# ---- 3. cross-reference to the classification table
text, k = re.subn(r"(BBBP [^\d\s]0\.020; Table 1)\)", r"\g<1>b)", text)
(done if k == 1 else fails).append(f"cross-reference to Table 1b: {k} (expected 1)")

open(DST, "w", encoding="utf-8", newline="\n").write(text)

print("APPLIED:")
for d in done:
    print("  ok  ", asc(d))
print("\nFAILED (do by hand):" if fails else "\nNo failures.")
for f in fails:
    print("  FAIL", asc(f))

print("\nLEFTOVER MARKERS:")
pats = [r"\[v2\.3", r"withdrawn", r"pending", r"stratified", r"ToxCast", r"\bHIV\b", r"seven classification",
        r"21 TDC", r"17 of 21", r"Table 1\b(?!b)", r"WORKING DRAFT"]
for i, l in enumerate(text.split("\n"), 1):
    hit = [p for p in pats if re.search(p, l)]
    if hit:
        print(f"  {i}: [{', '.join(hit)}] {asc(l[:105])}")
print(f"\nWrote {DST}")
