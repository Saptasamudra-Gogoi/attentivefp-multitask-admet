"""
fix_table2_rounding.py - correct four double-rounded Table 2 cells (MoE-GCN column) to the values the
result file rounds to. Run from paper_v2:   python fix_table2_rounding.py
Each pattern must match exactly once or NOTHING is written.
"""
import re

P = "manuscript_v2.33_DRAFT.md"
t = open(P, encoding="utf-8").read().replace("\r\n", "\n")
R = [
    ("BBB MoE s.d. 0.016 -> 0.015", r"(\| 0\.859 \S 0\.002 \| 0\.857 \S )0\.016", r"\g<1>0.015"),
    ("CYP2D6 substrate MoE 0.777 -> 0.776", r"(\| 0\.802 \S 0\.010 \| )0\.777", r"\g<1>0.776"),
    ("hERG MoE 0.670 -> 0.669", r"(\| 0\.736 \S 0\.019 \| )0\.670", r"\g<1>0.669"),
    ("Lipophilicity MoE 0.547 -> 0.546", r"(\| 0\.594 \S 0\.007 \| )0\.547", r"\g<1>0.546"),
]
bad = [f"{n}: matched {len(re.findall(p, t))}" for n, p, _ in R if len(re.findall(p, t)) != 1]
if bad:
    print("NOT WRITTEN:", *bad, sep="\n  ")
    raise SystemExit(1)
for _, p, r in R:
    t = re.sub(p, r, t)
open(P, "w", encoding="utf-8", newline="\n").write(t)
print("ok:", ", ".join(n for n, _, _ in R))
