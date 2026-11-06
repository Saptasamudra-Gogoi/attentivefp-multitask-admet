"""
fix_table1_sd.py - correct Table 1 MoE-GCN s.d. values and the ESOL P-value to match
results_moegcn_regr.json / results_gcn_matched.json (see check_table1.py output).
Run from paper_v2:   python fix_table1_sd.py
All patterns must match exactly once or NOTHING is written.
"""
import re

P = "manuscript_v2.33_DRAFT.md"
t = open(P, encoding="utf-8").read().replace("\r\n", "\n")

R = [
    ("ESOL MoE s.d.", r"(1\.067 \S )0\.046", r"\g<1>0.033"),
    ("FreeSolv MoE s.d.", r"(3\.591 \S )0\.527", r"\g<1>0.379"),
    ("Lipophilicity MoE s.d.", r"(0\.722 \S )0\.008", r"\g<1>0.006"),
    ("ESOL one-sided P", r"(\*t\*-test \*P\* = )0\.017", r"\g<1>0.018"),
    ("caption: s.d. convention", r"over the same five seeds on one deterministic",
     "over the same five seeds (population s.d.) on one deterministic"),
]
bad = []
for name, pat, rep in R:
    n = len(re.findall(pat, t))
    if n != 1:
        bad.append(f"{name}: matched {n}, expected 1")
if bad:
    print("NOT WRITTEN:")
    for b in bad:
        print("  ", b)
    raise SystemExit(1)
for name, pat, rep in R:
    t = re.sub(pat, rep, t)
open(P, "w", encoding="utf-8", newline="\n").write(t)
print("ok:", ", ".join(n for n, _, _ in R))
