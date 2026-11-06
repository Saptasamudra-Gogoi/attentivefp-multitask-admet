"""
patch_authors.py - make the manuscript sole-author: author block, affiliation, correspondence email,
Funding, Authors' contributions, Acknowledgements, AI-tool statement.

Run from paper_v2:   python patch_authors.py your.email@example.com
Each edit must match exactly once; if any does not, NOTHING is written.
"""
import re
import sys

if len(sys.argv) != 2 or "@" not in sys.argv[1]:
    sys.exit("usage: python patch_authors.py your.email@example.com")
EMAIL = sys.argv[1].strip()

P = "manuscript_v2.33_DRAFT.md"
text = open(P, encoding="utf-8").read().replace("\r\n", "\n")
lines = text.split("\n")
errors, done = [], []


def find(prefix=None, contains=None):
    if prefix is not None:
        return [i for i, l in enumerate(lines) if l.startswith(prefix)]
    return [i for i, l in enumerate(lines) if contains in l]


def repline(name, prefix, new):
    idx = find(prefix=prefix)
    if len(idx) != 1:
        errors.append(f"{name}: {len(idx)} lines start with {prefix!r}, expected 1")
        return
    lines[idx[0]] = new
    done.append(name)


def delline(name, prefix):
    idx = find(prefix=prefix)
    if len(idx) != 1:
        errors.append(f"{name}: {len(idx)} lines start with {prefix!r}, expected 1")
        return
    del lines[idx[0]]
    done.append(name)


repline("author line", "Saptasamudra Gogoi", "Saptasamudra Gogoi¹·*")
delline("affiliation 2", "²State Key Laboratory of Green Pesticide")

idx = find(contains="yvquan.li@gzu.edu.cn")
if len(idx) != 1:
    errors.append(f"correspondence email: {len(idx)} lines contain the old address, expected 1")
else:
    lines[idx[0]] = lines[idx[0]].replace("yvquan.li@gzu.edu.cn", EMAIL)
    done.append("correspondence email")

repline("funding", "**Funding.**", "**Funding.** This research received no specific funding.")
repline("authors' contributions", "**Authors' contributions",
        "**Authors' contributions (CRediT taxonomy).** Conceptualization, Methodology, Software, "
        "Investigation, Formal analysis, Visualization, Writing – original draft, Writing – review & "
        "editing: S.G.")
repline("acknowledgements", "**Acknowledgements.**",
        "**Acknowledgements.** The author thanks the developers of RDKit, PyTorch Geometric, and the "
        "Therapeutics Data Commons for making their tools and datasets publicly available.")
repline("AI-tool statement", "Anthropic's Claude (a large language model) assisted",
        "Anthropic's Claude (a large language model) assisted with writing and checking analysis scripts, "
        "statistical re-computation from saved result files, and editing of the manuscript text. The "
        "author designed the study, ran all experiments, verified the reported numbers against the saved "
        "outputs, and takes full responsibility for the content.")

if errors:
    print("NOT WRITTEN. Problems:")
    for e in errors:
        print("  -", e.encode("ascii", "replace").decode())
    sys.exit(1)

open(P, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("Applied:", ", ".join(done))
print("\nRemaining mentions to check by hand (line: text):")
for i, l in enumerate(lines, 1):
    if re.search(r"Yuquan|\bY\.L\b|Green Pesticide|gzu\.edu|\bauthors\b|Supercomputing|NSFC|National Natural", l, re.I):
        print(f"  {i}: {l[:110].encode('ascii', 'replace').decode()}")
