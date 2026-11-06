"""
check_cited_files.py - every file the manuscript cites in `backticks` must exist in git.
Run from the repo ROOT:   python paper_v2\\check_cited_files.py paper_v2\\manuscript_v2.33_DRAFT.md
"""
import os
import re
import subprocess
import sys

ms = sys.argv[1] if len(sys.argv) > 1 else os.path.join("paper_v2", "manuscript_v2.33_DRAFT.md")
text = open(ms, encoding="utf-8").read()
names = sorted(set(re.findall(r"`([A-Za-z0-9_./\\-]+\.(?:py|json|csv|md|txt|npy|npz|yml|bat))`", text)))
tracked = [t for t in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n") if t]
on_disk = {}
for root, _, files in os.walk("."):
    if ".git" in root:
        continue
    for f in files:
        on_disk.setdefault(f, []).append(os.path.join(root, f))

missing = 0
for n in names:
    n2 = n.replace("\\", "/")
    base = os.path.basename(n2)
    git_hits = [t for t in tracked if t == n2 or os.path.basename(t) == base]
    if git_hits:
        print(f"  tracked   {n}  ->  {git_hits[0]}")
    elif base in on_disk:
        missing += 1
        print(f"  UNTRACKED {n}  (on disk at {on_disk[base][0]}, NOT in git)")
    else:
        missing += 1
        print(f"  MISSING   {n}  (not on disk, not in git)")
print(f"\n{len(names)} cited files, {missing} not in git.")
