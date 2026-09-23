import os, json, csv, hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(".")
SUB = Path("attentivefp-multitask-admet")

def file_hash(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def mtime(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

report = []

for p in ROOT.glob("*.csv"):
    if not p.is_file(): continue
    dup = SUB / p.name
    if dup.exists() and dup.is_file():
        same = file_hash(p) == file_hash(dup)
        report.append(f"{p.name}: root={mtime(p)} sub={mtime(dup)} identical={same}")

for p in ROOT.glob("*.json"):
    if not p.is_file(): continue
    dup = SUB / p.name
    if dup.exists() and dup.is_file():
        same = file_hash(p) == file_hash(dup)
        report.append(f"{p.name}: root={mtime(p)} sub={mtime(dup)} identical={same}")

results_dir = ROOT / "results"
if results_dir.exists():
    for p in results_dir.glob("*"):
        if not p.is_file(): continue
        dup = SUB / "results" / p.name
        if dup.exists() and dup.is_file():
            same = file_hash(p) == file_hash(dup)
            report.append(f"results/{p.name}: root={mtime(p)} sub={mtime(dup)} identical={same}")

suffixes = ["_v2", "_fixed", "_CORRECTED", "_final", "_exact"]
all_files = [p for p in ROOT.glob("*.csv") if p.is_file()] + [p for p in ROOT.glob("*.json") if p.is_file()]
names = {p.name: p for p in all_files}
for name, p in names.items():
    for suf in suffixes:
        if suf in name:
            base = name.replace(suf, "")
            if base in names:
                same = file_hash(p) == file_hash(names[base])
                report.append(f"SUFFIX PAIR: {base} vs {name} identical={same}")

for p in ROOT.glob("*.csv"):
    if not p.is_file(): continue
    try:
        with open(p, encoding="utf-8") as f:
            header = f.readline()
        if "Γ" in header:
            report.append(f"ENCODING ISSUE in {p.name}: {header.strip()}")
    except Exception as e:
        report.append(f"COULD NOT READ {p.name}: {e}")

with open("PROVENANCE_AUDIT.md", "w", encoding="utf-8") as f:
    f.write("# Provenance Audit\n\n")
    for line in report:
        f.write(f"- {line}\n")

print(f"Done. {len(report)} findings written to PROVENANCE_AUDIT.md")