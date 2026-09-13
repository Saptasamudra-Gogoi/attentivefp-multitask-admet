"""
Locate the manuscript source (docx/md/tex) and print any lines mentioning
AttentiveFP or GROVER near "Table 1", to check if std values were dropped
or never existed (review item 2.7).
"""
import glob
import os

exts = ["*.md", "*.tex", "*.txt"]
candidates = []
for ext in exts:
    candidates += glob.glob(os.path.join("**", ext), recursive=True)

print(f"Found {len(candidates)} candidate text files:")
for c in candidates:
    print(" ", c)

print("\n--- Searching for AttentiveFP / GROVER lines ---")
for c in candidates:
    try:
        with open(c, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        continue
    for i, line in enumerate(lines):
        if "attentivefp" in line.lower() or "grover" in line.lower():
            print(f"{c}:{i+1}: {line.strip()}")

print("\nAlso check for .docx manuscript files (can't grep those here):")
for f in glob.glob(os.path.join("**", "*.docx"), recursive=True):
    print(" ", f)
