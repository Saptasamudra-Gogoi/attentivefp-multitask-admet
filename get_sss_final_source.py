"""
Print the full content of sss_results_final.json (the file matching the
manuscript's r=0.019, P=0.934) and find which script produced it.

Run from D:\\molprop_project\\ with moe_admet activated.
"""
import json
import glob

print("=" * 70)
print("FULL CONTENT: sss_results_final.json")
print("=" * 70)
with open("sss_results_final.json") as f:
    data = json.load(f)
print(json.dumps(data, indent=2))

print("\n" + "=" * 70)
print("Searching all .py files for the string 'sss_results_final'")
print("=" * 70)
for f in glob.glob("*.py") + glob.glob("**/*.py", recursive=True):
    try:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except Exception:
        continue
    if "sss_results_final" in content:
        print(f"\nFOUND in: {f}")
