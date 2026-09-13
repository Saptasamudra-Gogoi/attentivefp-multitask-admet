"""
Check every results_*.json / expert_specialization_*.json that exists both
at project root and inside attentivefp-multitask-admet/, and flag any pair
whose values differ (root = likely stale duplicate, per the 2.8 finding).
"""
import json
import glob
import os

root_files = glob.glob("*.json")
sub_files = glob.glob(os.path.join("attentivefp-multitask-admet", "*.json"))
sub_names = {os.path.basename(f) for f in sub_files}

print("Checking for stale root-level duplicates...\n")
mismatches = []
for rf in root_files:
    name = os.path.basename(rf)
    if name in sub_names:
        sf = os.path.join("attentivefp-multitask-admet", name)
        try:
            with open(rf, encoding="utf-8") as f:
                root_data = json.load(f)
            with open(sf, encoding="utf-8") as f:
                sub_data = json.load(f)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

        if root_data == sub_data:
            print(f"[OK]      {name} -- identical")
        else:
            print(f"[DIFFER]  {name} -- root and subfolder copies do NOT match")
            mismatches.append(name)

print(f"\n{len(mismatches)} mismatched file(s) found:")
for m in mismatches:
    print(" ", m)
print("\nFor each mismatch: subfolder copy (attentivefp-multitask-admet/) is the")
print("more recently-modified/corrected one per the 2.8 finding -- verify with")
print("`dir` timestamps, then delete or archive the stale root copy.")
