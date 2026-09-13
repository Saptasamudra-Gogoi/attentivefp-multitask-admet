"""
Find which SSS formula/run produced Spearman r=0.019, P=0.934, so the
manuscript's abstract sentence can be corrected to describe the real test.

Run from D:\\molprop_project\\ with moe_admet activated.
"""
import json
import glob

candidates = glob.glob("sss_*.json") + glob.glob("kmeans_control_*.json")
print(f"Checking {len(candidates)} JSON files for r near 0.019 or 0.0187...\n")

for f in candidates:
    try:
        with open(f) as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"[skip] {f}: {e}")
        continue

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, (int, float)) and isinstance(v, float):
                    if 0.015 <= abs(v) <= 0.022:
                        print(f"{f} :: {new_path} = {v}")
                walk(v, new_path)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(data)

print("\nAlso printing full top-level structure of each file for context:\n")
for f in candidates:
    try:
        with open(f) as fh:
            data = json.load(fh)
    except Exception:
        continue
    print(f"--- {f} ---")
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        # print best_r / best_p / spearman fields directly if present
        for key in ["best_r", "best_p", "best_predictor", "spearman_r", "spearman_p",
                    "spearman_refined_sss", "p_refined_sss", "all_predictors"]:
            if key in data:
                print(f"  {key}: {data[key]}")
    print()
