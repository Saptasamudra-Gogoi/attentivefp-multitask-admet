import json

files = [
    "results_moegcn_regr.json",
    "results_moegcn_classif.json",
    "results_gcn_tdc.json",
    "results_moegcn_tdc_v2.json",
]

for f in files:
    with open(f) as fp:
        d = json.load(fp)
    print(f"\n=== {f} ===")
    print("Top-level keys:", list(d.keys())[:10])
    # show first entry
    first_key = list(d.keys())[0]
    print(f"Sample [{first_key}]:", d[first_key])