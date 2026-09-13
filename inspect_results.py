import json

files = [
    "results_moegcn_regr.json",
    "results_moegcn_classif.json",
    "results_gcn_tdc.json",
    "results_moegcn_tdc_v2.json",
    "expanded_descriptor_results.json",
    "expert_specialization_SUMMARY.json",
    "statistical_significance_results.json",
    "results_moedmpnn_regr.json",
]

for f in files:
    try:
        with open(f) as fp:
            d = json.load(fp)
        keys = list(d.keys())
        print(f"\n=== {f} ===")
        print("Keys:", keys[:15])
        first = keys[0]
        print(f"Sample [{first}]:", d[first])
    except Exception as e:
        print(f"\n=== {f} === ERROR: {e}")