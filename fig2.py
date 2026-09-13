import json
with open("expert_specialization_SUMMARY.json") as f:
    d = json.load(f)
for ds in d["per_dataset"]:
    print(ds, list(d["per_dataset"][ds]["stats"].keys()))