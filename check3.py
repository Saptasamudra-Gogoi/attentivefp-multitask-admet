import json

d = json.load(open('expert_specialization_SUMMARY.json'))
print("datasets_analysed:", d['datasets_analysed'])
print()
print("datasets_collapsed:", d['datasets_collapsed'])
print()
pd = d['per_dataset']
print("Type of per_dataset:", type(pd))
print("Number of entries:", len(pd))
first_key = list(pd.keys())[0] if isinstance(pd, dict) else 0
first_entry = pd[first_key]
print(f"Sample entry ({first_key}):", first_entry)