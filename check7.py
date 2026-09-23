import json
d = json.load(open('expert_specialization_SUMMARY.json'))
for name, entry in d['per_dataset'].items():
    print(name, "n_molecules:", entry.get('n_molecules'))