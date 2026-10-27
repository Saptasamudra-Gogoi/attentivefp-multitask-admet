import json

d = json.load(open('expert_specialization_SUMMARY.json'))
print("Type:", type(d))
if isinstance(d, dict):
    print("Keys:", list(d.keys()))
elif isinstance(d, list):
    print("Length:", len(d))
    print("Sample:", d[0])