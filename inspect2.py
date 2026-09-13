import json

with open("expanded_descriptor_results.json") as f:
    d = json.load(f)

print(type(d))
if isinstance(d, list):
    print("Length:", len(d))
    print("First item:", d[0])
elif isinstance(d, dict):
    print("Keys:", list(d.keys())[:5])