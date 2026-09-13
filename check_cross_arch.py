import json
d = json.load(open('cross_arch_results.json'))
print(json.dumps(d, indent=1)[:4000])