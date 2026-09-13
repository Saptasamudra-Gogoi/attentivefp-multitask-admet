import json
import pprint

d = json.load(open('ablation_routing_results.json'))
print('All datasets:', list(d.keys()))
print()
for ds in d:
    print(f'=== {ds} ===')
    pprint.pprint(d[ds])
    print()
