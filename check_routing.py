import json
import pprint

print('=== ablation_routing_results.json ===')
d = json.load(open('ablation_routing_results.json'))
print('Keys:', list(d.keys())[:10])
first = list(d.keys())[0]
print('First key:', first)
pprint.pprint(d[first])

print()
print('=== results_phase3_fixed.json ===')
d2 = json.load(open('results_phase3_fixed.json'))
print('Keys:', list(d2.keys())[:10])
first2 = list(d2.keys())[0]
print('First key:', first2)
pprint.pprint(d2[first2])
