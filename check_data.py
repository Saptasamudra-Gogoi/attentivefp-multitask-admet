import json
import pprint

d = json.load(open('expert_specialization_solubility_aqsoldb.json'))
print('dataset:', d['dataset'])
print('stats keys:', list(d['stats'].keys()))
print()
pprint.pprint(d['stats'])
