import json
plain = json.load(open('results_dmpnn_regr.json'))
moe = json.load(open('results_moedmpnn_regr.json'))
print("PLAIN:", json.dumps(plain, indent=1)[:1500])
print("\nMOE:", json.dumps(moe, indent=1)[:1500])