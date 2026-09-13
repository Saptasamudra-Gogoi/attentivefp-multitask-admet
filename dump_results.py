import json

files = [
    'results/attentivefp_20260522_1333.json',
    'results/baselines_20260521_2140.json',
    'results/baselines_20260522_0241.json',
    'results/fingerprint_baselines_20260522_1305.json',
    'results/moe_attentivefp_20260522_1511.json',
    'results/optuna_moe_20260522_1949.json',
]

for f in files:
    try:
        print(f'\n=== {f} ===')
        print(json.dumps(json.load(open(f)), indent=2))
    except Exception as e:
        print(f'Error: {e}')
