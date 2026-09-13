import json, os
from kmeans_control import run_control
from tdc.single_pred import ADME, Tox

DATASETS = {
    "caco2_wang":               ("ADME", "Caco2_Wang", 4),
    "solubility_aqsoldb":       ("ADME", "Solubility_AqSolDB", 8),
    "lipophilicity_astrazeneca":("ADME", "Lipophilicity_AstraZeneca", 4),
    "ld50_zhu":                 ("Tox",  "LD50_Zhu", 16),
    "bbb_martins":               ("ADME", "BBB_Martins", 16),
    "herg":                      ("Tox",  "hERG", 8),
    "ames":                      ("Tox",  "AMES", 16),
    "dili":                      ("Tox",  "DILI", 16),
}

print("Starting kmeans control batch...", flush=True)

results = []
for ds_key, (cls, tdc_name, n_experts) in DATASETS.items():
    ckpt = f"models/moegcn_{ds_key}_seed0_v2.pt"
    if not os.path.exists(ckpt):
        print(f"SKIP {ds_key} — no checkpoint at {ckpt}")
        continue
    try:
        Cls = ADME if cls == "ADME" else Tox
        data = Cls(name=tdc_name)
        smiles = list(data.get_data()["Drug"])
        r = run_control(ds_key, ckpt, smiles, n_experts)
        results.append(r)
        print(f"{ds_key}: {r['kmeans_eta2']}")
    except Exception as e:
        print(f"FAILED {ds_key}: {e}")

with open("kmeans_control_all.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results -> kmeans_control_all.json")