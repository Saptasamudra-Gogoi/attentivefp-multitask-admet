"""
random_partition_null.py
=========================
P0-5a fix. Your own recommendations section tells the community to run
"a clustering OR random-partition null" -- you only ran the clustering
null (kmeans_control_all.json). This script runs the missing
random-partition null.

For each dataset (all 22 TDC, using the real routing_{ds}_seed{s}_v2.npy
arrays already on disk from retrain_and_extract.py):
  1. Take the REAL dominant-expert group sizes for that dataset/seed.
  2. Shuffle molecule-to-expert assignment 1000 times, preserving those
     exact group sizes (a random partition with the same size profile).
  3. Recompute eta2 for all 8 descriptors on each shuffle.
  4. Report the 95th percentile of the shuffled eta2 distribution as the
     null ceiling for that dataset x descriptor.

Any real (router) eta2 reported in Table 5 / Fig 3A that sits ABOVE this
ceiling is distinguishable from a same-size-groups random partition.
Any that sits AT OR BELOW it is not distinguishable from noise, however
small the ANOVA p-value -- p-values shrink with n regardless of effect
size, but the null ceiling does not.

Requires the SAME TDC scaffold split (seed=42) used by
retrain_and_extract.py, so molecule order matches the saved .npy arrays.

Run: python random_partition_null.py
Cost: pure script, no training. Should finish in well under an hour.
Output: random_partition_null.json
"""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

N_SHUFFLES = 1000
SEEDS = [0, 1, 2, 3, 4]
SAVE_PATH = "random_partition_null.json"

# -- Same 22 datasets / TDC names as retrain_and_extract.py's BEST_PARAMS --
TDC_CONFIG = {
    "solubility_aqsoldb":               ("ADME", "Solubility_AqSolDB"),
    "caco2_wang":                       ("ADME", "Caco2_Wang"),
    "lipophilicity_astrazeneca":        ("ADME", "Lipophilicity_AstraZeneca"),
    "ppbr_az":                          ("ADME", "PPBR_AZ"),
    "ld50_zhu":                         ("Tox",  "LD50_Zhu"),
    "vdss_lombardo":                    ("ADME", "VDss_Lombardo"),
    "half_life_obach":                  ("ADME", "Half_Life_Obach"),
    "clearance_microsome_az":           ("ADME", "Clearance_Microsome_AZ"),
    "clearance_hepatocyte_az":          ("ADME", "Clearance_Hepatocyte_AZ"),
    "hia_hou":                          ("ADME", "HIA_Hou"),
    "pgp_broccatelli":                  ("ADME", "Pgp_Broccatelli"),
    "bioavailability_ma":               ("ADME", "Bioavailability_Ma"),
    "bbb_martins":                      ("ADME", "BBB_Martins"),
    "cyp2d6_veith":                     ("ADME", "CYP2D6_Veith"),
    "cyp3a4_veith":                     ("ADME", "CYP3A4_Veith"),
    "cyp2c9_veith":                     ("ADME", "CYP2C9_Veith"),
    "cyp2d6_substrate_carbonmangels":   ("ADME", "CYP2D6_Substrate_CarbonMangels"),
    "cyp3a4_substrate_carbonmangels":   ("ADME", "CYP3A4_Substrate_CarbonMangels"),
    "cyp2c9_substrate_carbonmangels":   ("ADME", "CYP2C9_Substrate_CarbonMangels"),
    "herg":                             ("Tox",  "hERG"),
    "ames":                             ("Tox",  "AMES"),
    "dili":                             ("Tox",  "DILI"),
}

DESCRIPTOR_FNS = {
    "MW":        Descriptors.ExactMolWt,
    "LogP":      Descriptors.MolLogP,
    "HBA":       rdMolDescriptors.CalcNumHBA,
    "HBD":       rdMolDescriptors.CalcNumHBD,
    "TPSA":      Descriptors.TPSA,
    "RotBonds":  rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "ArRings":   rdMolDescriptors.CalcNumAromaticRings,
}


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0


def load_tdc_smiles(tdc_name, cls_name):
    """Reproduce the exact train+val+test SMILES order used by
    retrain_and_extract.py's build_dataset(), including its SMILES
    validity filter, so indices line up with the saved routing .npy."""
    from tdc.single_pred import ADME, Tox
    Cls = ADME if cls_name == "ADME" else Tox
    data = Cls(name=tdc_name)
    split = data.get_split(method="scaffold", seed=42)

    def valid_smiles(smiles_list):
        out = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is not None:
                    out.append(str(smi))
            except Exception:
                continue
        return out

    train_smi = valid_smiles(split["train"]["Drug"])
    val_smi   = valid_smiles(split["valid"]["Drug"])
    test_smi  = valid_smiles(split["test"]["Drug"])
    return train_smi + val_smi + test_smi


def compute_descriptors(smiles_list):
    desc_vals = {k: [] for k in DESCRIPTOR_FNS}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        for name, fn in DESCRIPTOR_FNS.items():
            try:
                desc_vals[name].append(float(fn(mol)))
            except Exception:
                desc_vals[name].append(np.nan)
    return {k: np.array(v) for k, v in desc_vals.items()}


def random_partition_ceiling(dominant, desc_arrays, n_shuffles, rng):
    """Shuffle dominant-expert labels n_shuffles times, preserving group
    sizes exactly (a permutation of the label array), recompute eta2 per
    descriptor each time, return the 95th percentile per descriptor."""
    n = len(dominant)
    null_dists = {k: [] for k in desc_arrays}

    for _ in range(n_shuffles):
        shuffled = rng.permutation(dominant)
        for name, vals in desc_arrays.items():
            mask = ~np.isnan(vals)
            groups = [vals[mask][shuffled[mask] == e]
                      for e in np.unique(shuffled)
                      if (shuffled[mask] == e).sum() >= 3]
            if len(groups) < 2:
                null_dists[name].append(0.0)
                continue
            null_dists[name].append(eta_squared(groups))

    return {name: {
        "null_p95": float(np.percentile(vals, 95)),
        "null_mean": float(np.mean(vals)),
        "null_max": float(np.max(vals)),
    } for name, vals in null_dists.items()}


def real_eta2(dominant, desc_arrays):
    result = {}
    for name, vals in desc_arrays.items():
        mask = ~np.isnan(vals)
        groups = [vals[mask][dominant[mask] == e]
                  for e in np.unique(dominant)
                  if (dominant[mask] == e).sum() >= 3]
        result[name] = eta_squared(groups) if len(groups) >= 2 else None
    return result


def main():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            results = json.load(f)
        print(f"Resuming -- {len(results)} datasets done")
    else:
        results = {}

    rng = np.random.default_rng(42)

    for ds_key, (cls_name, tdc_name) in TDC_CONFIG.items():
        if ds_key in results:
            print(f"  Skipping {ds_key} (already done)")
            continue

        npy_path = f"routing_{ds_key}_seed0_v2.npy"
        if not os.path.exists(npy_path):
            print(f"  SKIP {ds_key} -- no routing file at {npy_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {ds_key}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            smiles = load_tdc_smiles(tdc_name, cls_name)
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        desc_arrays = compute_descriptors(smiles)

        seed_results = {}
        for seed in SEEDS:
            npy_path = f"routing_{ds_key}_seed{seed}_v2.npy"
            if not os.path.exists(npy_path):
                continue
            dominant = np.load(npy_path)

            if len(dominant) != len(smiles):
                print(f"    seed {seed}: length mismatch "
                      f"(routing={len(dominant)}, smiles={len(smiles)}) -- skipping")
                continue

            n_active = len(np.unique(dominant))
            if n_active < 2:
                print(f"    seed {seed}: routing collapsed (n_active={n_active}) -- skipping")
                continue

            real = real_eta2(dominant, desc_arrays)
            null_ceiling = random_partition_ceiling(dominant, desc_arrays, N_SHUFFLES, rng)

            above_null = {name: (real[name] is not None and real[name] > null_ceiling[name]["null_p95"])
                          for name in desc_arrays}

            seed_results[str(seed)] = {
                "n_active_experts": n_active,
                "real_eta2": real,
                "null_ceiling": null_ceiling,
                "above_null_p95": above_null,
            }
            print(f"    seed {seed}: n_active={n_active}  "
                  f"above_null={sum(above_null.values())}/8 descriptors")

        if seed_results:
            results[ds_key] = seed_results
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

    print(f"\nSaved -> {SAVE_PATH}")

    # -- Summary: how many dataset/seed/descriptor cells clear the null --
    total, above = 0, 0
    for ds, seeds in results.items():
        for seed, r in seeds.items():
            for name, flag in r["above_null_p95"].items():
                total += 1
                if flag:
                    above += 1
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {above}/{total} dataset x seed x descriptor cells "
          f"clear the random-partition 95th-percentile null")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
