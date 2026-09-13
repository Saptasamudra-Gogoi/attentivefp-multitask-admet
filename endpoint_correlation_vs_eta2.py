"""
endpoint_correlation_vs_eta2.py
=================================
P1-3 fix. The current explanation for why AstraZeneca Lipophilicity shows
suppressed specialization (chemical diversity) is contradicted by one of
two diversity metrics (Table S2) and reported as an unresolved open
question in the manuscript.

Better hypothesis, testable at zero training cost: when the prediction
target is itself strongly correlated with a descriptor, that descriptor's
variance is signal the network must preserve and spread evenly across
the representation to predict well -- not a nuisance axis the router is
free to partition on. For AstraZeneca Lipophilicity, the endpoint IS
LogP-like (the label literally is a lipophilicity measurement), so LogP
correlates strongly with the endpoint by construction -- exactly the
descriptor whose specialization is most suppressed in Table 5.

Prediction: across all 22 TDC datasets, higher |corr(endpoint, descriptor)|
should correlate with LOWER eta2 for that descriptor (a negative
relationship). This uses:
  - The eta2 values already in expert_specialization_SUMMARY.json /
    the per-dataset json files (routing_score-adjacent, or recomputed
    directly from routing_{ds}_seed0_v2.npy + descriptors, for consistency
    with what's already computed).
  - TDC endpoint labels (re-fetched, cheap).
  - Pearson |corr(endpoint, descriptor)| per dataset per descriptor.
  - Spearman correlation between |corr| and eta2 across all
    dataset x descriptor pairs.

Run: python endpoint_correlation_vs_eta2.py
Cost: pure script, no training. Should finish in a few minutes.
Output: endpoint_correlation_analysis.json + printed summary.
"""

import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

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

# Same 22 datasets as retrain_and_extract.py / random_partition_null.py
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

SAVE_PATH = "endpoint_correlation_analysis.json"


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0


def load_tdc_smiles_and_labels(tdc_name, cls_name):
    """Same train+val+test concatenation as retrain_and_extract.py, with
    the same validity filter, so indices align with routing .npy arrays."""
    from tdc.single_pred import ADME, Tox
    Cls = ADME if cls_name == "ADME" else Tox
    data = Cls(name=tdc_name)
    split = data.get_split(method="scaffold", seed=42)

    smiles, labels = [], []
    for part in ["train", "valid", "test"]:
        for smi, y in zip(split[part]["Drug"], split[part]["Y"]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is not None:
                    smiles.append(str(smi))
                    labels.append(float(y))
            except Exception:
                continue
    return smiles, np.array(labels)


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


def main():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            results = json.load(f)
        print(f"Resuming -- {len(results)} datasets done")
    else:
        results = {}

    for ds_key, (cls_name, tdc_name) in TDC_CONFIG.items():
        if ds_key in results:
            print(f"  Skipping {ds_key} (done)")
            continue

        npy_path = f"routing_{ds_key}_seed0_v2.npy"
        if not os.path.exists(npy_path):
            print(f"  SKIP {ds_key} -- no routing file")
            continue

        print(f"\n{'='*60}\n  {ds_key}\n{'='*60}")

        try:
            smiles, labels = load_tdc_smiles_and_labels(tdc_name, cls_name)
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        dominant = np.load(npy_path)
        if len(dominant) != len(smiles):
            print(f"  Length mismatch (routing={len(dominant)}, smiles={len(smiles)}) -- skipping")
            continue

        n_active = len(np.unique(dominant))
        if n_active < 2:
            print(f"  Routing collapsed (n_active={n_active}) -- skipping")
            continue

        desc_arrays = compute_descriptors(smiles)

        ds_result = {}
        for desc_name, vals in desc_arrays.items():
            mask = ~np.isnan(vals) & ~np.isnan(labels)
            if mask.sum() < 20:
                continue

            # |correlation| between endpoint and descriptor
            try:
                r, _ = pearsonr(labels[mask], vals[mask])
                abs_corr = abs(float(r)) if not np.isnan(r) else None
            except Exception:
                abs_corr = None

            # eta2 for this descriptor on the router's dominant-expert labels
            groups = [vals[mask][dominant[mask] == e] for e in np.unique(dominant)
                      if (dominant[mask] == e).sum() >= 3]
            eta2 = eta_squared(groups) if len(groups) >= 2 else None

            ds_result[desc_name] = {"abs_endpoint_corr": abs_corr, "eta2": eta2}
            print(f"    {desc_name:10} |corr(endpoint,desc)|={abs_corr:.3f}  eta2={eta2:.4f}"
                  if abs_corr is not None and eta2 is not None else
                  f"    {desc_name:10} incomplete")

        results[ds_key] = ds_result
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # -- Pool all dataset x descriptor pairs and test the prediction --
    all_corr, all_eta2, pairs = [], [], []
    for ds, desc_dict in results.items():
        for desc_name, vals in desc_dict.items():
            if vals.get("abs_endpoint_corr") is not None and vals.get("eta2") is not None:
                all_corr.append(vals["abs_endpoint_corr"])
                all_eta2.append(vals["eta2"])
                pairs.append((ds, desc_name))

    print(f"\n{'='*70}")
    print(f"  POOLED TEST: |corr(endpoint, descriptor)| vs eta2")
    print(f"  n = {len(all_corr)} dataset x descriptor pairs")
    print(f"{'='*70}")

    if len(all_corr) >= 10:
        rho, p = spearmanr(all_corr, all_eta2)
        print(f"  Spearman rho = {rho:.4f}, P = {p:.4f}")
        if rho < 0 and p < 0.05:
            print(f"  SUPPORTED: higher endpoint correlation predicts LOWER eta2 "
                  f"(the descriptor becomes signal, not a nuisance axis)")
        elif p >= 0.05:
            print(f"  NOT SIGNIFICANT: no clear pooled relationship")
        else:
            print(f"  OPPOSITE DIRECTION: does not support the hypothesis")

        # specific check: LogP on lipophilicity_astrazeneca
        az_logp = results.get("lipophilicity_astrazeneca", {}).get("LogP")
        if az_logp:
            print(f"\n  Specific case -- AstraZeneca Lipophilicity, LogP:")
            print(f"    |corr(endpoint, LogP)| = {az_logp['abs_endpoint_corr']:.3f}")
            print(f"    eta2(LogP) = {az_logp['eta2']:.4f}")
            others_corr = [v["abs_endpoint_corr"] for ds, d in results.items()
                           if ds != "lipophilicity_astrazeneca"
                           for name, v in d.items() if name == "LogP"
                           and v.get("abs_endpoint_corr") is not None]
            if others_corr:
                print(f"    Mean |corr(endpoint, LogP)| on other datasets: "
                      f"{np.mean(others_corr):.3f}")
    else:
        print("  Too few pairs for a meaningful pooled test")

    print(f"\nSaved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
