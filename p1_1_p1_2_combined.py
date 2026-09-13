"""
p1_1_p1_2_combined.py
=======================
P1-1: Benjamini-Hochberg correction across Table 5's 32 ANOVA/Kruskal-
      Wallis tests (4 datasets x 8 descriptors).
P1-2: Bootstrap 95% CIs (2000 resamples) and omega-squared (less biased
      than eta2 for small groups) for every Table 5 cell.

Recomputes directly from routing_{ds}_seed0_v2.npy + RDKit descriptors,
same as random_partition_null.py / endpoint_correlation_vs_eta2.py, so
everything is self-consistent with what's already in the manuscript.

Run: python p1_1_p1_2_combined.py
Output: p1_stats_results.json + printed summary tables
"""

import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
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

# The 4 datasets in Table 5 (main specialization analysis)
TABLE5_DATASETS = {
    "solubility_aqsoldb":        ("ADME", "Solubility_AqSolDB"),
    "caco2_wang":                ("ADME", "Caco2_Wang"),
    "ld50_zhu":                  ("Tox",  "LD50_Zhu"),
    "lipophilicity_astrazeneca": ("ADME", "Lipophilicity_AstraZeneca"),
}

N_BOOTSTRAP = 2000
SAVE_PATH = "p1_stats_results.json"


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0


def omega_squared(groups):
    """Less biased than eta2, corrects for number of groups (k) and
    total n: omega2 = (SSB - (k-1)*MSW) / (SST + MSW)"""
    k = len(groups)
    all_v = np.concatenate(groups)
    n = len(all_v)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    ssw = sst - ssb
    df_w = n - k
    if df_w <= 0:
        return None
    msw = ssw / df_w
    denom = sst + msw
    if denom <= 0:
        return None
    return float((ssb - (k - 1) * msw) / denom)


def bootstrap_eta2_ci(dominant, vals, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_etas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_b, v_b = dominant[idx], vals[idx]
        groups = [v_b[d_b == e] for e in np.unique(d_b) if (d_b == e).sum() >= 3]
        if len(groups) >= 2:
            boot_etas.append(eta_squared(groups))
    if len(boot_etas) < 100:
        return None, None
    lo, hi = np.percentile(boot_etas, [2.5, 97.5])
    return float(lo), float(hi)


def load_tdc_smiles(tdc_name, cls_name):
    from tdc.single_pred import ADME, Tox
    Cls = ADME if cls_name == "ADME" else Tox
    data = Cls(name=tdc_name)
    split = data.get_split(method="scaffold", seed=42)
    smiles = []
    for part in ["train", "valid", "test"]:
        for smi in split[part]["Drug"]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                smiles.append(str(smi))
    return smiles


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
    all_results = {}
    all_pvalues_anova, all_pvalues_kw = [], []
    cell_keys = []

    for ds_key, (cls_name, tdc_name) in TABLE5_DATASETS.items():
        npy_path = f"routing_{ds_key}_seed0_v2.npy"
        if not os.path.exists(npy_path):
            print(f"SKIP {ds_key} -- no routing file")
            continue

        print(f"\n{'='*60}\n  {ds_key}\n{'='*60}")
        smiles = load_tdc_smiles(tdc_name, cls_name)
        dominant = np.load(npy_path)

        if len(dominant) != len(smiles):
            print(f"  Length mismatch -- skipping")
            continue

        desc_arrays = compute_descriptors(smiles)
        all_results[ds_key] = {}

        for desc_name, vals in desc_arrays.items():
            mask = ~np.isnan(vals)
            groups = [vals[mask][dominant[mask] == e] for e in np.unique(dominant)
                      if (dominant[mask] == e).sum() >= 3]
            if len(groups) < 2:
                continue

            eta2 = eta_squared(groups)
            omega2 = omega_squared(groups)
            f_stat, p_anova = f_oneway(*groups)
            h_stat, p_kw = kruskal(*groups)
            ci_lo, ci_hi = bootstrap_eta2_ci(dominant[mask], vals[mask])

            all_results[ds_key][desc_name] = {
                "eta2": round(eta2, 4),
                "omega2": round(omega2, 4) if omega2 is not None else None,
                "eta2_ci_95": [round(ci_lo, 4), round(ci_hi, 4)] if ci_lo is not None else None,
                "p_anova_raw": float(p_anova),
                "p_kruskal_raw": float(p_kw),
            }
            all_pvalues_anova.append(float(p_anova))
            all_pvalues_kw.append(float(p_kw))
            cell_keys.append((ds_key, desc_name))

            print(f"  {desc_name:10} eta2={eta2:.4f}  omega2={omega2:.4f}  "
                  f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]  p_anova={p_anova:.2e}")

    # -- P1-1: Benjamini-Hochberg correction --
    print(f"\n{'='*70}")
    print(f"  P1-1: Benjamini-Hochberg correction, n={len(all_pvalues_anova)} tests")
    print(f"{'='*70}")

    _, p_anova_bh, _, _ = multipletests(all_pvalues_anova, alpha=0.05, method='fdr_bh')
    _, p_kw_bh, _, _ = multipletests(all_pvalues_kw, alpha=0.05, method='fdr_bh')

    n_sig_before = sum(p < 0.05 for p in all_pvalues_anova)
    n_sig_after = sum(p < 0.05 for p in p_anova_bh)
    print(f"  ANOVA: {n_sig_before}/{len(all_pvalues_anova)} significant before BH, "
          f"{n_sig_after}/{len(all_pvalues_anova)} after")
    print(f"  Max raw p-value: {max(all_pvalues_anova):.2e}")
    print(f"  Max BH-adjusted p-value: {max(p_anova_bh):.2e}")

    for i, (ds, desc) in enumerate(cell_keys):
        all_results[ds][desc]["p_anova_bh"] = float(p_anova_bh[i])
        all_results[ds][desc]["p_kruskal_bh"] = float(p_kw_bh[i])

    with open(SAVE_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("  SUMMARY TABLE (Table 5 companion: eta2, 95% CI, omega2, BH-adjusted p)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<28} {'Descriptor':<10} {'eta2':>8} {'95%CI':>18} {'omega2':>8} {'p_BH':>10}")
    for ds, desc_dict in all_results.items():
        for desc, v in desc_dict.items():
            ci = f"[{v['eta2_ci_95'][0]:.3f},{v['eta2_ci_95'][1]:.3f}]" if v['eta2_ci_95'] else "N/A"
            print(f"  {ds:<28} {desc:<10} {v['eta2']:>8.4f} {ci:>18} "
                  f"{v['omega2']:>8.4f} {v['p_anova_bh']:>10.2e}")

    print(f"\nSaved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
