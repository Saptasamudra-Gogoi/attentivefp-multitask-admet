"""
extract_smiles_and_specialize.py
=================================
Combines routing extraction + specialization in one script.
Loads TDC data, gets SMILES, loads routing .npy files,
computes eta-squared for all 22 datasets, outputs exact SSS.

Place in D:\molprop_project\ and run:
    python extract_smiles_and_specialize.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import f_oneway, kruskal, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Imports OK | Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# TDC DATASET NAMES
# ══════════════════════════════════════════════════════════════════════════

TDC_NAMES = {
    "solubility_aqsoldb":             "Solubility_AqSolDB",
    "caco2_wang":                     "Caco2_Wang",
    "lipophilicity_astrazeneca":      "Lipophilicity_AstraZeneca",
    "ld50_zhu":                       "LD50_Zhu",
    "ppbr_az":                        "PPBR_AZ",
    "hia_hou":                        "HIA_Hou",
    "pgp_broccatelli":                "Pgp_Broccatelli",
    "bioavailability_ma":             "Bioavailability_Ma",
    "bbb_martins":                    "BBB_Martins",
    "cyp2d6_veith":                   "CYP2D6_Veith",
    "cyp3a4_veith":                   "CYP3A4_Veith",
    "cyp2c9_veith":                   "CYP2C9_Veith",
    "cyp2d6_substrate_carbonmangels": "CYP2D6_Substrate_CarbonMangels",
    "cyp3a4_substrate_carbonmangels": "CYP3A4_Substrate_CarbonMangels",
    "cyp2c9_substrate_carbonmangels": "CYP2C9_Substrate_CarbonMangels",
    "herg":                           "hERG",
    "ames":                           "AMES",
    "dili":                           "DILI",
    "vdss_lombardo":                  "VDss_Lombardo",
    "half_life_obach":                "Half_Life_Obach",
    "clearance_microsome_az":         "Clearance_Microsome_AZ",
    "clearance_hepatocyte_az":        "Clearance_Hepatocyte_AZ",
}

DISPLAY_NAMES = {
    "solubility_aqsoldb":             "Solubility",
    "caco2_wang":                     "Caco-2",
    "lipophilicity_astrazeneca":      "Lipophilicity",
    "ppbr_az":                        "PPBR",
    "ld50_zhu":                       "LD50",
    "hia_hou":                        "HIA",
    "pgp_broccatelli":                "PGP",
    "bioavailability_ma":             "Bioavailability",
    "bbb_martins":                    "BBB",
    "cyp2d6_veith":                   "CYP2D6 Inh",
    "cyp3a4_veith":                   "CYP3A4 Inh",
    "cyp2c9_veith":                   "CYP2C9 Inh",
    "cyp2d6_substrate_carbonmangels": "CYP2D6 Sub",
    "cyp3a4_substrate_carbonmangels": "CYP3A4 Sub",
    "cyp2c9_substrate_carbonmangels": "CYP2C9 Sub",
    "herg":                           "hERG",
    "ames":                           "AMES",
    "dili":                           "DILI",
    "vdss_lombardo":                  "VDss",
    "half_life_obach":                "Half-Life",
    "clearance_microsome_az":         "CL-Microsome",
    "clearance_hepatocyte_az":        "CL-Hepatocyte",
}

ENDPOINT_CLASSES = {
    "solubility_aqsoldb":             "Physicochemical",
    "caco2_wang":                     "Physicochemical",
    "lipophilicity_astrazeneca":      "Physicochemical",
    "ppbr_az":                        "Physicochemical",
    "ld50_zhu":                       "Toxicity",
    "hia_hou":                        "Absorption",
    "pgp_broccatelli":                "Absorption",
    "bioavailability_ma":             "Absorption",
    "bbb_martins":                    "Absorption",
    "vdss_lombardo":                  "Distribution",
    "half_life_obach":                "Metabolic",
    "clearance_microsome_az":         "Metabolic",
    "clearance_hepatocyte_az":        "Metabolic",
    "cyp2d6_veith":                   "CYP Inhibition",
    "cyp3a4_veith":                   "CYP Inhibition",
    "cyp2c9_veith":                   "CYP Inhibition",
    "cyp2d6_substrate_carbonmangels": "Metabolic",
    "cyp3a4_substrate_carbonmangels": "Metabolic",
    "cyp2c9_substrate_carbonmangels": "Metabolic",
    "herg":                           "Toxicity",
    "ames":                           "Toxicity",
    "dili":                           "Toxicity",
}

CAUSAL_DESCRIPTORS = {
    "Physicochemical": ["LogP", "MW", "TPSA", "HBA", "HBD"],
    "Absorption":      ["LogP", "MW", "TPSA", "HBD"],
    "Distribution":    ["LogP", "MW", "TPSA"],
    "Metabolic":       ["LogP", "MW", "RingCount", "ArRings"],
    "CYP Inhibition":  ["LogP", "ArRings", "MW"],
    "Toxicity":        ["LogP", "MW", "TPSA", "ArRings"],
}

CLASS_COLORS = {
    "Physicochemical": "#2196F3",
    "Absorption":      "#4CAF50",
    "Distribution":    "#9C27B0",
    "Metabolic":       "#F44336",
    "CYP Inhibition":  "#FF9800",
    "Toxicity":        "#795548",
}

MOE_GAINS = {
    "caco2_wang":                     20.521,
    "lipophilicity_astrazeneca":      8.747,
    "solubility_aqsoldb":             8.760,
    "ppbr_az":                        1.295,
    "vdss_lombardo":                  6.310,
    "half_life_obach":               -36.701,
    "clearance_microsome_az":         12.900,
    "clearance_hepatocyte_az":       -11.407,
    "hia_hou":                        1.054,
    "pgp_broccatelli":                0.463,
    "bioavailability_ma":             3.705,
    "bbb_martins":                   -0.490,
    "cyp2d6_veith":                  -2.404,
    "cyp3a4_veith":                   1.830,
    "cyp2c9_veith":                   0.211,
    "cyp2d6_substrate_carbonmangels": -2.999,
    "cyp3a4_substrate_carbonmangels": -5.418,
    "cyp2c9_substrate_carbonmangels": -0.231,
    "herg":                          -7.368,
    "ames":                          -0.357,
    "dili":                          -5.203,
    "ld50_zhu":                       4.782,
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

# ══════════════════════════════════════════════════════════════════════════
# LOAD TDC SMILES
# ══════════════════════════════════════════════════════════════════════════

def load_smiles_from_tdc(tdc_name):
    """Load all SMILES for a dataset from TDC."""
    loaders = []

    try:
        from tdc.single_pred import ADMET
        loaders.append(('ADMET', ADMET))
    except: pass
    try:
        from tdc.single_pred import ADME
        loaders.append(('ADME', ADME))
    except: pass
    try:
        from tdc.single_pred import Tox
        loaders.append(('Tox', Tox))
    except: pass

    for loader_name, LoaderClass in loaders:
        try:
            data = LoaderClass(name=tdc_name)
            split = data.get_split(method="scaffold", seed=42)
            all_smiles = (
                list(split["train"]["Drug"]) +
                list(split["valid"]["Drug"]) +
                list(split["test"]["Drug"])
            )
            print(f"    Loaded {len(all_smiles)} SMILES via {loader_name}")
            return all_smiles
        except Exception as e:
            continue

    # Fallback — try get_data() directly
    for loader_name, LoaderClass in loaders:
        try:
            data = LoaderClass(name=tdc_name)
            df = data.get_data()
            smiles_col = next(
                (c for c in ['Drug','smiles','SMILES'] if c in df.columns),
                df.columns[0]
            )
            smiles = df[smiles_col].tolist()
            print(f"    Loaded {len(smiles)} SMILES via {loader_name}.get_data()")
            return smiles
        except Exception as e:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════
# COMPUTE DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════

def compute_descriptors(smiles_list):
    desc_vals = defaultdict(list)
    valid_mask = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                valid_mask.append(False)
                for k in DESCRIPTOR_FNS:
                    desc_vals[k].append(np.nan)
                continue
            valid_mask.append(True)
            for name, fn in DESCRIPTOR_FNS.items():
                try:
                    desc_vals[name].append(float(fn(mol)))
                except:
                    desc_vals[name].append(np.nan)
        except:
            valid_mask.append(False)
            for k in DESCRIPTOR_FNS:
                desc_vals[k].append(np.nan)
    return {k: np.array(v) for k, v in desc_vals.items()}, \
           np.array(valid_mask)


# ══════════════════════════════════════════════════════════════════════════
# COMPUTE ETA-SQUARED
# ══════════════════════════════════════════════════════════════════════════

def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean)**2
        for g in groups if len(g) > 0
    )
    ss_total = np.sum((all_vals - grand_mean)**2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def mi_score(x, y):
    mask = ~np.isnan(x)
    x, y = x[mask], y[mask]
    n_bins = min(10, max(2, len(np.unique(x))))
    try:
        kbd = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile")
        x_b = kbd.fit_transform(x.reshape(-1,1)).ravel().astype(int)
        return float(mutual_info_score(x_b, y))
    except:
        return 0.0


def run_specialization(smiles_list, routing_assignments):
    """Compute eta-squared for all descriptors."""
    desc_arrays, valid_mask = compute_descriptors(smiles_list)

    # Align routing to valid molecules
    n = min(len(smiles_list), len(routing_assignments))
    valid_mask = valid_mask[:n]
    assignments = routing_assignments[:n][valid_mask]

    expert_ids = np.unique(assignments)
    if len(expert_ids) < 2:
        print(f"    Routing collapsed to {len(expert_ids)} expert(s)")
        return {}, len(assignments)

    eta2_dict = {}
    p_dict    = {}

    for desc_name, vals in desc_arrays.items():
        vals = vals[:n][valid_mask]
        not_nan = ~np.isnan(vals)
        if not_nan.sum() < 20:
            continue

        groups = [
            vals[(assignments == eid) & not_nan]
            for eid in expert_ids
            if ((assignments == eid) & not_nan).sum() >= 3
        ]
        if len(groups) < 2:
            continue

        try:
            _, p_val = f_oneway(*groups)
            eta2 = eta_squared(groups)
            mi   = mi_score(vals, assignments)
            eta2_dict[desc_name] = {
                "eta2": round(float(eta2), 6),
                "p_anova": float(p_val),
                "MI": float(mi),
            }
            p_dict[desc_name] = p_val
        except:
            continue

    return eta2_dict, len(assignments)


def compute_causal_sss(eta2_dict, endpoint_class):
    causal = CAUSAL_DESCRIPTORS.get(
        endpoint_class, list(DESCRIPTOR_FNS.keys()))
    vals = [
        eta2_dict[d]["eta2"]
        for d in causal
        if d in eta2_dict
    ]
    return float(np.mean(vals)) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Full Specialization + SSS — All 22 Datasets")
    print("=" * 65)

    all_eta2    = {}
    all_sss     = {}
    failed      = []

    # ── Step 1: Load existing JSONs ────────────────────────────────────
    print("\n[1] Loading existing specialization JSONs...")
    existing_json_datasets = []
    for dataset in TDC_NAMES:
        spec_file = Path(f"expert_specialization_{dataset}.json")
        if spec_file.exists():
            with open(spec_file) as f:
                data = json.load(f)
            stats = data.get("stats", {})
            if not stats:
                continue
            eta2_dict = {
                desc: {
                    "eta2":    ddata.get("eta2", 0.0),
                    "p_anova": ddata.get("p_anova", 1.0),
                    "MI":      ddata.get("MI", 0.0),
                }
                for desc, ddata in stats.items()
            }
            all_eta2[dataset] = eta2_dict
            existing_json_datasets.append(dataset)
            print(f"  Loaded: {dataset}")

    # ── Step 2: Process remaining datasets ────────────────────────────
    print(f"\n[2] Processing {22 - len(existing_json_datasets)} "
          f"remaining datasets...")

    for dataset in TDC_NAMES:
        if dataset in all_eta2:
            continue

        tdc_name = TDC_NAMES[dataset]
        display  = DISPLAY_NAMES[dataset]
        print(f"\n  [{display}]")

        # Load routing assignments
        routing_file = Path(f"routing_{dataset}.npy")
        if not routing_file.exists():
            print(f"    No routing file — skipping")
            failed.append(dataset)
            continue

        routing = np.load(routing_file)
        print(f"    Routing: {len(routing)} assignments, "
              f"{len(np.unique(routing))} unique experts")

        # Load SMILES from TDC
        print(f"    Loading SMILES from TDC ({tdc_name})...")
        smiles = load_smiles_from_tdc(tdc_name)
        if smiles is None:
            print(f"    TDC load failed — skipping")
            failed.append(dataset)
            continue

        # Compute eta-squared
        eta2_dict, n_valid = run_specialization(smiles, routing)
        if not eta2_dict:
            print(f"    No specialization signal (routing collapsed)")
            all_eta2[dataset] = {}
            continue

        all_eta2[dataset] = eta2_dict
        best_desc = max(eta2_dict, key=lambda d: eta2_dict[d]["eta2"])
        print(f"    Best eta2: {eta2_dict[best_desc]['eta2']:.4f} "
              f"({best_desc})")

        # Save individual JSON for future runs
        save_data = {
            "dataset": dataset,
            "n_molecules": n_valid,
            "stats": {
                desc: {
                    "eta2":    v["eta2"],
                    "p_anova": v["p_anova"],
                    "MI":      v["MI"],
                }
                for desc, v in eta2_dict.items()
            },
            "significant_descriptors": [
                d for d, v in eta2_dict.items()
                if v["p_anova"] < 0.05
            ],
        }
        with open(f"expert_specialization_{dataset}.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"    Saved expert_specialization_{dataset}.json")

    # ── Step 3: Compute SSS ────────────────────────────────────────────
    print("\n[3] Computing Causal SSS...")
    print(f"\n  {'Dataset':<32} {'Causal SSS':>11} "
          f"{'Best eta2':>10} {'Best desc':<12} {'MoE Gain':>10} Class")
    print("  " + "-"*85)

    for dataset in TDC_NAMES:
        ep_class = ENDPOINT_CLASSES.get(dataset, "Physicochemical")
        display  = DISPLAY_NAMES.get(dataset, dataset)
        gain     = MOE_GAINS.get(dataset)
        eta2_dict = all_eta2.get(dataset, {})

        if eta2_dict:
            causal_sss = compute_causal_sss(eta2_dict, ep_class)
            best_desc  = max(eta2_dict,
                             key=lambda d: eta2_dict[d]["eta2"])
            best_eta2  = eta2_dict[best_desc]["eta2"]
            source     = "exact"
        else:
            # Routing collapsed — SSS = 0
            causal_sss = 0.0
            best_desc  = "collapsed"
            best_eta2  = 0.0
            source     = "collapsed"

        cls = ("C1" if causal_sss >= 0.15 else
               "C2" if causal_sss >= 0.07 else "C3")
        gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"
        est = "*" if source in ["estimated", "collapsed"] else " "

        print(f"  {display:<32} {causal_sss:>10.4f}{est} "
              f"{best_eta2:>10.4f} {best_desc:<12} "
              f"{gain_str:>10}  [{cls}]")

        all_sss[dataset] = {
            "sss":             round(causal_sss, 4),
            "best_eta2":       round(best_eta2, 4),
            "best_descriptor": best_desc,
            "moe_gain":        gain,
            "endpoint_class":  ep_class,
            "display_name":    display,
            "source":          source,
        }

    # ── Step 4: Spearman validation ────────────────────────────────────
    pairs = [(v["sss"], v["moe_gain"])
             for v in all_sss.values()
             if v["moe_gain"] is not None]
    sss_v  = [p[0] for p in pairs]
    gain_v = [p[1] for p in pairs]
    r, p_val = spearmanr(sss_v, gain_v)

    exact_pairs = [(v["sss"], v["moe_gain"])
                   for v in all_sss.values()
                   if v["moe_gain"] is not None
                   and v["source"] == "exact"]
    print(f"\n  Spearman r (all 22, n={len(pairs)}): "
          f"r={r:.4f}, p={p_val:.6f}")
    if len(exact_pairs) >= 5:
        r_ex, p_ex = spearmanr(
            [p[0] for p in exact_pairs],
            [p[1] for p in exact_pairs]
        )
        print(f"  Spearman r (exact only, n={len(exact_pairs)}): "
              f"r={r_ex:.4f}, p={p_ex:.6f}")

    c1 = sum(1 for v in all_sss.values() if v["sss"] >= 0.15)
    c2 = sum(1 for v in all_sss.values() if 0.07 <= v["sss"] < 0.15)
    c3 = sum(1 for v in all_sss.values() if v["sss"] < 0.07)
    print(f"\n  Class 1 (SSS≥0.15): {c1} endpoints")
    print(f"  Class 2 (SSS 0.07-0.15): {c2} endpoints")
    print(f"  Class 3 (SSS<0.07): {c3} endpoints")

    # ── Step 5: Save ───────────────────────────────────────────────────
    output = {
        "sss_per_dataset":  all_sss,
        "spearman_r":       round(r, 4),
        "spearman_p":       round(p_val, 6),
        "n":                len(pairs),
        "class_counts":     {"C1": c1, "C2": c2, "C3": c3},
        "failed_datasets":  failed,
    }
    with open("sss_results_final.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  [SAVED] sss_results_final.json")

    # ── Step 6: Figure ─────────────────────────────────────────────────
    plot_spectrum(all_sss, r, p_val)

    print(f"\n{'='*65}")
    print(f"  FINAL SPEARMAN r={r:.4f} (p={p_val:.6f})")
    if r >= 0.70:
        print(f"  Nature Methods level — proceed with full paper")
    elif r >= 0.60:
        print(f"  Nature Communications level")
    else:
        print(f"  Needs refinement")
    print(f"{'='*65}")


def plot_spectrum(sss_data, r, p_val,
                 save="sss_learnability_spectrum_final.png"):
    sorted_items = sorted(sss_data.items(),
                         key=lambda x: x[1]["sss"], reverse=True)
    names   = [v["display_name"]    for _, v in sorted_items]
    scores  = [v["sss"]             for _, v in sorted_items]
    colors  = [CLASS_COLORS.get(v["endpoint_class"], "#888")
               for _, v in sorted_items]
    gains   = [v["moe_gain"] or 0.0 for _, v in sorted_items]
    sources = [v["source"]           for _, v in sorted_items]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={'height_ratios': [3, 1.5]})
    fig.patch.set_facecolor('white')

    bars = ax1.barh(range(len(names)), scores,
                   color=colors, alpha=0.88,
                   edgecolor='white', linewidth=0.5)

    ax1.axvline(0.15, color='#1565C0', linewidth=1.5,
               linestyle='--', alpha=0.7)
    ax1.axvline(0.07, color='#F44336', linewidth=1.5,
               linestyle='--', alpha=0.7)

    for bar, score, src in zip(bars, scores, sources):
        marker = "*" if src in ["estimated","collapsed"] else ""
        ax1.text(score + 0.003,
                bar.get_y() + bar.get_height()/2,
                f'{score:.3f}{marker}',
                va='center', fontsize=8,
                color='#aaa' if marker else '#222')

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Causal Structural Sufficiency Score (SSS)', fontsize=10)
    ax1.set_title(
        f'Figure 1: ADMET Learnability Spectrum\n'
        f'Spearman r={r:.3f} with MoE gain (p={p_val:.4f}, n=22)',
        fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.2)

    mx = max(scores) * 1.12
    ax1.axvspan(0.15, mx,   alpha=0.04, color='#2196F3')
    ax1.axvspan(0.07, 0.15, alpha=0.04, color='#FF9800')
    ax1.axvspan(0.0,  0.07, alpha=0.04, color='#F44336')

    patches = [mpatches.Patch(color=c, label=k)
               for k, c in CLASS_COLORS.items()]
    ax1.legend(handles=patches, loc='lower right', fontsize=8)

    gain_colors = [CLASS_COLORS.get(
                   sss_data[k[0]]["endpoint_class"], "#888")
                   for k in sorted_items]
    ax2.scatter(scores, gains, c=gain_colors, s=70,
               alpha=0.88, edgecolors='white',
               linewidths=0.6, zorder=3)

    for k, v in sss_data.items():
        if abs(v["moe_gain"] or 0) > 5 or v["sss"] > 0.10:
            ax2.annotate(v["display_name"],
                        (v["sss"], v["moe_gain"] or 0),
                        fontsize=7.5, ha='left',
                        xytext=(4, 2),
                        textcoords='offset points')

    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.axvline(0.07, color='#F44336', linewidth=1.0,
               linestyle='--', alpha=0.5)
    ax2.axvline(0.15, color='#1565C0', linewidth=1.0,
               linestyle='--', alpha=0.5)
    ax2.set_xlabel('Causal SSS', fontsize=10)
    ax2.set_ylabel('MoE Gain (%)', fontsize=10)
    ax2.set_title('(B) Causal SSS vs MoE Performance Gain',
                 fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.2)
    ax2.text(0.98, 0.05, f'r={r:.3f}, p={p_val:.4f}',
            transform=ax2.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round',
                     facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save, dpi=180, bbox_inches='tight',
                facecolor='white')
    print(f"  [SAVED] {save}")
    plt.close()


if __name__ == "__main__":
    main()
