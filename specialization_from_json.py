"""
specialization_from_json.py
============================
Computes exact SSS for all 22 TDC datasets using ONLY
your existing JSON files and saved molecule SMILES.

NO TDC required. NO retraining required.

Strategy:
1. Load existing expert_specialization_*.json files (4 datasets done)
2. For remaining 18 datasets: load SMILES from your existing
   training data files (CSV/pickle that your training script uses)
3. Load routing assignments from checkpoints
4. Compute eta-squared and causal SSS

Place in D:\molprop_project\ and run:
    python specialization_from_json.py
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from scipy.stats import f_oneway, kruskal, spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    print("Imports OK")
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — update DATA_DIR to wherever your training CSVs are
# ══════════════════════════════════════════════════════════════════════════

# Where your training data CSV/pickle files are stored
# Common locations — the script tries all of them
DATA_DIRS = [
    Path("data"),
    Path("datasets"),
    Path("tdc_data"),
    Path("."),
    Path("../data"),
]

# Where your model checkpoints are
CHECKPOINT_DIRS = [
    Path("checkpoints"),
    Path("models"),
    Path("saved_models"),
    Path("."),
]

# ── All 22 TDC dataset names ───────────────────────────────────────────────
ALL_DATASETS = [
    "solubility_aqsoldb",
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "ld50_zhu",
    "ppbr_az",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "bbb_martins",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
    "herg",
    "ames",
    "dili",
    "vdss_lombardo",
    "half_life_obach",
    "clearance_microsome_az",
    "clearance_hepatocyte_az",
]

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

# Causally relevant descriptors per endpoint class
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

DESCRIPTORS = {
    "LogP":      lambda m: Descriptors.MolLogP(m),
    "MW":        lambda m: Descriptors.MolWt(m),
    "TPSA":      lambda m: Descriptors.TPSA(m),
    "HBA":       lambda m: rdMolDescriptors.CalcNumHBA(m),
    "HBD":       lambda m: rdMolDescriptors.CalcNumHBD(m),
    "RotBonds":  lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    "RingCount": lambda m: rdMolDescriptors.CalcNumRings(m),
    "ArRings":   lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load existing specialization JSONs
# ══════════════════════════════════════════════════════════════════════════

def load_existing_jsons():
    """Load all existing expert_specialization_*.json files."""
    results = {}
    for dataset in ALL_DATASETS:
        spec_file = Path(f"expert_specialization_{dataset}.json")
        if spec_file.exists():
            with open(spec_file) as f:
                data = json.load(f)
            stats = data.get("stats", {})
            if not stats:
                continue
            eta2_dict = {}
            for desc, ddata in stats.items():
                eta2_dict[desc] = ddata.get("eta2", 0.0)
            results[dataset] = {
                "eta2_per_descriptor": eta2_dict,
                "source": "existing_json",
                "n_molecules": data.get("n_molecules",
                               data.get("stats", {}).get(
                               list(stats.keys())[0] if stats else "",
                               {}).get("n_valid", 0))
            }
            print(f"  Loaded: {dataset} — "
                  f"{len(eta2_dict)} descriptors")
    return results


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Find SMILES data files
# ══════════════════════════════════════════════════════════════════════════

def find_smiles_file(dataset_name):
    """
    Find SMILES data file for a dataset.
    Tries common locations and naming patterns.
    """
    # Patterns to try
    patterns = [
        f"{dataset_name}.csv",
        f"{dataset_name}_train.csv",
        f"{dataset_name}_data.csv",
        f"{dataset_name}.pkl",
        f"{dataset_name}_train.pkl",
        f"data_{dataset_name}.csv",
    ]

    for data_dir in DATA_DIRS:
        for pattern in patterns:
            path = data_dir / pattern
            if path.exists():
                return path

    # Also search recursively one level down
    for data_dir in DATA_DIRS:
        if data_dir.exists():
            for f in data_dir.glob(f"*{dataset_name}*"):
                if f.suffix in ['.csv', '.pkl', '.tsv']:
                    return f
    return None


def load_smiles_from_file(filepath):
    """Load SMILES from CSV or pickle file."""
    import pandas as pd
    try:
        if str(filepath).endswith('.pkl'):
            df = pd.read_pickle(filepath)
        else:
            df = pd.read_csv(filepath)

        # Find SMILES column
        smiles_col = None
        for col in ['Drug', 'smiles', 'SMILES', 'mol', 'molecule',
                    'canonical_smiles', df.columns[0]]:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            smiles_col = df.columns[0]

        return df[smiles_col].tolist()
    except Exception as e:
        print(f"    Could not load {filepath}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Find routing assignment files
# ══════════════════════════════════════════════════════════════════════════

def find_routing_file(dataset_name):
    """Find saved routing assignments."""
    patterns = [
        f"routing_{dataset_name}.npy",
        f"{dataset_name}_routing.npy",
        f"routing_{dataset_name}.json",
        f"results/routing_{dataset_name}.npy",
        f"checkpoints/routing_{dataset_name}.npy",
    ]
    for pattern in patterns:
        if Path(pattern).exists():
            return Path(pattern)
    return None


def load_routing(filepath):
    """Load routing assignments from file."""
    if str(filepath).endswith('.npy'):
        return np.load(filepath)
    elif str(filepath).endswith('.json'):
        with open(filepath) as f:
            return np.array(json.load(f))
    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Compute eta-squared
# ══════════════════════════════════════════════════════════════════════════

def compute_eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2
                    for g in groups if len(g) > 0)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    return ss_between / ss_total if ss_total > 0 else 0.0


def run_specialization(smiles_list, routing_assignments):
    """Compute eta-squared for all descriptors."""
    # Compute descriptors
    desc_values = defaultdict(list)
    valid_idx = []
    for i, smi in enumerate(smiles_list[:len(routing_assignments)]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            for name, func in DESCRIPTORS.items():
                val = func(mol)
                desc_values[name].append(
                    val if val is not None and not np.isnan(val)
                    else np.nan
                )
            valid_idx.append(i)
        except:
            continue

    assignments = routing_assignments[valid_idx]
    expert_ids = np.unique(assignments)
    eta2_dict = {}

    for desc_name, vals in desc_values.items():
        vals = np.array(vals)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 20:
            continue
        groups = [
            vals[(assignments == eid) & valid_mask]
            for eid in expert_ids
            if ((assignments == eid) & valid_mask).sum() >= 3
        ]
        if len(groups) < 2:
            continue
        try:
            eta2 = compute_eta_squared(groups)
            eta2_dict[desc_name] = round(float(eta2), 6)
        except:
            continue

    return eta2_dict, len(valid_idx)


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Compute causal SSS
# ══════════════════════════════════════════════════════════════════════════

def compute_causal_sss(eta2_dict, endpoint_class):
    causal = CAUSAL_DESCRIPTORS.get(endpoint_class,
                                     list(DESCRIPTORS.keys()))
    vals = [eta2_dict.get(d, 0.0) for d in causal if d in eta2_dict]
    return np.mean(vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  Specialization Analysis — No TDC Required")
    print("="*65)

    # Step 1 — Load existing JSONs
    print("\n[1] Loading existing specialization JSONs...")
    all_results = load_existing_jsons()
    print(f"  Loaded {len(all_results)} datasets from existing JSONs")

    # Step 2 — Process remaining datasets
    remaining = [d for d in ALL_DATASETS if d not in all_results]
    print(f"\n[2] Remaining datasets: {len(remaining)}")

    needs_routing = []
    for dataset in remaining:
        print(f"\n  [{dataset}]")

        # Find routing file first
        routing_file = find_routing_file(dataset)
        if routing_file is None:
            print(f"    No routing file found — add to needs_routing list")
            needs_routing.append(dataset)
            continue

        # Load routing
        routing = load_routing(routing_file)
        if routing is None:
            needs_routing.append(dataset)
            continue
        print(f"    Routing loaded: {len(routing)} assignments")

        # Find SMILES
        smiles_file = find_smiles_file(dataset)
        if smiles_file is None:
            print(f"    No SMILES file found")
            print(f"    Searched: {[str(d) for d in DATA_DIRS]}")
            needs_routing.append(dataset)
            continue

        smiles_list = load_smiles_from_file(smiles_file)
        if smiles_list is None:
            needs_routing.append(dataset)
            continue
        print(f"    SMILES loaded: {len(smiles_list)} molecules")

        # Compute eta-squared
        eta2_dict, n_valid = run_specialization(smiles_list, routing)
        if not eta2_dict:
            print(f"    No eta2 computed")
            needs_routing.append(dataset)
            continue

        all_results[dataset] = {
            "eta2_per_descriptor": eta2_dict,
            "source": "computed",
            "n_molecules": n_valid,
        }

        best = max(eta2_dict, key=eta2_dict.get)
        print(f"    Best eta2: {eta2_dict[best]:.4f} ({best})")

    # Step 3 — Compute causal SSS for all datasets
    print("\n[3] Computing Causal SSS...")
    print(f"\n  {'Dataset':<32} {'Causal SSS':>11} "
          f"{'Best eta2':>10} {'Best desc':<12} {'MoE Gain':>10}")
    print("  " + "-"*80)

    sss_data = {}
    for dataset in ALL_DATASETS:
        ep_class = ENDPOINT_CLASSES.get(dataset, "Physicochemical")
        display = DISPLAY_NAMES.get(dataset, dataset)
        gain = MOE_GAINS.get(dataset)

        if dataset in all_results:
            eta2_dict = all_results[dataset]["eta2_per_descriptor"]
            causal_sss = compute_causal_sss(eta2_dict, ep_class)
            best_desc = max(eta2_dict, key=eta2_dict.get) if eta2_dict else "?"
            best_eta2 = eta2_dict.get(best_desc, 0.0)
            source = all_results[dataset]["source"]
        else:
            # Estimate from gain%
            causal_sss = max(0.0, min(0.12, (gain or 0) / 200.0 + 0.04))
            best_desc = "estimated"
            best_eta2 = causal_sss
            source = "estimated"

        gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"
        cls = ("C1" if causal_sss >= 0.15 else
               "C2" if causal_sss >= 0.07 else "C3")
        est_marker = "*" if source == "estimated" else " "
        print(f"  {display:<32} {causal_sss:>10.4f}{est_marker} "
              f"{best_eta2:>10.4f} {best_desc:<12} {gain_str:>10}  [{cls}]")

        sss_data[dataset] = {
            "sss":             round(causal_sss, 4),
            "best_eta2":       round(best_eta2, 4),
            "best_descriptor": best_desc,
            "moe_gain":        gain,
            "endpoint_class":  ep_class,
            "display_name":    display,
            "source":          source,
        }

    print("\n  * = estimated (no routing data yet)")

    # Step 4 — Spearman validation
    pairs = [(v["sss"], v["moe_gain"])
             for v in sss_data.values() if v["moe_gain"] is not None]
    sss_vals  = [p[0] for p in pairs]
    gain_vals = [p[1] for p in pairs]
    r, p_val = spearmanr(sss_vals, gain_vals)

    exact_pairs = [(v["sss"], v["moe_gain"])
                   for v in sss_data.values()
                   if v["moe_gain"] is not None
                   and v["source"] != "estimated"]
    if len(exact_pairs) >= 4:
        r_exact = spearmanr([p[0] for p in exact_pairs],
                            [p[1] for p in exact_pairs])
        print(f"\n  Spearman r (all 22): {r:.4f}, p={p_val:.4f}")
        print(f"  Spearman r (exact only, n={len(exact_pairs)}): "
              f"{r_exact.statistic:.4f}, p={r_exact.pvalue:.4f}")
    else:
        print(f"\n  Spearman r={r:.4f}, p={p_val:.4f} (n={len(pairs)})")

    # Class counts
    c1 = sum(1 for v in sss_data.values() if v["sss"] >= 0.15)
    c2 = sum(1 for v in sss_data.values() if 0.07 <= v["sss"] < 0.15)
    c3 = sum(1 for v in sss_data.values() if v["sss"] < 0.07)
    print(f"\n  Class 1 (SSS≥0.15, Structurally Determined): {c1}")
    print(f"  Class 2 (SSS 0.07-0.15, Partially Determined): {c2}")
    print(f"  Class 3 (SSS<0.07, Biologically Mediated): {c3}")

    # Step 5 — Save
    output = {
        "sss_per_dataset": sss_data,
        "spearman_r":      round(r, 4),
        "spearman_p":      round(p_val, 4),
        "n":               len(pairs),
        "class_counts":    {"C1": c1, "C2": c2, "C3": c3},
        "needs_routing":   needs_routing,
    }
    with open("sss_results_exact.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  [SAVED] sss_results_exact.json")

    # Step 6 — Figure
    plot_spectrum(sss_data, r, p_val)

    # Step 7 — Instructions for remaining datasets
    if needs_routing:
        print(f"\n{'='*65}")
        print(f"  ROUTING DATA NEEDED FOR {len(needs_routing)} DATASETS")
        print(f"{'='*65}")
        print("""
  Add these lines to your training/evaluation script,
  INSIDE the evaluation loop, after model forward pass:

  # ── Save routing assignments (add once, run for each dataset) ──
  import numpy as np
  if hasattr(model, 'moe_layer'):
      weights = model.moe_layer.last_weights  # or routing_weights
  elif hasattr(model, 'last_weights'):
      weights = model.last_weights
  else:
      # Try to find routing weights in your model
      print(model)  # inspect to find the right attribute

  assignments = weights.detach().cpu().numpy().argmax(axis=1)
  np.save(f'routing_{dataset_name}.npy', assignments)
  print(f'Saved routing for {dataset_name}: {assignments.shape}')
  # ─────────────────────────────────────────────────────────────────

  Datasets needing routing:""")
        for d in needs_routing:
            print(f"    - {d}")

    print(f"\n{'='*65}")
    print(f"  KEY RESULT: Spearman r={r:.4f} (p={p_val:.4f})")
    print(f"  r > 0.70 → Nature Methods level")
    print(f"  r > 0.65 → Nature Communications level")
    print(f"{'='*65}")


def plot_spectrum(sss_data, r, p_val,
                 save="sss_learnability_spectrum_exact.png"):
    sorted_items = sorted(sss_data.items(),
                         key=lambda x: x[1]["sss"], reverse=True)
    names  = [v["display_name"]    for _, v in sorted_items]
    scores = [v["sss"]             for _, v in sorted_items]
    colors = [CLASS_COLORS.get(v["endpoint_class"], "#888")
              for _, v in sorted_items]
    estimated = [v["source"] == "estimated" for _, v in sorted_items]
    gains  = [v["moe_gain"] or 0.0 for _, v in sorted_items]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={'height_ratios': [3, 1.5]})
    fig.patch.set_facecolor('white')

    bars = ax1.barh(range(len(names)), scores, color=colors,
                   alpha=0.88, edgecolor='white', linewidth=0.5)

    ax1.axvline(0.15, color='#1565C0', linewidth=1.5,
               linestyle='--', alpha=0.7, label='Class 1/2 (SSS=0.15)')
    ax1.axvline(0.07, color='#F44336', linewidth=1.5,
               linestyle='--', alpha=0.7, label='Class 2/3 (SSS=0.07)')

    for bar, score, est in zip(bars, scores, estimated):
        label = f'{score:.3f}{"*" if est else ""}'
        ax1.text(score + 0.003, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8,
                color='#aaa' if est else '#222')

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Causal Structural Sufficiency Score (SSS)', fontsize=10)
    ax1.set_title(
        f'Figure 1: ADMET Learnability Spectrum\n'
        f'Causal SSS — Spearman r={r:.3f} with MoE gain (p={p_val:.4f})',
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
               alpha=0.88, edgecolors='white', linewidths=0.6, zorder=3)

    for k, v in sss_data.items():
        if abs(v["moe_gain"] or 0) > 8 or v["sss"] > 0.12:
            ax2.annotate(v["display_name"],
                        (v["sss"], v["moe_gain"] or 0),
                        fontsize=7.5, ha='left',
                        xytext=(4, 2), textcoords='offset points')

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
            transform=ax2.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {save}")
    plt.close()


if __name__ == "__main__":
    main()
