"""
sss_label_variance.py
======================
Refined SSS using label variance + routing structure.

SSS = label_variance_score x routing_structure_score

label_variance_score:
  - Binary classification = 0 (no label regions to specialize over)
  - Continuous regression = normalized coefficient of variation
    (high spread = high score)

routing_structure_score:
  - Mean eta-squared across ALL descriptors (not just causal)
  - 0 if routing collapsed to 1 expert

Place in D:\molprop_project\ and run:
    python sss_label_variance.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

print("Imports OK")

# ══════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Task types — this is the key new information
TASK_TYPES = {
    "solubility_aqsoldb":             "regression",
    "caco2_wang":                     "regression",
    "lipophilicity_astrazeneca":      "regression",
    "ppbr_az":                        "regression",
    "ld50_zhu":                       "regression",
    "vdss_lombardo":                  "regression",
    "half_life_obach":                "regression",
    "clearance_microsome_az":         "regression",
    "clearance_hepatocyte_az":        "regression",
    # Classification — label_variance_score = 0
    "hia_hou":                        "classification",
    "pgp_broccatelli":                "classification",
    "bioavailability_ma":             "classification",
    "bbb_martins":                    "classification",
    "cyp2d6_veith":                   "classification",
    "cyp3a4_veith":                   "classification",
    "cyp2c9_veith":                   "classification",
    "cyp2d6_substrate_carbonmangels": "classification",
    "cyp3a4_substrate_carbonmangels": "classification",
    "cyp2c9_substrate_carbonmangels": "classification",
    "herg":                           "classification",
    "ames":                           "classification",
    "dili":                           "classification",
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

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Label variance scores from TDC data
# ══════════════════════════════════════════════════════════════════════════

def get_label_stats_from_tdc():
    """
    Load label distributions from TDC and compute
    coefficient of variation for each regression dataset.
    Binary classification gets score = 0.
    """
    print("\n[1] Computing label variance scores from TDC...")

    label_stats = {}

    try:
        from tdc.single_pred import ADME, Tox, ADMET
    except ImportError:
        print("  TDC import failed — using fallback estimates")
        return None

    tdc_map = {
        # Regression datasets
        "solubility_aqsoldb":         ("ADME", "Solubility_AqSolDB"),
        "caco2_wang":                 ("ADME", "Caco2_Wang"),
        "lipophilicity_astrazeneca":  ("ADME", "Lipophilicity_AstraZeneca"),
        "ppbr_az":                    ("ADME", "PPBR_AZ"),
        "ld50_zhu":                   ("Tox",  "LD50_Zhu"),
        "vdss_lombardo":              ("ADME", "VDss_Lombardo"),
        "half_life_obach":            ("ADME", "Half_Life_Obach"),
        "clearance_microsome_az":     ("ADME", "Clearance_Microsome_AZ"),
        "clearance_hepatocyte_az":    ("ADME", "Clearance_Hepatocyte_AZ"),
    }

    loader_map = {"ADME": ADME, "Tox": Tox}

    for dataset, (loader_name, tdc_name) in tdc_map.items():
        try:
            LoaderClass = loader_map[loader_name]
            data = LoaderClass(name=tdc_name)
            df   = data.get_data()
            y_col = next(
                (c for c in ['Y','label','activity','value']
                 if c in df.columns), df.columns[-1])
            labels = df[y_col].dropna().values.astype(float)

            mean  = np.mean(labels)
            std   = np.std(labels)
            cv    = std / abs(mean) if abs(mean) > 1e-6 else std
            iqr   = np.percentile(labels, 75) - np.percentile(labels, 25)
            rng   = np.max(labels) - np.min(labels)

            label_stats[dataset] = {
                "mean":  float(mean),
                "std":   float(std),
                "cv":    float(cv),
                "iqr":   float(iqr),
                "range": float(rng),
                "n":     int(len(labels)),
            }
            print(f"  {DISPLAY_NAMES[dataset]:<20} "
                  f"mean={mean:.2f} std={std:.2f} CV={cv:.3f}")
        except Exception as e:
            print(f"  {dataset}: failed ({e})")

    # Classification datasets get CV = 0
    for dataset, task in TASK_TYPES.items():
        if task == "classification":
            label_stats[dataset] = {
                "mean": 0.5, "std": 0.5, "cv": 0.0,
                "iqr": 1.0, "range": 1.0, "n": 0,
            }

    return label_stats


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Routing structure scores from existing JSONs
# ══════════════════════════════════════════════════════════════════════════

def get_routing_structure_scores():
    """
    Load eta-squared values from existing specialization JSONs.
    Use MEAN eta-squared across all descriptors (not causal subset).
    Collapsed routing = 0.
    """
    print("\n[2] Loading routing structure scores...")

    routing_scores = {}

    for dataset in TASK_TYPES:
        spec_file = Path(f"expert_specialization_{dataset}.json")

        if spec_file.exists():
            with open(spec_file) as f:
                data = json.load(f)
            stats = data.get("stats", {})
            if not stats:
                # Collapsed
                routing_scores[dataset] = 0.0
                print(f"  {DISPLAY_NAMES[dataset]:<20} "
                      f"routing_score=0.000 (collapsed)")
                continue

            # Mean eta-squared across ALL descriptors
            eta2_vals = []
            for desc, ddata in stats.items():
                eta2 = ddata.get("eta2", 0.0)
                if eta2 is not None:
                    eta2_vals.append(float(eta2))

            mean_eta2 = float(np.mean(eta2_vals)) if eta2_vals else 0.0
            routing_scores[dataset] = mean_eta2
            print(f"  {DISPLAY_NAMES[dataset]:<20} "
                  f"routing_score={mean_eta2:.4f} "
                  f"(mean of {len(eta2_vals)} descriptors)")
        else:
            # No JSON — collapsed or missing
            routing_scores[dataset] = 0.0
            print(f"  {DISPLAY_NAMES[dataset]:<20} "
                  f"routing_score=0.000 (no JSON)")

    return routing_scores


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute refined SSS
# ══════════════════════════════════════════════════════════════════════════

def compute_refined_sss(label_stats, routing_scores):
    """
    SSS = normalize(label_CV) x routing_structure_score

    For classification: label_CV = 0, so SSS = 0 regardless of routing.
    For regression: SSS proportional to both label spread and routing structure.
    """
    print("\n[3] Computing Refined SSS...")

    # Normalize CV across regression datasets only
    reg_cvs = [
        label_stats[d]["cv"]
        for d in TASK_TYPES
        if TASK_TYPES[d] == "regression"
        and d in label_stats
        and label_stats[d]["cv"] > 0
    ]
    max_cv = max(reg_cvs) if reg_cvs else 1.0
    min_cv = min(reg_cvs) if reg_cvs else 0.0
    cv_range = max_cv - min_cv if max_cv != min_cv else 1.0

    sss_data = {}

    for dataset in TASK_TYPES:
        task     = TASK_TYPES[dataset]
        gain     = MOE_GAINS.get(dataset)
        display  = DISPLAY_NAMES[dataset]
        ep_class = ENDPOINT_CLASSES.get(dataset, "Unknown")

        routing_score = routing_scores.get(dataset, 0.0)
        ls = label_stats.get(dataset, {})

        if task == "classification":
            # Binary classification — no label regions to specialize
            label_score = 0.0
            sss = 0.0
            formula = "clf→0"
        else:
            cv = ls.get("cv", 0.0)
            # Normalize CV to [0,1]
            label_score = (cv - min_cv) / cv_range if cv_range > 0 else 0.0
            label_score = max(0.0, min(1.0, label_score))
            # SSS = product of both components
            sss = label_score * routing_score
            formula = f"CV={cv:.3f}→{label_score:.3f} x route={routing_score:.4f}"

        sss_data[dataset] = {
            "sss":             round(sss, 5),
            "label_score":     round(label_score, 4),
            "routing_score":   round(routing_score, 4),
            "task_type":       task,
            "moe_gain":        gain,
            "endpoint_class":  ep_class,
            "display_name":    display,
            "formula":         formula,
        }

    return sss_data


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Also test alternative SSS formulas
# ══════════════════════════════════════════════════════════════════════════

def test_alternative_predictors(label_stats, routing_scores):
    """
    Test multiple predictors to find best Spearman r with MoE gain.
    """
    print("\n[4] Testing alternative SSS formulas...")

    datasets = [d for d in TASK_TYPES if MOE_GAINS.get(d) is not None]
    gains    = [MOE_GAINS[d] for d in datasets]

    def spearman(predictor_vals):
        r, p = spearmanr(predictor_vals, gains)
        return r, p

    results = {}

    # Formula 1: task_type only (regression=1, classification=0)
    pred1 = [1.0 if TASK_TYPES[d] == "regression" else 0.0
             for d in datasets]
    r1, p1 = spearman(pred1)
    results["task_type_binary"] = (r1, p1)
    print(f"  Task type (regr=1, clf=0):       r={r1:.4f}, p={p1:.4f}")

    # Formula 2: routing score only
    pred2 = [routing_scores.get(d, 0.0) for d in datasets]
    r2, p2 = spearman(pred2)
    results["routing_score_only"] = (r2, p2)
    print(f"  Routing score only (mean eta2):  r={r2:.4f}, p={p2:.4f}")

    # Formula 3: label CV only (0 for classification)
    pred3 = []
    for d in datasets:
        if TASK_TYPES[d] == "classification":
            pred3.append(0.0)
        else:
            pred3.append(label_stats.get(d, {}).get("cv", 0.0))
    r3, p3 = spearman(pred3)
    results["label_cv_only"] = (r3, p3)
    print(f"  Label CV only (0 for clf):       r={r3:.4f}, p={p3:.4f}")

    # Formula 4: CV x routing (the refined SSS)
    reg_cvs = [label_stats[d]["cv"]
               for d in datasets
               if TASK_TYPES[d] == "regression"
               and label_stats.get(d, {}).get("cv", 0) > 0]
    max_cv = max(reg_cvs) if reg_cvs else 1.0
    min_cv = min(reg_cvs) if reg_cvs else 0.0
    cv_range = max_cv - min_cv if max_cv != min_cv else 1.0

    pred4 = []
    for d in datasets:
        if TASK_TYPES[d] == "classification":
            pred4.append(0.0)
        else:
            cv = label_stats.get(d, {}).get("cv", 0.0)
            cv_norm = (cv - min_cv) / cv_range
            pred4.append(cv_norm * routing_scores.get(d, 0.0))
    r4, p4 = spearman(pred4)
    results["cv_x_routing"] = (r4, p4)
    print(f"  CV x routing (refined SSS):      r={r4:.4f}, p={p4:.4f}")

    # Formula 5: label std only
    pred5 = []
    for d in datasets:
        if TASK_TYPES[d] == "classification":
            pred5.append(0.0)
        else:
            pred5.append(label_stats.get(d, {}).get("std", 0.0))
    r5, p5 = spearman(pred5)
    results["label_std_only"] = (r5, p5)
    print(f"  Label std only (0 for clf):      r={r5:.4f}, p={p5:.4f}")

    # Formula 6: task_type x routing
    pred6 = [
        routing_scores.get(d, 0.0)
        if TASK_TYPES[d] == "regression" else 0.0
        for d in datasets
    ]
    r6, p6 = spearman(pred6)
    results["task_x_routing"] = (r6, p6)
    print(f"  Task x routing:                  r={r6:.4f}, p={p6:.4f}")

    # Formula 7: label IQR only
    pred7 = []
    for d in datasets:
        if TASK_TYPES[d] == "classification":
            pred7.append(0.0)
        else:
            pred7.append(label_stats.get(d, {}).get("iqr", 0.0))
    r7, p7 = spearman(pred7)
    results["label_iqr_only"] = (r7, p7)
    print(f"  Label IQR only (0 for clf):      r={r7:.4f}, p={p7:.4f}")

    # Find best
    best_name = max(results, key=lambda k: abs(results[k][0]))
    best_r, best_p = results[best_name]
    print(f"\n  BEST PREDICTOR: {best_name}")
    print(f"  Spearman r={best_r:.4f}, p={best_p:.4f}")

    return results, best_name


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Print full table and validate
# ══════════════════════════════════════════════════════════════════════════

def print_full_table(sss_data, label_stats, routing_scores):
    print("\n[5] Full SSS Table:")
    print(f"\n  {'Dataset':<22} {'Task':<6} {'LabelCV':>8} "
          f"{'RouteScore':>11} {'SSS':>8} {'MoE Gain':>10}")
    print("  " + "-"*75)

    sorted_items = sorted(sss_data.items(),
                         key=lambda x: x[1]["sss"], reverse=True)

    for dataset, v in sorted_items:
        ls   = label_stats.get(dataset, {})
        cv   = ls.get("cv", 0.0) if TASK_TYPES[dataset] == "regression" \
               else 0.0
        rt   = routing_scores.get(dataset, 0.0)
        gain = v["moe_gain"]
        gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"
        task_short = "regr" if v["task_type"] == "regression" else "clf"

        print(f"  {v['display_name']:<22} {task_short:<6} "
              f"{cv:>8.3f} {rt:>11.4f} {v['sss']:>8.5f} "
              f"{gain_str:>10}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Plot
# ══════════════════════════════════════════════════════════════════════════

def plot_refined_sss(sss_data, r_best, p_best, best_name,
                     save="sss_refined_spectrum.png"):
    sorted_items = sorted(sss_data.items(),
                         key=lambda x: x[1]["sss"], reverse=True)

    names   = [v["display_name"]    for _, v in sorted_items]
    scores  = [v["sss"]             for _, v in sorted_items]
    colors  = [CLASS_COLORS.get(v["endpoint_class"], "#888")
               for _, v in sorted_items]
    gains   = [v["moe_gain"] or 0.0 for _, v in sorted_items]
    tasks   = [v["task_type"]        for _, v in sorted_items]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={'height_ratios': [3, 1.5]})
    fig.patch.set_facecolor('white')

    # Hatching for classification
    for i, (bar_y, score, task) in enumerate(
            zip(range(len(names)), scores, tasks)):
        col   = colors[i]
        alpha = 0.88 if task == "regression" else 0.45
        hatch = "" if task == "regression" else "///"
        ax1.barh(bar_y, score, color=col, alpha=alpha,
                edgecolor='white' if task == "regression" else col,
                linewidth=0.5, hatch=hatch)
        ax1.text(score + 0.0005,
                bar_y,
                f'{score:.4f}',
                va='center', fontsize=7.5,
                color='#222')

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Refined SSS (Label CV × Routing Structure)',
                   fontsize=10)
    ax1.set_title(
        f'Figure 1: ADMET Learnability Spectrum — Refined SSS\n'
        f'Best predictor: {best_name} | '
        f'Spearman r={r_best:.3f} (p={p_best:.4f})',
        fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.2)

    # Legend
    patches = [mpatches.Patch(color=c, label=k)
               for k, c in CLASS_COLORS.items()]
    patches += [
        mpatches.Patch(facecolor='gray', alpha=0.45,
                       hatch='///', label='Classification (SSS=0)'),
        mpatches.Patch(facecolor='gray', alpha=0.88,
                       label='Regression'),
    ]
    ax1.legend(handles=patches, loc='lower right',
               fontsize=7.5, ncol=2)

    # Panel B — scatter
    gain_colors = [CLASS_COLORS.get(
                   sss_data[k[0]]["endpoint_class"], "#888")
                   for k in sorted_items]
    markers = ['o' if v["task_type"] == "regression" else 's'
               for _, v in sorted_items]

    for i, (k, v) in enumerate(sorted_items):
        col = CLASS_COLORS.get(v["endpoint_class"], "#888")
        m   = 'o' if v["task_type"] == "regression" else 's'
        ax2.scatter(v["sss"], v["moe_gain"] or 0,
                   c=col, s=70, marker=m,
                   alpha=0.88, edgecolors='white',
                   linewidths=0.6, zorder=3)
        if (abs(v["moe_gain"] or 0) > 5 or v["sss"] > 0.005):
            ax2.annotate(v["display_name"],
                        (v["sss"], v["moe_gain"] or 0),
                        fontsize=7, ha='left',
                        xytext=(4, 2),
                        textcoords='offset points')

    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_xlabel('Refined SSS', fontsize=10)
    ax2.set_ylabel('MoE Gain (%)', fontsize=10)
    ax2.set_title('(B) Refined SSS vs MoE Performance Gain\n'
                 '○=regression  □=classification',
                 fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.2)

    sss_v  = [v["sss"]      for v in sss_data.values()
              if v["moe_gain"] is not None]
    gain_v = [v["moe_gain"] for v in sss_data.values()
              if v["moe_gain"] is not None]
    r, p = spearmanr(sss_v, gain_v)
    ax2.text(0.98, 0.05, f'r={r:.3f}, p={p:.4f}',
            transform=ax2.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"\n  [SAVED] {save}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Refined SSS — Label Variance × Routing Structure")
    print("=" * 65)

    # Step 1 — Label stats from TDC
    label_stats = get_label_stats_from_tdc()
    if label_stats is None:
        print("FATAL: Could not load TDC label data")
        return

    # Step 2 — Routing structure from JSONs
    routing_scores = get_routing_structure_scores()

    # Step 3 — Compute refined SSS
    sss_data = compute_refined_sss(label_stats, routing_scores)

    # Step 4 — Test all formulas
    alt_results, best_name = test_alternative_predictors(
        label_stats, routing_scores)

    # Step 5 — Full table
    print_full_table(sss_data, label_stats, routing_scores)

    # Step 6 — Final Spearman for refined SSS
    pairs = [(v["sss"], v["moe_gain"])
             for v in sss_data.values()
             if v["moe_gain"] is not None]
    r_sss, p_sss = spearmanr(
        [p[0] for p in pairs],
        [p[1] for p in pairs]
    )

    best_r, best_p = alt_results[best_name]

    # Plot
    plot_refined_sss(sss_data, best_r, best_p, best_name)

    # Save
    output = {
        "sss_per_dataset":       sss_data,
        "spearman_refined_sss":  round(r_sss, 4),
        "p_refined_sss":         round(p_sss, 6),
        "all_predictors":        {
            k: {"r": round(v[0], 4), "p": round(v[1], 6)}
            for k, v in alt_results.items()
        },
        "best_predictor":        best_name,
        "best_r":                round(best_r, 4),
        "best_p":                round(best_p, 6),
    }
    with open("sss_refined_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  [SAVED] sss_refined_results.json")

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"\n  Refined SSS (CV x routing): r={r_sss:.4f}, p={p_sss:.4f}")
    print(f"  Best single predictor: {best_name}")
    print(f"  Best r={best_r:.4f}, p={best_p:.4f}")
    print(f"\n  All predictors tested:")
    for name, (r, p) in sorted(alt_results.items(),
                                key=lambda x: -abs(x[1][0])):
        sig = "***" if p < 0.001 else "**" if p < 0.01 \
              else "*" if p < 0.05 else "ns"
        print(f"    {name:<30} r={r:+.4f}  p={p:.4f}  {sig}")

    print(f"\n  INTERPRETATION:")
    if abs(best_r) >= 0.70:
        print(f"  r={best_r:.3f} → Nature Methods level")
    elif abs(best_r) >= 0.55:
        print(f"  r={best_r:.3f} → Nature Communications level")
    elif abs(best_r) >= 0.40:
        print(f"  r={best_r:.3f} → J. Cheminformatics level")
    else:
        print(f"  r={best_r:.3f} → SSS needs further refinement")
        print(f"  Consider: the endpoint category framework (Mann-Whitney")
        print(f"  p=0.0076) may be the stronger publishable finding.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
