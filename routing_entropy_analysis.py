"""
routing_entropy_analysis.py
============================
Computes routing entropy + learnability classification
for all 22 TDC datasets using existing routing files.

Run: python routing_entropy_analysis.py
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Imports OK")

DATASETS = [
    "solubility_aqsoldb", "caco2_wang", "lipophilicity_astrazeneca",
    "ppbr_az", "ld50_zhu", "vdss_lombardo", "half_life_obach",
    "clearance_microsome_az", "clearance_hepatocyte_az", "hia_hou",
    "pgp_broccatelli", "bioavailability_ma", "bbb_martins",
    "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels", "herg", "ames", "dili",
]

DISPLAY = {
    "solubility_aqsoldb":             "Solubility",
    "caco2_wang":                     "Caco-2",
    "lipophilicity_astrazeneca":      "Lipophilicity",
    "ppbr_az":                        "PPBR",
    "ld50_zhu":                       "LD50",
    "vdss_lombardo":                  "VDss",
    "half_life_obach":                "Half-Life",
    "clearance_microsome_az":         "CL-Microsome",
    "clearance_hepatocyte_az":        "CL-Hepatocyte",
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
}

TASK_TYPES = {
    "solubility_aqsoldb":"regression","caco2_wang":"regression",
    "lipophilicity_astrazeneca":"regression","ppbr_az":"regression",
    "ld50_zhu":"regression","vdss_lombardo":"regression",
    "half_life_obach":"regression","clearance_microsome_az":"regression",
    "clearance_hepatocyte_az":"regression","hia_hou":"classification",
    "pgp_broccatelli":"classification","bioavailability_ma":"classification",
    "bbb_martins":"classification","cyp2d6_veith":"classification",
    "cyp3a4_veith":"classification","cyp2c9_veith":"classification",
    "cyp2d6_substrate_carbonmangels":"classification",
    "cyp3a4_substrate_carbonmangels":"classification",
    "cyp2c9_substrate_carbonmangels":"classification",
    "herg":"classification","ames":"classification","dili":"classification",
}

ENDPOINT_CLASSES = {
    "solubility_aqsoldb":"Physicochemical","caco2_wang":"Physicochemical",
    "lipophilicity_astrazeneca":"Physicochemical","ppbr_az":"Physicochemical",
    "ld50_zhu":"Toxicity","vdss_lombardo":"Distribution",
    "half_life_obach":"Metabolic","clearance_microsome_az":"Metabolic",
    "clearance_hepatocyte_az":"Metabolic","hia_hou":"Absorption",
    "pgp_broccatelli":"Absorption","bioavailability_ma":"Absorption",
    "bbb_martins":"Absorption","cyp2d6_veith":"CYP Inhibition",
    "cyp3a4_veith":"CYP Inhibition","cyp2c9_veith":"CYP Inhibition",
    "cyp2d6_substrate_carbonmangels":"Metabolic",
    "cyp3a4_substrate_carbonmangels":"Metabolic",
    "cyp2c9_substrate_carbonmangels":"Metabolic",
    "herg":"Toxicity","ames":"Toxicity","dili":"Toxicity",
}

MOE_GAINS = {
    "caco2_wang":20.521,"lipophilicity_astrazeneca":8.747,
    "solubility_aqsoldb":8.760,"ppbr_az":1.295,"vdss_lombardo":6.310,
    "half_life_obach":-36.701,"clearance_microsome_az":12.900,
    "clearance_hepatocyte_az":-11.407,"hia_hou":1.054,
    "pgp_broccatelli":0.463,"bioavailability_ma":3.705,
    "bbb_martins":-0.490,"cyp2d6_veith":-2.404,"cyp3a4_veith":1.830,
    "cyp2c9_veith":0.211,"cyp2d6_substrate_carbonmangels":-2.999,
    "cyp3a4_substrate_carbonmangels":-5.418,
    "cyp2c9_substrate_carbonmangels":-0.231,"herg":-7.368,
    "ames":-0.357,"dili":-5.203,"ld50_zhu":4.782,
}

CLASS_COLORS = {
    "Physicochemical":"#2196F3","Absorption":"#4CAF50",
    "Distribution":"#9C27B0","Metabolic":"#F44336",
    "CYP Inhibition":"#FF9800","Toxicity":"#795548",
}

SEEDS = [0, 1, 2, 3, 4]


def compute_entropy_stats():
    """Compute routing entropy and n_experts for all datasets."""
    results = {}

    print("\n[1] Computing routing entropy from v2 files...")
    print(f"\n  {'Dataset':<25} {'Entropy':>8} {'n_exp':>7} "
          f"{'Source':<12} {'MoE Gain':>10}")
    print("  " + "-"*65)

    for dataset in DATASETS:
        entropies, n_exps = [], []

        # Try v2 files first (from retrain_and_extract.py)
        for seed in SEEDS:
            wf = Path(f"routing_weights_{dataset}_seed{seed}_v2.npy")
            rf = Path(f"routing_{dataset}_seed{seed}_v2.npy")
            if wf.exists() and rf.exists():
                w = np.load(wf)           # [N, num_experts]
                mean_w = w.mean(axis=0)   # [num_experts]
                mean_w = mean_w / (mean_w.sum() + 1e-10)
                h = float(entropy(mean_w))
                n = int(len(np.unique(np.load(rf))))
                entropies.append(h)
                n_exps.append(n)

        source = "v2"

        # Fallback to v1 files
        if not entropies:
            for seed in SEEDS:
                wf = Path(f"routing_weights_{dataset}.npy")
                rf = Path(f"routing_{dataset}.npy")
                if wf.exists() and rf.exists():
                    w = np.load(wf)
                    mean_w = w.mean(axis=0)
                    mean_w = mean_w / (mean_w.sum() + 1e-10)
                    h = float(entropy(mean_w))
                    n = int(len(np.unique(np.load(rf))))
                    entropies.append(h)
                    n_exps.append(n)
                    break
            source = "v1" if entropies else "missing"

        if entropies:
            mean_h  = float(np.mean(entropies))
            std_h   = float(np.std(entropies))
            mean_n  = float(np.mean(n_exps))
            gain    = MOE_GAINS.get(dataset)
            gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"

            results[dataset] = {
                "entropy":      round(mean_h, 4),
                "entropy_std":  round(std_h, 4),
                "n_experts":    round(mean_n, 1),
                "moe_gain":     gain,
                "task":         TASK_TYPES[dataset],
                "endpoint_class": ENDPOINT_CLASSES[dataset],
                "display_name": DISPLAY[dataset],
                "source":       source,
            }
            print(f"  {DISPLAY[dataset]:<25} {mean_h:>8.4f} "
                  f"{mean_n:>7.1f} {source:<12} {gain_str:>10}")
        else:
            print(f"  {DISPLAY[dataset]:<25} {'NO DATA':>8}")

    return results


def assign_learnability_class(entropy_val, n_experts, task):
    """
    Three learnability classes based on routing signals.

    Class 1 — Structurally Determined:
        High entropy + multiple experts + regression
        MoE routing finds rich chemical subspace structure

    Class 2 — Partially Determined:
        Moderate entropy OR classification with some structure
        Mixed signal — structure matters but not sufficient alone

    Class 3 — Biologically Mediated:
        Zero/collapsed entropy + few experts
        Routing finds no useful structure — biological context needed
    """
    # Collapsed routing = always Class 3
    if n_experts <= 1 or entropy_val < 0.05:
        return "Class 3", "Biologically Mediated"

    # Regression with rich routing = Class 1
    if task == "regression" and entropy_val > 0.5 and n_experts >= 4:
        return "Class 1", "Structurally Determined"

    # Regression with moderate routing = Class 1 or 2
    if task == "regression" and entropy_val > 0.3:
        return "Class 1", "Structurally Determined"

    if task == "regression":
        return "Class 2", "Partially Determined"

    # Classification with high entropy = Class 2
    # (rich routing but binary labels limit gain)
    if entropy_val > 0.5:
        return "Class 2", "Partially Determined"

    return "Class 3", "Biologically Mediated"


def print_learnability_table(results):
    """Print the key paper table."""
    print("\n[2] Learnability Classification Table:")
    print(f"\n  {'Dataset':<22} {'Class':<10} {'Entropy':>8} "
          f"{'n_exp':>6} {'Task':<6} {'MoE Gain':>10} "
          f"{'Endpoint Category'}")
    print("  " + "-"*85)

    class_counts = {"Class 1": [], "Class 2": [], "Class 3": []}

    sorted_items = sorted(
        results.items(),
        key=lambda x: (
            0 if "Class 1" in assign_learnability_class(
                x[1]["entropy"], x[1]["n_experts"], x[1]["task"])[0]
            else 1 if "Class 2" in assign_learnability_class(
                x[1]["entropy"], x[1]["n_experts"], x[1]["task"])[0]
            else 2,
            -x[1]["entropy"]
        )
    )

    for dataset, v in sorted_items:
        cls, cls_desc = assign_learnability_class(
            v["entropy"], v["n_experts"], v["task"])
        gain = v["moe_gain"]
        gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"
        task_s = "regr" if v["task"] == "regression" else "clf"
        cls_num = cls.split()[1]  # "1", "2", or "3"

        print(f"  {v['display_name']:<22} {cls:<10} "
              f"{v['entropy']:>8.4f} {v['n_experts']:>6.1f} "
              f"{task_s:<6} {gain_str:>10}  "
              f"{v['endpoint_class']}")

        class_counts[cls].append({
            "name": v["display_name"],
            "gain": gain,
            "entropy": v["entropy"],
        })

    print(f"\n  Class 1 (Structurally Determined): "
          f"{len(class_counts['Class 1'])} endpoints")
    print(f"  Class 2 (Partially Determined):    "
          f"{len(class_counts['Class 2'])} endpoints")
    print(f"  Class 3 (Biologically Mediated):   "
          f"{len(class_counts['Class 3'])} endpoints")

    # Mean gain per class
    for cls_name, items in class_counts.items():
        gains = [x["gain"] for x in items if x["gain"] is not None]
        if gains:
            print(f"  {cls_name} mean MoE gain: {np.mean(gains):+.1f}%")

    return class_counts


def plot_diagnostic_figure(results, class_counts,
                           save="learnability_diagnostic.png"):
    """
    Three-panel diagnostic figure:
    A — Routing entropy vs MoE gain scatter
    B — Learnability spectrum bar chart
    C — Summary box plot by class
    """
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor('white')

    # ── Panel A — Entropy vs Gain scatter ─────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)

    for dataset, v in results.items():
        if v["moe_gain"] is None:
            continue
        col  = CLASS_COLORS.get(v["endpoint_class"], "#888")
        mrk  = 'o' if v["task"] == "regression" else 's'
        cls, _ = assign_learnability_class(
            v["entropy"], v["n_experts"], v["task"])
        alpha = 0.9

        ax1.scatter(v["entropy"], v["moe_gain"],
                   c=col, s=80, marker=mrk,
                   alpha=alpha, edgecolors='white',
                   linewidths=0.8, zorder=3)

        if abs(v["moe_gain"]) > 5 or v["entropy"] > 1.2:
            ax1.annotate(v["display_name"],
                        (v["entropy"], v["moe_gain"]),
                        fontsize=6.5, ha='left',
                        xytext=(3, 2),
                        textcoords='offset points')

    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    ax1.axvline(0.3, color='#1565C0', linewidth=1.2,
               linestyle='--', alpha=0.6, label='Class 1/2 (H=0.3)')
    ax1.axvline(0.05, color='#F44336', linewidth=1.2,
               linestyle='--', alpha=0.6, label='Class 2/3 (H=0.05)')

    ax1.set_xlabel('Routing Entropy (H)', fontsize=10)
    ax1.set_ylabel('MoE Gain (%)', fontsize=10)
    ax1.set_title('(A) Routing Entropy vs MoE Gain\n'
                 '○=regression  □=classification',
                 fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=7)

    patches = [mpatches.Patch(color=c, label=k)
               for k, c in CLASS_COLORS.items()]
    ax1.legend(handles=patches, fontsize=7,
              loc='upper left', title='Endpoint class',
              title_fontsize=7)

    # ── Panel B — Learnability spectrum (horizontal bars) ─────────────
    ax2 = fig.add_subplot(1, 3, 2)

    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1]["entropy"], reverse=True)

    names   = [v["display_name"]    for _, v in sorted_items]
    entrs   = [v["entropy"]         for _, v in sorted_items]
    colors  = [CLASS_COLORS.get(v["endpoint_class"], "#888")
               for _, v in sorted_items]
    cls_labels = []
    for _, v in sorted_items:
        cls, _ = assign_learnability_class(
            v["entropy"], v["n_experts"], v["task"])
        cls_labels.append(cls)

    bar_colors = []
    for cls in cls_labels:
        if "1" in cls:
            bar_colors.append('#2196F3')
        elif "2" in cls:
            bar_colors.append('#FF9800')
        else:
            bar_colors.append('#F44336')

    bars = ax2.barh(range(len(names)), entrs,
                   color=bar_colors, alpha=0.82,
                   edgecolor='white', linewidth=0.5)

    for i, (bar, h, cls) in enumerate(
            zip(bars, entrs, cls_labels)):
        ax2.text(h + 0.01, bar.get_y() + bar.get_height()/2,
                f'{h:.3f}',
                va='center', fontsize=7, color='#333')

    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Routing Entropy (H)', fontsize=10)
    ax2.set_title('(B) ADMET Learnability Spectrum\n'
                 'Ranked by routing entropy',
                 fontsize=10, fontweight='bold')
    ax2.grid(axis='x', alpha=0.2)

    ax2.axvline(0.3, color='#1565C0', linewidth=1.0,
               linestyle='--', alpha=0.6)
    ax2.axvline(0.05, color='#F44336', linewidth=1.0,
               linestyle='--', alpha=0.6)

    c1_patch = mpatches.Patch(color='#2196F3',
                              label='Class 1 — Structurally Determined')
    c2_patch = mpatches.Patch(color='#FF9800',
                              label='Class 2 — Partially Determined')
    c3_patch = mpatches.Patch(color='#F44336',
                              label='Class 3 — Biologically Mediated')
    ax2.legend(handles=[c1_patch, c2_patch, c3_patch],
              fontsize=7, loc='lower right')

    # ── Panel C — MoE gain by learnability class ──────────────────────
    ax3 = fig.add_subplot(1, 3, 3)

    c1_gains = [v["moe_gain"] for v in results.values()
                if v["moe_gain"] is not None and
                "1" in assign_learnability_class(
                    v["entropy"], v["n_experts"], v["task"])[0]]
    c2_gains = [v["moe_gain"] for v in results.values()
                if v["moe_gain"] is not None and
                "2" in assign_learnability_class(
                    v["entropy"], v["n_experts"], v["task"])[0]]
    c3_gains = [v["moe_gain"] for v in results.values()
                if v["moe_gain"] is not None and
                "3" in assign_learnability_class(
                    v["entropy"], v["n_experts"], v["task"])[0]]

    data    = [c1_gains, c2_gains, c3_gains]
    labels  = ['Class 1\nStructurally\nDetermined',
               'Class 2\nPartially\nDetermined',
               'Class 3\nBiologically\nMediated']
    bcolors = ['#2196F3', '#FF9800', '#F44336']

    bp = ax3.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], bcolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='-')

    # Overlay individual points
    for i, (gains, color) in enumerate(zip(data, bcolors), 1):
        jitter = np.random.uniform(-0.1, 0.1, len(gains))
        ax3.scatter(np.full(len(gains), i) + jitter,
                   gains, color=color, s=40,
                   alpha=0.8, zorder=3,
                   edgecolors='white', linewidths=0.5)

    # Mean labels
    for i, gains in enumerate(data, 1):
        if gains:
            ax3.text(i, max(gains) + 1.5,
                    f'μ={np.mean(gains):+.1f}%',
                    ha='center', fontsize=8, fontweight='bold')

    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel('MoE Gain (%)', fontsize=10)
    ax3.set_title('(C) MoE Gain by Learnability Class',
                 fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.2)

    plt.suptitle(
        'MoE Routing as a Diagnostic Instrument for ADMET Endpoint Learnability',
        fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(save, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"\n  [SAVED] {save}")
    plt.close()


def main():
    print("=" * 65)
    print("  Routing Entropy + Learnability Classification")
    print("=" * 65)

    results = compute_entropy_stats()

    if not results:
        print("\nNo routing files found.")
        print("Make sure routing_weights_*_v2.npy files exist")
        print("from running retrain_and_extract.py")
        return

    # Add learnability class to results
    for dataset, v in results.items():
        cls, desc = assign_learnability_class(
            v["entropy"], v["n_experts"], v["task"])
        v["learnability_class"] = cls
        v["learnability_desc"]  = desc

    class_counts = print_learnability_table(results)

    plot_diagnostic_figure(results, class_counts)

    # Save JSON
    with open("learnability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  [SAVED] learnability_results.json")

    # Print paper-ready summary
    print(f"\n{'='*65}")
    print("  PAPER-READY FINDING")
    print(f"{'='*65}")

    c1 = [v for v in results.values()
          if v["learnability_class"] == "Class 1"]
    c2 = [v for v in results.values()
          if v["learnability_class"] == "Class 2"]
    c3 = [v for v in results.values()
          if v["learnability_class"] == "Class 3"]

    c1_gains = [v["moe_gain"] for v in c1 if v["moe_gain"]]
    c2_gains = [v["moe_gain"] for v in c2 if v["moe_gain"]]
    c3_gains = [v["moe_gain"] for v in c3 if v["moe_gain"]]

    print(f"""
  MoE routing entropy identifies three learnability classes
  across 22 TDC ADMET endpoints:

  Class 1 — Structurally Determined (n={len(c1)}):
    High routing entropy → rich chemical subspace structure
    Mean MoE gain: {np.mean(c1_gains):+.1f}%
    Endpoints: {', '.join(v['display_name'] for v in c1)}

  Class 2 — Partially Determined (n={len(c2)}):
    Moderate entropy → partial structural signal
    Mean MoE gain: {np.mean(c2_gains):+.1f}%
    Endpoints: {', '.join(v['display_name'] for v in c2)}

  Class 3 — Biologically Mediated (n={len(c3)}):
    Collapsed/zero entropy → no useful structural partitioning
    Mean MoE gain: {np.mean(c3_gains):+.1f}%
    Endpoints: {', '.join(v['display_name'] for v in c3)}

  This is the first empirical learnability map of ADMET
  pharmacokinetics derived from GNN routing behavior.
""")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
