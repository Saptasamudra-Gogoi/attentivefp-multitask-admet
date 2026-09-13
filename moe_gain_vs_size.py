"""
moe_gain_vs_size.py
Correlates MoE performance gain with dataset size
Uses results_moegcn_tdc_v2.json vs results_gcn_tdc.json

Place in D:\molprop_project\ and run:
    python moe_gain_vs_size.py
"""

import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, pearsonr

print("Imports OK")

# ══════════════════════════════════════════════════════════════════════════
# DATASET SIZES  (hardcoded from your TDC output — no re-download needed)
# ══════════════════════════════════════════════════════════════════════════

DATASET_SIZES = {
    "caco2_wang":                    910,
    "solubility_aqsoldb":           9982,
    "lipophilicity_astrazeneca":    4200,
    "hia_hou":                       578,
    "pgp_broccatelli":              1218,
    "bioavailability_ma":            640,
    "vdss_lombardo":                1130,
    "ppbr_az":                      1614,
    "half_life_obach":               667,
    "clearance_hepatocyte_az":      1213,
    "clearance_microsome_az":       1102,
    "bbb_martins":                  2030,
    "cyp2d6_veith":                13130,
    "cyp3a4_veith":                12328,
    "cyp2c9_veith":                12092,
    "cyp2c9_substrate_carbonmangels": 669,
    "cyp2d6_substrate_carbonmangels": 667,
    "cyp3a4_substrate_carbonmangels": 670,
    "ld50_zhu":                     7385,
    "herg":                          655,
    # MoleculeNet
    "ESOL":                         1128,
    "FreeSolv":                      642,
    "Lipo":                         4200,
}

# ══════════════════════════════════════════════════════════════════════════
# LOAD RESULTS
# ══════════════════════════════════════════════════════════════════════════

def compute_gain(moe_score, base_score, metric):
    """
    Returns gain % — positive means MoE is better.
    For AUROC/spearman: higher is better → gain = (moe - base) / base * 100
    For mae/rmse:       lower is better  → gain = (base - moe) / base * 100
    """
    if base_score is None or base_score == 0:
        return None
    metric = metric.lower()
    if metric in ['mae', 'rmse']:
        return (base_score - moe_score) / abs(base_score) * 100
    else:  # auroc, spearman, accuracy
        return (moe_score - base_score) / abs(base_score) * 100


def load_results():
    print("\n[2] Loading results...")

    # MoE results
    with open("results_moegcn_tdc_v2.json") as f:
        moe_tdc = json.load(f)
    print(f"  MoE TDC datasets: {len(moe_tdc)}")

    # Baseline results
    with open("results_gcn_tdc.json") as f:
        base_tdc = json.load(f)
    print(f"  GCN TDC datasets: {len(base_tdc)}")

    # MoleculeNet — MoE
    moe_mol, base_mol = {}, {}
    if os.path.exists("results_moegcn_regr.json"):
        with open("results_moegcn_regr.json") as f:
            moe_mol = json.load(f)
    if os.path.exists("results_moedmpnn_regr.json"):
        with open("results_moedmpnn_regr.json") as f:
            tmp = json.load(f)
            for k, v in tmp.items():
                if k not in moe_mol:
                    moe_mol[k] = v

    # Baseline MoleculeNet — try dmpnn or gcn regr
    for fname in ["results_dmpnn_regr.json", "results_gcn_regr.json"]:
        if os.path.exists(fname):
            with open(fname) as f:
                base_mol = json.load(f)
            print(f"  Baseline MolNet from {fname}")
            break

    return moe_tdc, base_tdc, moe_mol, base_mol


# ══════════════════════════════════════════════════════════════════════════
# MATCH AND COMPUTE GAINS
# ══════════════════════════════════════════════════════════════════════════

def build_gain_table(moe_tdc, base_tdc, moe_mol, base_mol):
    rows = []

    print(f"\n  {'Dataset':<42} {'N':>7} {'MoE':>8} {'GCN':>8} {'Gain%':>8} Metric")
    print(f"  {'-'*80}")

    # ── TDC datasets ───────────────────────────────────────────────────────
    for key in moe_tdc:
        moe_v  = moe_tdc[key]
        base_v = base_tdc.get(key)

        if base_v is None:
            print(f"  {key:<42}   no baseline")
            continue

        moe_score  = moe_v.get('mean')
        base_score = base_v.get('mean')
        metric     = moe_v.get('metric', 'unknown')
        n          = DATASET_SIZES.get(key)

        if moe_score is None or base_score is None or n is None:
            print(f"  {key:<42}   missing data")
            continue

        gain = compute_gain(moe_score, base_score, metric)
        if gain is None:
            continue

        rows.append({
            'name':    key,
            'n':       n,
            'moe':     moe_score,
            'base':    base_score,
            'gain':    gain,
            'metric':  metric,
            'source':  'TDC'
        })
        print(f"  {key:<42} {n:>7} {moe_score:>8.4f} {base_score:>8.4f} "
              f"{gain:>+8.2f}% {metric}")

    # ── MoleculeNet datasets ───────────────────────────────────────────────
    for key in moe_mol:
        if key in [r['name'] for r in rows]:
            continue
        moe_v  = moe_mol[key]
        base_v = base_mol.get(key)

        if base_v is None:
            continue

        moe_score  = moe_v.get('mean')
        base_score = base_v.get('mean')
        metric     = moe_v.get('metric', 'rmse')
        n          = DATASET_SIZES.get(key)

        if moe_score is None or base_score is None or n is None:
            continue

        gain = compute_gain(moe_score, base_score, metric)
        if gain is None:
            continue

        rows.append({
            'name':   key,
            'n':      n,
            'moe':    moe_score,
            'base':   base_score,
            'gain':   gain,
            'metric': metric,
            'source': 'MolNet'
        })
        print(f"  {key:<42} {n:>7} {moe_score:>8.4f} {base_score:>8.4f} "
              f"{gain:>+8.2f}% {metric}")

    return rows


# ══════════════════════════════════════════════════════════════════════════
# CORRELATION
# ══════════════════════════════════════════════════════════════════════════

def run_correlation(rows):
    names   = [r['name']   for r in rows]
    ns      = np.array([r['n']    for r in rows])
    gains   = np.array([r['gain'] for r in rows])
    metrics = [r['metric'] for r in rows]

    r_sp, p_sp = spearmanr(ns, gains)
    r_pe, p_pe = pearsonr(ns, gains)

    print(f"\n  Spearman r = {r_sp:+.3f}   p = {p_sp:.4f}")
    print(f"  Pearson  r = {r_pe:+.3f}   p = {p_pe:.4f}")
    print(f"  N datasets = {len(rows)}")

    if r_sp < -0.3 and p_sp < 0.05:
        print(f"\n  ✓ HYPOTHESIS CONFIRMED")
        print(f"    Larger datasets → smaller MoE gains (r={r_sp:.2f}, p={p_sp:.3f})")
        print(f"    Supports: MoE sparse routing = implicit regularization")
        print(f"    PAPER CLAIM READY — see interpretation section below")
    elif r_sp < -0.15:
        print(f"\n  ~ Weak negative trend (r={r_sp:.2f}, p={p_sp:.3f})")
        print(f"    Trend in expected direction but not statistically significant")
        print(f"    More datasets would strengthen this")
    elif r_sp > 0.15:
        print(f"\n  ✗ Positive correlation (r={r_sp:.2f})")
        print(f"    MoE gains MORE on larger datasets — unexpected")
        print(f"    May be driven by metric heterogeneity (AUROC vs MAE)")
    else:
        print(f"\n  ~ No clear trend (r={r_sp:.2f})")

    return names, ns, gains, metrics, r_sp, p_sp, r_pe, p_pe


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_all(names, ns, gains, metrics, r_sp, p_sp, r_pe, p_pe):

    # Color by metric
    cmap = {
        'auroc': '#E91E63', 'auc': '#E91E63',
        'mae':   '#2196F3',
        'rmse':  '#1565C0',
        'spearman': '#9C27B0',
    }
    colors = [cmap.get(m.lower(), '#9E9E9E') for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Plot A: linear x ───────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(ns, gains, c=colors, s=90, alpha=0.85,
               edgecolors='white', linewidth=0.5, zorder=3)

    z = np.polyfit(ns, gains, 1)
    xs = np.linspace(ns.min(), ns.max(), 300)
    ax.plot(xs, np.poly1d(z)(xs), '--', color='#333', linewidth=1.5,
            alpha=0.7, label='Linear fit', zorder=2)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':', zorder=1)

    for i, name in enumerate(names):
        short = name.split('_')[0]
        ax.annotate(short, (ns[i], gains[i]),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7, alpha=0.85)

    ax.set_xlabel('Dataset Size (n molecules)', fontsize=11)
    ax.set_ylabel('MoE Gain over GCN Baseline (%)', fontsize=11)
    ax.set_title(f'MoE Gain vs Dataset Size\n'
                 f'Spearman r={r_sp:+.3f} (p={p_sp:.4f})  '
                 f'Pearson r={r_pe:+.3f}',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # ── Plot B: log x ──────────────────────────────────────────────────────
    ax2 = axes[1]
    log_ns = np.log10(ns)
    ax2.scatter(log_ns, gains, c=colors, s=90, alpha=0.85,
                edgecolors='white', linewidth=0.5, zorder=3)

    z2 = np.polyfit(log_ns, gains, 1)
    xs2 = np.linspace(log_ns.min(), log_ns.max(), 300)
    ax2.plot(xs2, np.poly1d(z2)(xs2), '--', color='#333', linewidth=1.5,
             alpha=0.7, zorder=2)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':', zorder=1)

    for i, name in enumerate(names):
        short = name.split('_')[0]
        ax2.annotate(short, (log_ns[i], gains[i]),
                     textcoords='offset points', xytext=(5, 3),
                     fontsize=7, alpha=0.85)

    ticks = [2, 3, 3.5, 4, 4.5]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f'{10**t:,.0f}' for t in ticks], fontsize=8)
    ax2.set_xlabel('Dataset Size (log₁₀ scale)', fontsize=11)
    ax2.set_ylabel('MoE Gain over GCN Baseline (%)', fontsize=11)
    ax2.set_title('Same — log₁₀ x-axis\n(clearer for wide size range)',
                  fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.25)

    # Legend
    patches = [
        mpatches.Patch(color='#E91E63', label='AUROC (classification)'),
        mpatches.Patch(color='#2196F3', label='MAE (regression)'),
        mpatches.Patch(color='#9C27B0', label='Spearman (regression)'),
        mpatches.Patch(color='#9E9E9E', label='Other'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle('Hypothesis: MoE gains larger on smaller datasets\n'
                 '(sparse routing as implicit regularization)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('moe_gain_vs_size.png', dpi=150, bbox_inches='tight')
    print("[SAVED] moe_gain_vs_size.png")
    plt.close()

    # ── Plot C: sorted bar ─────────────────────────────────────────────────
    order = np.argsort(gains)[::-1]
    s_names  = [names[i].replace('_', '\n') for i in order]
    s_gains  = gains[order]
    s_sizes  = ns[order]
    s_colors = [colors[i] for i in order]

    fig3, ax3 = plt.subplots(figsize=(max(12, len(names)*0.7), 6))
    bars = ax3.bar(range(len(s_names)), s_gains, color=s_colors,
                   alpha=0.85, edgecolor='white', linewidth=0.5)

    for i, (bar, n_val) in enumerate(zip(bars, s_sizes)):
        yoff = 0.3 if bar.get_height() >= 0 else -1.5
        ax3.text(i, bar.get_height() + yoff, f'n={n_val:,}',
                 ha='center', va='bottom', fontsize=6.5, color='#444')

    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_xticks(range(len(s_names)))
    ax3.set_xticklabels(s_names, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('MoE Gain (%)', fontsize=11)
    ax3.set_title(f'MoE Gain per Dataset (sorted) — annotated with n\n'
                  f'Spearman r={r_sp:+.3f} (p={p_sp:.4f})',
                  fontsize=11, fontweight='bold')
    ax3.legend(handles=patches, fontsize=8, loc='upper right')
    ax3.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig('moe_gain_sorted.png', dpi=150, bbox_inches='tight')
    print("[SAVED] moe_gain_sorted.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("  MoE Gain vs Dataset Size — Correlation Analysis")
    print("="*70)

    moe_tdc, base_tdc, moe_mol, base_mol = load_results()

    print("\n[3] Computing gains per dataset...")
    rows = build_gain_table(moe_tdc, base_tdc, moe_mol, base_mol)

    if len(rows) < 5:
        print(f"\n  Only {len(rows)} matched datasets — not enough.")
        return

    print(f"\n[4] Running correlation (n={len(rows)} datasets)...")
    names, ns, gains, metrics, r_sp, p_sp, r_pe, p_pe = run_correlation(rows)

    print("\n[5] Plotting...")
    plot_all(names, ns, gains, metrics, r_sp, p_sp, r_pe, p_pe)

    # Save JSON
    out = {
        'spearman_r': round(r_sp, 4),
        'spearman_p': round(p_sp, 4),
        'pearson_r':  round(r_pe, 4),
        'pearson_p':  round(p_pe, 4),
        'n_datasets': len(rows),
        'datasets': [{
            'name':   r['name'],
            'n':      r['n'],
            'moe':    round(r['moe'],  4),
            'base':   round(r['base'], 4),
            'gain':   round(r['gain'], 3),
            'metric': r['metric']
        } for r in rows]
    }
    with open('moe_gain_vs_size_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("[SAVED] moe_gain_vs_size_results.json")

    print(f"\n{'='*70}")
    print("  INTERPRETATION FOR PAPER")
    print(f"{'='*70}")
    print(f"""
  Spearman r = {r_sp:+.3f}  (p = {p_sp:.4f})

  r < -0.3 AND p < 0.05  →  PAPER CLAIM:
    "MoE-GCN provides significantly larger performance gains on smaller
     datasets (Spearman r={r_sp:.2f}, p={p_sp:.3f}, n={len(rows)} datasets),
     consistent with sparse expert routing acting as implicit regularization
     in the low-data regimes characteristic of ADMET benchmarking."

  -0.3 < r < 0  →  Weak trend, mention cautiously or drop.

  r > 0  →  No size dependence. Keep routing analysis as main contribution.

  NOTE: Mixed metrics (AUROC + MAE + Spearman) can mask correlation.
  Consider running separate correlations for regression vs classification
  subsets if the overall r is weak.
""")


if __name__ == '__main__':
    main()

