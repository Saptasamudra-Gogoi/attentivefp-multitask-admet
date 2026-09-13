"""
analyze_entropy_correlation.py (FIXED)
=======================================
Fixed: pingouin p-val column name varies by version -> use manual fallback
Fixed: nan handling in B2 mean correlation
Fixed: added top_k diagnostic table

Run after extract_routing_entropy.py completes.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ENTROPY_DIR  = Path("entropy_results")
FIGURES_DIR  = Path("figures")
ANALYSIS_DIR = Path("analysis")
FIGURES_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

TIER_COLORS = {
    "Tier 1 (>10%)":  "#2ecc71",
    "Tier 2 (1-10%)": "#f39c12",
    "Tier 3 (<1%)":   "#e74c3c",
    "Unknown":        "#95a5a6",
}

def classify_tier(perf_gain):
    if perf_gain is None: return "Unknown"
    if perf_gain > 0.10:  return "Tier 1 (>10%)"
    if perf_gain > 0.01:  return "Tier 2 (1-10%)"
    return "Tier 3 (<1%)"

def load_global_summary():
    path = ENTROPY_DIR / "global_entropy_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Run extract_routing_entropy.py first!")
    with open(path) as f:
        return json.load(f)

# ═══════════════════════════════════════════════════════════════
# B1: ENTROPY vs PERFORMANCE CORRELATION
# ═══════════════════════════════════════════════════════════════

def analyze_b1(data):
    # Exclude top_k=1 datasets (entropy always 0 by definition)
    valid = {}
    excluded = []
    for k, v in data.items():
        if v["perf_gain"] is None:
            continue
        params = v.get("params", {})
        topk = params.get("top_k", 99)
        if topk <= 1 or v["dataset_entropy"] < 0.001:
            excluded.append(k)
            continue
        valid[k] = v

    print(f"\n{'='*60}")
    print("  B1: ROUTING ENTROPY vs PERFORMANCE GAIN")
    print(f"{'='*60}")
    if excluded:
        print(f"  Excluded (top_k=1 or zero entropy): {excluded}")
    print(f"  N datasets for correlation: {len(valid)}")

    if len(valid) < 5:
        print("  WARNING: Too few datasets for meaningful correlation")
        return {"pearson_r": float('nan'), "pearson_p": float('nan'),
                "spearman_r": float('nan'), "spearman_p": float('nan'),
                "ci_low": float('nan'), "ci_high": float('nan'),
                "n_datasets": len(valid), "interpretation": "INSUFFICIENT_DATA",
                "entropies": [], "perf_gains": [], "dataset_keys": []}

    entropies    = np.array([v["dataset_entropy"] for v in valid.values()])
    perf_gains   = np.array([v["perf_gain"]       for v in valid.values()])
    dataset_keys = list(valid.keys())

    r_pearson,  p_pearson  = pearsonr(entropies, perf_gains)
    r_spearman, p_spearman = spearmanr(entropies, perf_gains)

    # Bootstrap CI
    np.random.seed(42)
    boot_r = []
    for _ in range(1000):
        idx = np.random.choice(len(entropies), len(entropies), replace=True)
        if len(np.unique(idx)) < 3:
            continue
        r, _ = pearsonr(entropies[idx], perf_gains[idx])
        boot_r.append(r)
    ci_low  = float(np.percentile(boot_r, 2.5))  if boot_r else float('nan')
    ci_high = float(np.percentile(boot_r, 97.5)) if boot_r else float('nan')

    print(f"  Pearson r  = {r_pearson:.4f}  (p={p_pearson:.4f})")
    print(f"  95% CI     = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Spearman ρ = {r_spearman:.4f}  (p={p_spearman:.4f})")

    if abs(r_pearson) < 0.3:
        interp = "WEAK — entropy does NOT predict performance gain"
        print(f"  ✗ {interp}")
    elif abs(r_pearson) < 0.5:
        interp = "MODERATE"
        print(f"  ~ {interp}")
    else:
        interp = "STRONG"
        print(f"  ✓ {interp}")

    return {
        "pearson_r":    float(r_pearson),
        "pearson_p":    float(p_pearson),
        "spearman_r":   float(r_spearman),
        "spearman_p":   float(p_spearman),
        "ci_low":       ci_low,
        "ci_high":      ci_high,
        "n_datasets":   len(valid),
        "interpretation": interp,
        "entropies":    entropies.tolist(),
        "perf_gains":   perf_gains.tolist(),
        "dataset_keys": dataset_keys,
    }

def plot_b1(data, b1):
    valid = {k: v for k, v in data.items()
             if v["perf_gain"] is not None
             and v["dataset_entropy"] >= 0.001
             and v.get("params", {}).get("top_k", 99) > 1}

    fig, ax = plt.subplots(figsize=(11, 7))
    for key, v in valid.items():
        tier  = classify_tier(v["perf_gain"])
        color = TIER_COLORS.get(tier, "gray")
        ax.scatter(v["dataset_entropy"], v["perf_gain"],
                   c=color, s=90, zorder=5, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
        if abs(v["perf_gain"]) > 0.08 or v["dataset_entropy"] < 0.4:
            ax.annotate(key.replace("_", "\n"), (v["dataset_entropy"], v["perf_gain"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 7), textcoords="offset points")

    entropies  = np.array([v["dataset_entropy"] for v in valid.values()])
    perf_gains = np.array([v["perf_gain"]       for v in valid.values()])
    if len(entropies) >= 3:
        z  = np.polyfit(entropies, perf_gains, 1)
        xs = np.linspace(entropies.min(), entropies.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "k--", alpha=0.4, linewidth=1.5)

    r  = b1["pearson_r"]
    pv = b1["pearson_p"]
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.3f}\np = {pv:.3f}\nn = {b1['n_datasets']} datasets",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Dataset-Level Routing Entropy", fontsize=13)
    ax.set_ylabel("MoE Performance Gain vs GCN Baseline", fontsize=13)
    ax.set_title("B1: Routing Entropy vs MoE Performance Gain\n"
                 "(top_k=1 datasets excluded; dashed=linear fit)", fontsize=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=t)
                        for t, c in TIER_COLORS.items()],
              loc="lower left", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "B1_entropy_vs_performance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

# ═══════════════════════════════════════════════════════════════
# B2: SEED INVARIANCE
# ═══════════════════════════════════════════════════════════════

def analyze_b2(data):
    corr_values = []
    invariant, variable = [], []
    for key, v in data.items():
        corr = v.get("seed_corr", None)
        if corr is None or np.isnan(corr):
            continue
        corr_values.append(corr)
        if corr > 0.7:
            invariant.append(key)
        else:
            variable.append(key)

    mean_corr = float(np.mean(corr_values)) if corr_values else float('nan')

    print(f"\n{'='*60}")
    print("  B2: SEED INVARIANCE")
    print(f"{'='*60}")
    print(f"  Mean seed-pair Spearman correlation: {mean_corr:.4f}")
    print(f"  Invariant (ρ>0.7): {len(invariant)}/{len(data)}")
    if invariant: print(f"    {', '.join(invariant)}")
    if variable:  print(f"  Variable: {len(variable)} datasets")

    if mean_corr > 0.7:
        conclusion = "INVARIANT — entropy is data-intrinsic"
    elif mean_corr > 0.5:
        conclusion = "MODERATE"
    else:
        conclusion = "VARIABLE — entropy reflects model initialization"
    print(f"  → {conclusion}")

    return {
        "mean_seed_corr":     mean_corr,
        "n_invariant":        len(invariant),
        "n_total":            len(data),
        "invariant_datasets": invariant,
        "variable_datasets":  variable,
        "corr_values":        corr_values,
        "conclusion":         conclusion,
    }

def plot_b2(data, b2):
    keys  = [k for k in data.keys() if not np.isnan(data[k].get("seed_corr", float('nan')))]
    corrs = [data[k]["seed_corr"] for k in keys]
    colors = ["#2ecc71" if c > 0.7 else "#e74c3c" for c in corrs]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(keys)), corrs, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(y=0.7, color="black", linestyle="--", linewidth=1.5,
               label="Invariance threshold (ρ=0.7)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Seed-Pair Spearman Correlation", fontsize=11)
    ax.set_title("B2: Seed Invariance of Routing Entropy Per Dataset\n"
                 "(Green=invariant ρ>0.7, Red=variable)", fontsize=11)
    ax.set_ylim(-0.1, 1.1)
    ax.text(0.02, 0.95,
            f"Mean ρ={b2['mean_seed_corr']:.3f}\n"
            f"Invariant: {b2['n_invariant']}/{b2['n_total']}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            verticalalignment="top")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "B2_seed_invariance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

# ═══════════════════════════════════════════════════════════════
# B3: PARTIAL CORRELATION (manual fallback, no pingouin dependency)
# ═══════════════════════════════════════════════════════════════

def analyze_b3(data, b1):
    valid = {k: v for k, v in data.items()
             if v["perf_gain"] is not None
             and v["dataset_entropy"] >= 0.001
             and v.get("params", {}).get("top_k", 99) > 1}

    entropies   = np.array([v["dataset_entropy"]        for v in valid.values()])
    perf_gains  = np.array([v["perf_gain"]              for v in valid.values()])
    log_n_mol   = np.array([np.log(v["n_molecules"]+1)  for v in valid.values()])
    task_binary = np.array([0 if v["task"]=="clf" else 1 for v in valid.values()])
    topk_vals   = np.array([v.get("params",{}).get("top_k",2) for v in valid.values()])

    print(f"\n{'='*60}")
    print("  B3: PARTIAL CORRELATION ANALYSIS")
    print(f"{'='*60}")

    # Manual partial correlation via OLS residuals
    from numpy.linalg import lstsq

    def partial_r(X_control, x_target, y_target):
        """Partial correlation of x_target and y_target controlling for X_control."""
        X = np.column_stack([np.ones(len(x_target)), X_control])
        coef_x, _, _, _ = lstsq(X, x_target, rcond=None)
        coef_y, _, _, _ = lstsq(X, y_target, rcond=None)
        resid_x = x_target - X @ coef_x
        resid_y = y_target - X @ coef_y
        if resid_x.std() < 1e-10 or resid_y.std() < 1e-10:
            return float('nan'), float('nan')
        r, p = pearsonr(resid_x, resid_y)
        return float(r), float(p)

    # Control 1: dataset size + task type
    r1, p1 = partial_r(
        np.column_stack([log_n_mol, task_binary]),
        entropies, perf_gains
    )
    print(f"  Partial r (controlling size + task):  r={r1:.4f}, p={p1:.4f}")

    # Control 2: top_k (the main confound)
    r2, p2 = partial_r(
        np.column_stack([topk_vals]),
        entropies, perf_gains
    )
    print(f"  Partial r (controlling top_k):        r={r2:.4f}, p={p2:.4f}")

    # Control 3: all three
    r3, p3 = partial_r(
        np.column_stack([log_n_mol, task_binary, topk_vals]),
        entropies, perf_gains
    )
    print(f"  Partial r (controlling size+task+k):  r={r3:.4f}, p={p3:.4f}")

    # top_k vs perf_gain direct correlation
    r_topk, p_topk = pearsonr(topk_vals, perf_gains)
    print(f"\n  Direct: top_k vs perf_gain:           r={r_topk:.4f}, p={p_topk:.4f}")

    # Confound check: is entropy just top_k?
    r_e_k, p_e_k = pearsonr(entropies, topk_vals)
    print(f"  Direct: entropy vs top_k:              r={r_e_k:.4f}, p={p_e_k:.4f}")
    if abs(r_e_k) > 0.7:
        print("  ⚠️  CONFIRMED: entropy is strongly confounded with top_k")
        print("     → Entropy is not an independent measure of dataset difficulty")
    else:
        print("  ✓  Entropy partially independent of top_k")

    return {
        "partial_r_size_task":   r1, "partial_p_size_task":   p1,
        "partial_r_topk":        r2, "partial_p_topk":        p2,
        "partial_r_all":         r3, "partial_p_all":         p3,
        "topk_vs_gain_r":        r_topk, "topk_vs_gain_p":    p_topk,
        "entropy_vs_topk_r":     r_e_k,  "entropy_vs_topk_p": p_e_k,
    }

# ═══════════════════════════════════════════════════════════════
# TOP_K DIAGNOSTIC TABLE
# ═══════════════════════════════════════════════════════════════

def topk_analysis(data):
    print(f"\n{'='*60}")
    print("  TOP_K ANALYSIS — The Real Driver")
    print(f"{'='*60}")
    print(f"  {'Dataset':<38} {'k':>3} {'Entropy':>8} {'PerfGain':>9} {'Metric':>10}")
    print("  " + "-"*72)

    by_topk = {}
    for k, v in sorted(data.items(), key=lambda x: (
        x[1].get("params",{}).get("top_k", 99),
        x[1].get("perf_gain") or 0
    )):
        params = v.get("params", {})
        topk   = params.get("top_k", "?")
        pg     = v.get("perf_gain")
        pg_str = f"{pg:+.3f}" if pg is not None else "N/A"
        metric = v.get("metric", "?")
        print(f"  {k:<38} {str(topk):>3} {v['dataset_entropy']:>8.4f} {pg_str:>9} {metric:>10}")
        if topk not in by_topk:
            by_topk[topk] = []
        if pg is not None:
            by_topk[topk].append(pg)

    print(f"\n  Performance gain by top_k group:")
    print(f"  {'top_k':>6} {'N':>4} {'Mean gain':>10} {'Std':>8}")
    for topk in sorted(by_topk.keys()):
        vals = by_topk[topk]
        if vals:
            print(f"  {str(topk):>6} {len(vals):>4} {np.mean(vals):>+10.4f} {np.std(vals):>8.4f}")

    # ANOVA-style: does top_k predict performance gain?
    topk_vals  = []
    gain_vals  = []
    for k, v in data.items():
        pg    = v.get("perf_gain")
        topk  = v.get("params", {}).get("top_k")
        if pg is not None and topk is not None:
            topk_vals.append(topk)
            gain_vals.append(pg)

    if len(topk_vals) >= 5:
        r, p = pearsonr(topk_vals, gain_vals)
        print(f"\n  top_k vs perf_gain: Pearson r={r:.4f}, p={p:.4f}")
        if p < 0.05:
            print("  ✓ top_k significantly predicts performance gain")
        else:
            print("  ✗ top_k does not significantly predict performance gain")

    with open(ANALYSIS_DIR / "topk_analysis.json", "w") as f:
        json.dump({"by_topk": {str(k): v for k, v in by_topk.items()},
                   "topk_vals": topk_vals, "gain_vals": gain_vals}, f, indent=2)

    return by_topk

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE GAIN TABLE — what actually worked
# ═══════════════════════════════════════════════════════════════

def performance_analysis(data):
    print(f"\n{'='*60}")
    print("  PERFORMANCE GAIN RANKING (what actually happened)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<38} {'PerfGain':>9} {'MoE':>8} {'GCN':>8} {'Metric':>10} {'task'}")
    print("  " + "-"*82)

    valid = [(k, v) for k, v in data.items() if v.get("perf_gain") is not None]
    for k, v in sorted(valid, key=lambda x: x[1]["perf_gain"], reverse=True):
        pg   = v["perf_gain"]
        moe  = v.get("moe_score", "?")
        gcn  = v.get("gcn_score", "?")
        moe_s = f"{moe:.4f}" if isinstance(moe, float) else str(moe)
        gcn_s = f"{gcn:.4f}" if isinstance(gcn, float) else str(gcn)
        pg_s  = f"{pg:+.3f}"
        print(f"  {k:<38} {pg_s:>9} {moe_s:>8} {gcn_s:>8} {v.get('metric','?'):>10} {v.get('task','?')}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading global entropy summary...")
    data = load_global_summary()
    n = len(data)
    n_gain = sum(1 for v in data.values() if v.get("perf_gain") is not None)
    print(f"Loaded {n} datasets, {n_gain} with performance gain")

    # Run analyses
    b1 = analyze_b1(data)
    plot_b1(data, b1)

    b2 = analyze_b2(data)
    plot_b2(data, b2)

    b3 = analyze_b3(data, b1)

    # Additional diagnostic tables
    topk_analysis(data)
    performance_analysis(data)

    # Save all results
    all_results = {"B1": b1, "B2": b2, "B3": b3}
    with open(ANALYSIS_DIR / "correlation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Final verdict
    print(f"\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")
    r = b1["pearson_r"]
    sc = b2["mean_seed_corr"]
    r3 = b3["partial_r_all"]

    print(f"  B1 Entropy-Performance r = {r:.3f}  → {'SIGNIFICANT' if b1['pearson_p']<0.05 else 'NOT SIGNIFICANT'}")
    print(f"  B2 Seed invariance mean  = {sc:.3f}  → {b2['conclusion']}")
    print(f"  B3 Partial r (all ctrl)  = {r3:.3f}")
    print(f"  Entropy-topK confound    = {b3['entropy_vs_topk_r']:.3f}")

    print()
    if abs(r) < 0.3:
        print("  RESULT: Entropy-performance hypothesis NOT supported.")
        print("  The routing entropy signal is dominated by top_k choice.")
        print()
        print("  → PIVOT: The real story is in the performance gain table.")
        print("    Look at what separates wins (caco2 +22%) from fails (half_life -38%).")
        print("    That pattern — not entropy — is the paper's scientific contribution.")
    elif abs(r) < 0.5:
        print("  RESULT: Weak correlation. Publishable but needs stronger framing.")
    else:
        print("  RESULT: Strong correlation confirmed. Proceed with original plan.")

    print(f"\n  Figures saved to {FIGURES_DIR}/")
    print(f"  Results saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
