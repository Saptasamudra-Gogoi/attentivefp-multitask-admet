"""Fig. 2 from exp2_topk_dynamics.json: expert load vs epoch per K (one seed shown, others in thin lines),
plus gate-norm and eta2 summaries. Usage: python make_fig2_dynamics.py [dataset] [seed_shown]"""
import json, sys, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ds = sys.argv[1] if len(sys.argv) > 1 else "tox21"; show = int(sys.argv[2]) if len(sys.argv) > 2 else 0
D = json.load(open("exp2_topk_dynamics.json")); runs = collections.defaultdict(dict)
for k, v in D.items():
    d, K, s = k.split("|")[:3]
    if d == ds and not k.endswith('|frozen'): runs[int(K[1:])][int(s[4:])] = v
Ks = sorted(runs); COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, ax = plt.subplots(2, len(Ks), figsize=(3.6 * len(Ks), 6.2), dpi=300, squeeze=False)
for j, K in enumerate(Ks):
    r = runs[K][show] if show in runs[K] else runs[K][min(runs[K])]
    L = np.array(r["log"]["load"]); order = np.argsort(-L[-1]); a = ax[0, j]
    for i, e in enumerate(order): a.plot(range(1, len(L) + 1), L[:, e], color=COL[i % 4], lw=1.6, label=f"E{e}" if i < 4 else None)
    for i, e in enumerate(order[:4]): a.text(len(L) + 0.5, L[-1, e], f"E{e}", fontsize=7, color="#0b0b0b", va="center")
    a.axvline(r["best_epoch"] + 1, color="#c3c2b7", lw=1, ls=(0, (4, 3)))
    a.set_title(f"K = {K}  (seed {r['seed']}, {r['active_experts']}/{r['E']} active)", fontsize=9, loc="left"); a.set_xlabel("Epoch"); a.set_ylim(0, 1)
    if j == 0: a.set_ylabel("Fraction of train molecules per expert")
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e1e0d9", lw=.6)
    b = ax[1, j]
    for s, rr in runs[K].items(): b.plot(rr["log"]["gate_norm"], color=COL[0], lw=1.2, alpha=.8)
    b.set_xlabel("Epoch"); b.set_title("‖gate weights‖ (all seeds)", fontsize=8.5, loc="left"); b.spines[["top", "right"]].set_visible(False)
    b.grid(axis="y", color="#e1e0d9", lw=.6)
    if j == 0: b.set_ylabel("Gate-weight norm")
fig.suptitle(f"Fig. 2 | Expert load during training, {ds} — dashed line: best-validation epoch", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(f"Fig2_load_dynamics_{ds}.png"); print("saved")
# numeric summary
print(f"{'K':>2} {'n':>2} {'best_val':>14} {'active':>8} {'eta2 LogP':>16} {'eta2 ArRings':>16} {'gate end':>9}")
for K in Ks:
    R_ = list(runs[K].values()); f = lambda g: np.array([g(r) for r in R_], float)
    ms = lambda x: f"{np.nanmean(x):.3f}±{np.nanstd(x, ddof=1) if len(x) > 1 else 0:.3f}"
    print(f"{K:>2} {len(R_):>2} {ms(f(lambda r: r['best_val'])):>14} {ms(f(lambda r: r['active_experts'])):>8} "
          f"{ms(f(lambda r: r['eta2']['LogP'] or np.nan)):>16} {ms(f(lambda r: r['eta2']['ArRings'] or np.nan)):>16} {ms(f(lambda r: r['log']['gate_norm'][-1])):>9}")
