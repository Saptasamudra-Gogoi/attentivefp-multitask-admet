"""Combined Fig. 2 from paper_v2/results/exp2_topk_dynamics.json ONLY.
Run from the repo root:  python paper_v2\make_fig2_combined.py
Rows: A load Tox21, A gate norm Tox21, B load Lipophilicity, B gate norm Lipophilicity. Columns K = 1, 2, 4.
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.join("paper_v2", "results", "exp2_topk_dynamics.json")
OUT = os.path.join("paper_v2", "figures")
D = json.load(open(SRC))

runs = collections.defaultdict(lambda: collections.defaultdict(dict))  # runs[ds][K][seed]  (learned gate)
frozen = collections.defaultdict(dict)                                   # frozen[ds][seed] (K=2 frozen gate)
def _gn(r):
    return np.array([x if np.isscalar(x) else x[0] for x in r["log"]["gate_norm"]], dtype=float)
for key, r in D.items():
    ds = key.split("|")[0]
    g = _gn(r)
    const = bool(np.ptp(g) < 1e-9)
    flag = bool(r.get("frozen", False)) or key.endswith("|frozen")
    if flag != const:
        print(f"WARNING {key}: frozen flag={flag} but gate-norm constant={const}")
    is_frozen = flag or const
    if is_frozen:
        frozen[ds][int(r["seed"])] = r
    else:
        runs[ds][int(r["K"])][int(r["seed"])] = r

DS = [("tox21", "Tox21", "A"), ("lipophilicity_astrazeneca", "AstraZeneca Lipophilicity", "B")]
KS = [1, 2, 4]
COL = ["#2b6fd6", "#e8643a", "#1fa97a", "#e8a317"]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0})
fig, ax = plt.subplots(4, 3, figsize=(6.7, 8.6))

for di, (ds, nice, letter) in enumerate(DS):
    rl, rg = 2 * di, 2 * di + 1
    for ci, K in enumerate(KS):
        seeds = runs[ds].get(K, {})
        if not seeds:
            raise SystemExit(f"missing {ds} K={K}")
        r0 = seeds[min(seeds)]
        L = np.array(r0["log"]["load"], dtype=float)           # (epochs, E)
        ep = np.arange(1, L.shape[0] + 1)
        order = np.argsort(-L[-1])[:4]
        a = ax[rl, ci]
        ys = {e: L[-1, e] for e in order}
        pos = {}; last = -9
        for e in sorted(order, key=lambda e: ys[e]):          # de-overlap end labels (min gap 0.06)
            p = max(ys[e], last + 0.06); pos[e] = p; last = p
        for j, e in enumerate(order):
            a.plot(ep, L[:, e], color=COL[j % 4])
            a.text(ep[-1] + 0.8, pos[e], f"E{e}", fontsize=5.5, va="center")
        a.axvline(r0["best_epoch"] + 1, color="#aaa", ls="--", lw=0.7)
        a.set_ylim(0, 1); a.set_xlim(0, ep[-1] + 6)
        a.set_title(f"K = {K} (seed {r0['seed']}, {r0['active_experts']}/{r0['E']} active)", loc="left")
        a.set_xlabel("Epoch")
        if ci == 0: a.set_ylabel("Fraction of training molecules\nper expert")
        a.spines[["top", "right"]].set_visible(False)
        b = ax[rg, ci]
        for s, r in sorted(seeds.items()):
            g = _gn(r)
            b.plot(np.arange(len(g)), g, color="#3a7bd5", lw=0.8, alpha=0.9)
            print(f"{ds:28s} K={K} seed={s} best_epoch={r['best_epoch']} (plotted {r['best_epoch']+1}) "
                  f"active={r['active_experts']}/{r['E']} gate_norm {g[0]:.3f} -> {g[-1]:.3f} n_ep={len(g)}")
        if K == 2 and frozen.get(ds):
            for s_, r in sorted(frozen[ds].items()):
                g = _gn(r)
                b.plot(np.arange(len(g)), g, color="#888", lw=0.8, ls="--")
                print(f"{ds:28s} K=2 FROZEN seed={s_} best_epoch={r['best_epoch']} gate_norm {g[0]:.3f} -> {g[-1]:.3f}")
            b.text(0.97, 0.20, "solid: learned gate\ndashed grey: frozen gate", transform=b.transAxes, ha="right", va="bottom", fontsize=6, color="#444")
        b.set_xlabel("Epoch")
        b.set_title(f"Gate-weight norm (n = {len(seeds)} seeds)", loc="left")
        if ci == 0: b.set_ylabel("Gate-weight norm")
        b.spines[["top", "right"]].set_visible(False)
    ax[rl, 0].text(-0.30, 1.22, letter, transform=ax[rl, 0].transAxes, fontsize=11, fontweight="bold")
    ax[rl, 1].text(0.5, 1.22, nice, transform=ax[rl, 1].transAxes, ha="center", fontsize=8, fontweight="bold")

fig.tight_layout(h_pad=1.2, w_pad=1.0)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig2_load_dynamics.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Fig2_load_dynamics.pdf"))
print("saved Fig2_load_dynamics.png/.pdf")
