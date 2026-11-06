"""Fig. 5 from paper_v2/results/exp1_routing_controls.json. Run from repo root: python make_fig5.py
No embedded title (caption carries it); 17 cm width."""
import json, os, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
INK = "#0b0b0b"; SEC = "#52514e"; GRID = "#e1e0d9"
D = json.load(open(os.path.join("paper_v2", "results", "exp1_routing_controls.json"))); R = collections.defaultdict(dict)
for k, v in D.items():
    d, s, a = k.split("|"); R[d].setdefault(a, {})[int(s[4:])] = v
ARMS = [("learned", "Learned", "#2a78d6", "o"), ("frozen_gate", "Frozen", "#eb6834", "s"),
        ("fixed_kmeans", "k-means", "#1baf7a", "^"), ("fixed_hash", "Random", "#eda100", "D")]
order = [("caco2_wang", "Caco-2"), ("solubility_aqsoldb", "Solubility"), ("ld50_zhu", "LD50"), ("lipophilicity_astrazeneca", "Lipophilicity")]
fig = plt.figure(figsize=(6.7, 5.0)); gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.6, wspace=0.5)
rng = np.random.default_rng(0)
for j, (d, name) in enumerate(order):
    ax = fig.add_subplot(gs[0, j]); r0 = R[d]["learned"][min(R[d]["learned"])]; K = r0["top_k"]; Ee = r0["E"]; n = len(R[d]["learned"])
    for i, (a, lab, col, mk) in enumerate(ARMS):
        v = np.array([R[d][a][s]["test_mae"] for s in sorted(R[d][a])]); x = i + rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(x, v, s=14, color=col, marker=mk, edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, linewidth=1.3, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _, _ in ARMS], rotation=35, ha="right"); ax.set_xlim(-0.6, 3.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f"{name}\nK = {K}, E = {Ee}, {n} seeds", loc="left")
    if j == 0: ax.set_ylabel("Test MAE (lower is better)")
ax = fig.add_subplot(gs[1, :])
def active(sw):
    Mx = np.array([[np.nan if x is None else x for x in r] for r in sw["matrix"]], float); g = sw["groups"]
    if len(g) < 2: return np.nan
    sub = Mx[np.ix_(g, g)]; dg = np.diag(sub); off = np.array([np.mean(np.delete(sub[i], i)) for i in range(len(g))]); return off.mean() / dg.mean()
w = 0.32
for jj, (d, name) in enumerate(order):
    for kk, (a, lab, col, mk) in enumerate(ARMS[:2]):
        v = np.array([active(R[d][a][s]["swap"]) for s in sorted(R[d][a])]); x = jj + (kk - 0.5) * w * 1.3 + rng.uniform(-0.05, 0.05, len(v))
        ax.scatter(x, v, s=14, color=col, marker=mk, edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        ax.hlines(v.mean(), jj + (kk - 0.5) * w * 1.3 - 0.14, jj + (kk - 0.5) * w * 1.3 + 0.14, color=INK, linewidth=1.3, zorder=4)
ax.axhline(1.0, color=SEC, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1); ax.text(3.62, 1.0, "1.0 = experts\ninterchangeable", fontsize=6, color=SEC, va="center")
ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in order]); ax.set_xlim(-0.6, 4.25); ax.set_ylabel("Forced-swap error ratio\n(other active experts / own expert)")
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
ax.scatter([], [], color=ARMS[0][2], marker="o", label="Learned router"); ax.scatter([], [], color=ARMS[1][2], marker="s", label="Frozen random gate")
ax.set_ylim(top=1.85); ax.legend(frameon=False, loc="upper left", ncol=2)
fig.text(0.01, 0.995, "A", fontsize=11, fontweight="bold", va="top"); fig.text(0.01, 0.40, "B", fontsize=11, fontweight="bold", va="top")
os.makedirs(os.path.join("paper_v2", "figures"), exist_ok=True)
fig.savefig(os.path.join("paper_v2", "figures", "Fig5_routing_controls.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join("paper_v2", "figures", "Fig5_routing_controls.pdf"), bbox_inches="tight")
print("saved Fig5_routing_controls.png/.pdf")
