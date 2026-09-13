"""
Fig 2 — Ablation Study: Routing vs No Routing
Journal of Cheminformatics submission figure
300 DPI, single-column width (88mm)

Shows MoE-GCN (routing) vs Dense-uniform vs Dense-wide
across 5 regression datasets with error bars.
Inset: parameter count comparison.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────

d = json.load(open('ablation_routing_results.json'))

DATASETS = ['ESOL', 'FreeSolv', 'Lipo', 'caco2_wang', 'solubility_aqsoldb']
DATASET_LABELS = ['ESOL', 'FreeSolv', 'Lipo', 'Caco-2', 'Solubility']
MODELS = ['MoE-GCN (routing)', 'Dense-uniform (no route)', 'Dense-wide (no route)']
MODEL_SHORT = ['MoE-GCN\n(routing)', 'Dense-uniform\n(no routing)', 'Dense-wide\n(no routing)']
COLORS = ['#2196F3', '#FF9800', '#9C27B0']
MARKERS = ['o', 's', '^']

# Extract means and stds
means = {m: [] for m in MODELS}
stds  = {m: [] for m in MODELS}
for ds in DATASETS:
    for m in MODELS:
        means[m].append(d[ds][m]['mean'])
        stds[m].append(d[ds][m]['std'])

# Parameter counts (from ESOL which has n_params)
params = {}
for m in MODELS:
    if 'n_params' in d['ESOL'][m]:
        params[m] = d['ESOL'][m]['n_params']
    else:
        # caco2/solubility don't have n_params — use ESOL values
        params[m] = d['ESOL'].get(m, {}).get('n_params', None)

# ── Figure setup ───────────────────────────────────────────────────────────────

# Double column for better readability
fig = plt.figure(figsize=(7.09, 4.2), dpi=300)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

gs = GridSpec(1, 2, figure=fig,
              wspace=0.38,
              left=0.08, right=0.97,
              top=0.88, bottom=0.18)

ax_bar   = fig.add_subplot(gs[0, 0])   # grouped bar — RMSE/MAE per dataset
ax_delta = fig.add_subplot(gs[0, 1])   # delta plot — MoE vs baselines

# ── Panel A — Grouped bar chart ───────────────────────────────────────────────

n_ds    = len(DATASETS)
n_mod   = len(MODELS)
x       = np.arange(n_ds)
width   = 0.24
offsets = [-width, 0, width]

for i, (m, color, offset) in enumerate(zip(MODELS, COLORS, offsets)):
    bars = ax_bar.bar(x + offset,
                      means[m],
                      width,
                      yerr=stds[m],
                      color=color,
                      alpha=0.82,
                      edgecolor='white',
                      linewidth=0.4,
                      error_kw={'elinewidth': 0.7,
                                'capsize': 2.0,
                                'capthick': 0.7,
                                'ecolor': '#444'})

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(DATASET_LABELS, fontsize=6.5, rotation=20, ha='right')
ax_bar.set_ylabel('RMSE / MAE ↓', fontsize=7)
ax_bar.set_title('Ablation: routing vs no routing\n(5 regression datasets, 5 seeds)', fontsize=7.5, pad=4)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color=COLORS[0], alpha=0.82, label='MoE-GCN (routing)'),
    mpatches.Patch(color=COLORS[1], alpha=0.82, label='Dense-uniform (no routing)'),
    mpatches.Patch(color=COLORS[2], alpha=0.82, label='Dense-wide (no routing)'),
]
ax_bar.legend(handles=legend_patches,
              fontsize=5.5,
              loc='upper left',
              framealpha=0.8,
              edgecolor='none',
              handlelength=1.0)

# Panel A label
ax_bar.text(-0.13, 1.05, 'A', transform=ax_bar.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ── Panel B — Delta plot (MoE vs each baseline) ───────────────────────────────

# Compute % improvement of MoE-GCN over baselines (lower RMSE = better)
# delta = (baseline - MoE) / baseline * 100
delta_uniform = []
delta_wide    = []

for i, ds in enumerate(DATASETS):
    moe_m  = means['MoE-GCN (routing)'][i]
    uni_m  = means['Dense-uniform (no route)'][i]
    wide_m = means['Dense-wide (no route)'][i]
    delta_uniform.append((uni_m  - moe_m) / uni_m  * 100)
    delta_wide.append(   (wide_m - moe_m) / wide_m * 100)

x2 = np.arange(n_ds)
w2 = 0.32

bars_u = ax_delta.bar(x2 - w2/2, delta_uniform,
                      w2, color=COLORS[1], alpha=0.82,
                      edgecolor='white', linewidth=0.4,
                      label='vs Dense-uniform')
bars_w = ax_delta.bar(x2 + w2/2, delta_wide,
                      w2, color=COLORS[2], alpha=0.82,
                      edgecolor='white', linewidth=0.4,
                      label='vs Dense-wide')

ax_delta.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax_delta.set_xticks(x2)
ax_delta.set_xticklabels(DATASET_LABELS, fontsize=6.5, rotation=20, ha='right')
ax_delta.set_ylabel('MoE-GCN improvement (%)\n(positive = MoE-GCN better)', fontsize=6.5)
ax_delta.set_title('MoE routing improvement\nover parameter-matched baselines', fontsize=7.5, pad=4)
ax_delta.spines['top'].set_visible(False)
ax_delta.spines['right'].set_visible(False)
ax_delta.legend(fontsize=5.5, loc='upper right',
                framealpha=0.8, edgecolor='none')

# Annotate delta values
for i, (du, dw) in enumerate(zip(delta_uniform, delta_wide)):
    col_u = '#d32f2f' if du < 0 else '#1565C0'
    col_w = '#6a1b9a' if dw < 0 else '#4a148c'
    offset_u = 0.15 if du >= 0 else -0.4
    offset_w = 0.15 if dw >= 0 else -0.4
    ax_delta.text(i - w2/2, du + offset_u, f'{du:+.1f}%',
                  ha='center', va='bottom', fontsize=4.8,
                  color=col_u, fontweight='bold')
    ax_delta.text(i + w2/2, dw + offset_w, f'{dw:+.1f}%',
                  ha='center', va='bottom', fontsize=4.8,
                  color=col_w, fontweight='bold')

# Panel B label
ax_delta.text(-0.13, 1.05, 'B', transform=ax_delta.transAxes,
              fontsize=9, fontweight='bold', va='top')

# Parameter inset on Panel B
if all(v is not None for v in params.values()):
    ax_ins = ax_delta.inset_axes([0.02, 0.55, 0.42, 0.38])
    p_vals = [params[m] / 1000 for m in MODELS]
    p_bars = ax_ins.bar(range(3), p_vals,
                        color=COLORS, alpha=0.75,
                        edgecolor='white', linewidth=0.3)
    ax_ins.set_xticks(range(3))
    ax_ins.set_xticklabels(['MoE', 'Unif', 'Wide'], fontsize=4.5)
    ax_ins.set_ylabel('Params (K)', fontsize=4.5)
    ax_ins.set_title('Parameters', fontsize=5, pad=2)
    ax_ins.tick_params(axis='both', labelsize=4.5, length=2)
    ax_ins.spines['top'].set_visible(False)
    ax_ins.spines['right'].set_visible(False)
    for bar, val in zip(p_bars, p_vals):
        ax_ins.text(bar.get_x() + bar.get_width()/2, val + 5,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=4)

# ── Save ───────────────────────────────────────────────────────────────────────

out = 'fig2_ablation.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')

out_pdf = 'fig2_ablation.pdf'
fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_pdf}')

plt.close()
print('Done.')
