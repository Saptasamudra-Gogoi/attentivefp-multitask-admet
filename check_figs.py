from PIL import Image
import os

figs = [
    'fig1_performance.png',
    'fig2_3d_vs_2d.png',
    'fig3_expert_space.png',
    'fig4_cross_arch.png',
    'fig5_ablation.png',
    'expert_profiles_panel.png',
    'expert_heatmap.png',
    'expert_radar.png',
]

for f in figs:
    if os.path.exists(f):
        img = Image.open(f)
        print(f'{f}: {img.size} px')
    else:
        print(f'{f}: NOT FOUND')
