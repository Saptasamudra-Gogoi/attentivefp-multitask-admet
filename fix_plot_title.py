path = 'ablation_routing.py'
content = open(path, encoding='utf-8').read()

old_title = """        ax.set_title('Ablation: MoE Routing vs Equal-Parameter Baselines\\n'
                     '* confirms gain from routing, not parameter count',
                     fontsize=11, fontweight='bold')"""

new_title = """        ax.set_title('Ablation: MoE Routing vs Equal-Parameter Baselines\\n'
                     '(comparable performance; routing adds interpretability, not accuracy)',
                     fontsize=11, fontweight='bold')"""

assert old_title in content, "exact title block not found — check whitespace/quoting"
content = content.replace(old_title, new_title)
open(path, 'w', encoding='utf-8').write(content)
print("Plot title fixed.")