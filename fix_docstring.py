path = 'ablation_routing.py'
content = open(path, encoding='utf-8').read()

old = 'MoE-GCN outperforms both ablations across regression datasets'
new = 'MoE-GCN achieves comparable performance to both equal-parameter ablations across regression datasets (no configuration dominates; see Results for significance test)'

assert old in content, 'string not found — check exact wording'
content = content.replace(old, new)

open(path, 'w', encoding='utf-8').write(content)
print('Fixed line 472.')