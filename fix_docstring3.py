path = 'ablation_routing.py'
content = open(path, encoding='utf-8').read()

old_block = """  MoE-GCN achieves comparable performance to both equal-parameter ablations across regression datasets (no configuration dominates; see Results for significance test)
  (Table X), confirming that the performance gain is attributable
  to the routing mechanism itself — the ability to selectively
  activate experts based on molecular chemical space position —
  rather than the additional parameters introduced by the expert
  networks."""

new_block = """  MoE-GCN achieves comparable performance to both equal-parameter
  ablations across regression datasets (Table X; paired significance
  test in Results, p-values reported) — no configuration dominates
  outright. This indicates the routing mechanism does not provide a
  raw accuracy advantage over parameter-matched dense baselines.
  Instead, routing's value lies in interpretable expert specialization
  (Fig. 2), which the dense ablations cannot provide by construction:
  Dense-uniform and Dense-wide match MoE-GCN's total parameter count
  but lack any mechanism to route inputs to specialized subnetworks."""

assert old_block in content, "exact block not found — check for hidden whitespace/encoding differences"
content = content.replace(old_block, new_block)

open(path, 'w', encoding='utf-8').write(content)
print("Docstring rewritten — lines 471-476 replaced.")