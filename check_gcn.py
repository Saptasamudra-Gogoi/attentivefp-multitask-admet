import json
import numpy as np

with open("baselines/gin_gcn_gat/baselines/results/gnn_results.json") as f:
    data = json.load(f)

from collections import defaultdict
gcn = defaultdict(list)
for row in data:
    if row["model"] == "GCN":
        gcn[row["dataset"]].append(row["test"])

for ds, vals in sorted(gcn.items()):
    print(f"{ds}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")