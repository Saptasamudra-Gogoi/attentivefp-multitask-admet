"""
Print the core logic of the scripts behind the k-means 15/16 result and the
Spearman r=0.019 result, so we document what was ACTUALLY computed rather
than assume.

Run from D:\\molprop_project\\ with moe_admet activated.
"""
import os

FILES_TO_INSPECT = [
    "kmeans_control.py",
    "run_kmeans_control_all.py",
    "summarize_kmeans_comparison.py",
    "sss_v2.py",
    "sss_label_variance.py",
    "routing_entropy_sss.py",
]

for f in FILES_TO_INSPECT:
    print("=" * 80)
    print(f)
    print("=" * 80)
    if not os.path.exists(f):
        print("  NOT FOUND")
        continue
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    print(content)
    print()
