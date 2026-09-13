import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("kmeans_control_all.json") as f:
    kmeans_data = json.load(f)

router_eta2 = {}
if os.path.exists("p1_stats_results.json"):
    with open("p1_stats_results.json") as f:
        p1 = json.load(f)
    for ds, descs in p1.items():
        router_eta2[ds] = {k: v["eta2"] for k, v in descs.items()}

for ds_key in ["bbb_martins", "herg", "ames", "dili"]:
    fpath = f"expert_specialization_{ds_key}.json"
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        stats = d.get("stats", {})
        router_eta2[ds_key] = {k: v["eta2"] for k, v in stats.items() if "eta2" in v}

points = []
for entry in kmeans_data:
    ds = entry["dataset"]
    kmeans_e = entry.get("kmeans_eta2", {})
    r_e = router_eta2.get(ds, {})
    for desc in ["LogP", "ArRings"]:
        if desc in kmeans_e and kmeans_e[desc] is not None and desc in r_e:
            points.append((ds, desc, r_e[desc], kmeans_e[desc]))

print(f"Found {len(points)} points")
for p in points:
    print(f"  {p[0]:<28} {p[1]:<8} router={p[2]:.4f}  kmeans={p[3]:.4f}")

router_vals = [p[2] for p in points]
kmeans_vals = [p[3] for p in points]
n_ge = sum(1 for r, k in zip(router_vals, kmeans_vals) if k >= r)

fig, ax = plt.subplots(figsize=(6, 6))
colors = {"LogP": "#2196F3", "ArRings": "#FF9800"}
seen = set()
for ds, desc, rv, kv in points:
    lbl = desc if desc not in seen else None
    seen.add(desc)
    ax.scatter(rv, kv, c=colors.get(desc, "gray"), s=80,
               edgecolor="white", linewidth=0.8, zorder=3, label=lbl)

lims = [0, max(max(router_vals), max(kmeans_vals)) * 1.1]
ax.plot(lims, lims, "k--", alpha=0.4, zorder=1, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Router \u03b7\u00b2 (MoE dominant-expert assignment)", fontsize=11)
ax.set_ylabel("K-means \u03b7\u00b2 (same representation, post-hoc clustering)", fontsize=11)
ax.set_title(f"K-means on the trained representation matches or exceeds\n"
             f"router specialization ({n_ge}/{len(points)} points)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("fig_kmeans_vs_router.png", dpi=300, bbox_inches="tight")
print(f"\nSaved fig_kmeans_vs_router.png ({len(points)} points, {n_ge}/{len(points)} k-means >= router)")
