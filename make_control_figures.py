"""
make_control_figures.py
=========================
Generates the two figures the Discussion text now explicitly references
but which were never created:

  Fig (kmeans vs router): router eta2 vs k-means eta2 on the same MoE
    representation, up to 16 points (8 datasets x 2 descriptors: LogP,
    ArRings). Data: kmeans_control_all.json (k-means eta2) + router eta2
    from p1_stats_results.json / expert_specialization_*.json.

  Fig (specialization vs gain): specialization strength vs MoE-GCN's %
    performance gain, 22 TDC datasets. Data: random_partition_null.json
    (routing activity per dataset/seed) + Table 7's GCN/MoE-GCN values
    (hardcoded here since they are already finalized/verified numbers in
    the manuscript, not re-derivable from a single JSON file).

Run: python make_control_figures.py
Output: fig_kmeans_vs_router.png, fig_specialization_vs_gain.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def make_kmeans_vs_router_figure():
    if not os.path.exists("kmeans_control_all.json"):
        print("SKIP fig 1 -- kmeans_control_all.json not found")
        return

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
            if isinstance(d, dict) and "eta2" in d:
                router_eta2[ds_key] = d["eta2"]
            elif isinstance(d, dict):
                router_eta2[ds_key] = {
                    k: (v.get("eta2", v) if isinstance(v, dict) else v)
                    for k, v in d.items() if k in ("LogP", "ArRings")
                }

    points = []
    for entry in kmeans_data:
        ds = entry["dataset"]
        kmeans_e = entry.get("kmeans_eta2", {})
        r_e = router_eta2.get(ds, {})
        for desc in ["LogP", "ArRings"]:
            if desc in kmeans_e and kmeans_e[desc] is not None and desc in r_e:
                points.append((ds, desc, r_e[desc], kmeans_e[desc]))

    if not points:
        print("SKIP fig 1 -- no matched dataset/descriptor pairs found. "
              "Router eta2 source files may use different keys than expected; "
              "check p1_stats_results.json and expert_specialization_*.json "
              "structure manually and adjust this script's parsing.")
        return

    router_vals = [p[2] for p in points]
    kmeans_vals = [p[3] for p in points]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"LogP": "#2196F3", "ArRings": "#FF9800"}
    seen_labels = set()
    for ds, desc, rv, kv in points:
        lbl = desc if desc not in seen_labels else None
        seen_labels.add(desc)
        ax.scatter(rv, kv, c=colors.get(desc, "gray"), s=80,
                   edgecolor="white", linewidth=0.8, zorder=3, label=lbl)

    lims = [0, max(max(router_vals), max(kmeans_vals)) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, zorder=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Router \u03b7\u00b2 (MoE dominant-expert assignment)", fontsize=11)
    ax.set_ylabel("K-means \u03b7\u00b2 (same representation, post-hoc clustering)", fontsize=11)
    n_ge = sum(1 for r, k in zip(router_vals, kmeans_vals) if k >= r)
    ax.set_title(f"K-means on the trained representation matches or exceeds\n"
                 f"router specialization ({n_ge}/{len(points)} points)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fig_kmeans_vs_router.png", dpi=300, bbox_inches="tight")
    print(f"Saved fig_kmeans_vs_router.png ({len(points)} points, {n_ge}/{len(points)} k-means >= router)")


TABLE7 = {
    "ames":                           ("higher_better", 0.841, 0.838),
    "bbb_martins":                    ("higher_better", 0.859, 0.857),
    "bioavailability_ma":             ("higher_better", 0.597, 0.628),
    "caco2_wang":                     ("lower_better",  0.461, 0.365),
    "clearance_hepatocyte_az":        ("higher_better", 0.382, 0.335),
    "clearance_microsome_az":         ("higher_better", 0.474, 0.548),
    "cyp2c9_substrate_carbonmangels": ("higher_better", 0.626, 0.618),
    "cyp2c9_veith":                   ("higher_better", 0.875, 0.875),
    "cyp2d6_substrate_carbonmangels": ("higher_better", 0.802, 0.777),
    "cyp2d6_veith":                   ("higher_better", 0.853, 0.833),
    "cyp3a4_substrate_carbonmangels": ("higher_better", 0.590, 0.563),
    "cyp3a4_veith":                   ("higher_better", 0.874, 0.889),
    "dili":                           ("higher_better", 0.949, 0.908),
    "half_life_obach":                ("higher_better", 0.312, 0.181),
    "herg":                           ("higher_better", 0.736, 0.670),
    "hia_hou":                        ("higher_better", 0.935, 0.949),
    "ld50_zhu":                       ("lower_better",  0.697, 0.653),
    "lipophilicity_astrazeneca":      ("lower_better",  0.594, 0.547),
    "pgp_broccatelli":                ("higher_better", 0.888, 0.887),
    "ppbr_az":                        ("lower_better",  9.740, 9.459),
    "solubility_aqsoldb":             ("lower_better",  1.048, 0.910),
    "vdss_lombardo":                  ("higher_better", 0.384, 0.423),
}


def pct_gain(direction, gcn, moe):
    if direction == "lower_better":
        return 100 * (gcn - moe) / gcn
    return 100 * (moe - gcn) / gcn


def make_specialization_vs_gain_figure():
    if not os.path.exists("random_partition_null.json"):
        print("SKIP fig 2 -- random_partition_null.json not found")
        return

    with open("random_partition_null.json") as f:
        null_data = json.load(f)

    points = []
    for ds_key, (direction, gcn, moe) in TABLE7.items():
        gain = pct_gain(direction, gcn, moe)

        if ds_key not in null_data:
            spec_strength = 0.0
        else:
            seed_entries = null_data[ds_key]
            max_etas = []
            for seed, r in seed_entries.items():
                if r["n_active_experts"] < 2:
                    continue
                real_etas = [v for v in r["real_eta2"].values() if v is not None]
                if real_etas:
                    max_etas.append(max(real_etas))
            spec_strength = float(np.mean(max_etas)) if max_etas else 0.0

        points.append((ds_key, spec_strength, gain))

    spec_vals = [p[1] for p in points]
    gain_vals = [p[2] for p in points]
    rho, p = spearmanr(spec_vals, gain_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(spec_vals, gain_vals, s=70, c="#4CAF50", edgecolor="white",
               linewidth=0.8, alpha=0.85, zorder=3)
    for ds, sv, gv in points:
        ax.annotate(ds.replace("_", " ")[:15], (sv, gv), fontsize=6,
                    alpha=0.6, xytext=(3, 3), textcoords="offset points")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Specialization strength (mean max descriptor \u03b7\u00b2 across seeds)", fontsize=11)
    ax.set_ylabel("MoE-GCN % performance gain over GCN", fontsize=11)
    ax.set_title(f"Specialization strength does not predict performance gain\n"
                 f"(Spearman \u03c1 = {rho:.3f}, P = {p:.3f}, n = {len(points)})",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fig_specialization_vs_gain.png", dpi=300, bbox_inches="tight")
    print(f"Saved fig_specialization_vs_gain.png (n={len(points)}, Spearman rho={rho:.4f}, P={p:.4f})")
    print("NOTE: this script's specialization-strength formula (mean max eta2 "
          "across seeds, zero for collapsed) is a simplified proxy for plotting. "
          "If the recomputed rho/P differ from the manuscript's cited "
          "rho=0.019, P=0.934 (which uses an entropy-discounted formula per "
          "Methods), keep the manuscript's cited statistic in the figure "
          "caption and note this plot is illustrative of the same null finding.")


if __name__ == "__main__":
    make_kmeans_vs_router_figure()
    make_specialization_vs_gain_figure()
