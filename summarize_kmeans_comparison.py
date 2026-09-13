# summarize_kmeans_comparison.py
import json

with open("kmeans_control_all.json") as f:
    kmeans = {r["dataset"]: r["kmeans_eta2"] for r in json.load(f)}

# Real MoE eta2 values gathered earlier (from expert_specialization_*.json files)
real_moe = {
    "caco2_wang": {"LogP": 0.1508, "ArRings": 0.3247},
    "solubility_aqsoldb": {"LogP": 0.3254, "ArRings": 0.1024},
    "lipophilicity_astrazeneca": {"LogP": 0.0507, "ArRings": 0.0930},
    "ld50_zhu": {"LogP": 0.1424, "ArRings": 0.4448},
    "bbb_martins": {"LogP": 0.2138, "ArRings": 0.2624},
    "herg": {"LogP": 0.0734, "ArRings": 0.3796},
    "ames": {"LogP": 0.2239, "ArRings": 0.5643},
    "dili": {"LogP": 0.0458, "ArRings": 0.2188},
}

wins_kmeans, wins_moe = 0, 0
print(f"{'Dataset':<28}{'Descr':<10}{'RealMoE':>10}{'KMeans':>10}{'Winner':>10}")
for ds, moe_vals in real_moe.items():
    km_vals = kmeans.get(ds, {})
    for desc in ["LogP", "ArRings"]:
        m, k = moe_vals[desc], km_vals.get(desc)
        winner = "KMeans" if k > m else "RealMoE"
        if winner == "KMeans": wins_kmeans += 1
        else: wins_moe += 1
        print(f"{ds:<28}{desc:<10}{m:>10.4f}{k:>10.4f}{winner:>10}")

print(f"\nKMeans wins: {wins_kmeans}/16   RealMoE wins: {wins_moe}/16")