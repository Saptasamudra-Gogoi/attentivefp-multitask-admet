"""
Chemical Diversity Analysis — MoE-GCN
======================================
Hypothesis: Chemical space heterogeneity of a dataset predicts
whether MoE routing will help or hurt performance.

Run from D:\\molprop_project:
    python chemical_diversity_analysis.py

Outputs:
    diversity_results.json
    diversity_vs_gain_silhouette.png
    diversity_vs_gain_logpstd.png
    diversity_vs_gain_tanimoto.png
    diversity_all_metrics.png
    diversity_summary.md
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET REGISTRY
#     moe_gain positive  = MoE wins
#     For RMSE/MAE (lower better):  gain = (gcn - moe) / gcn * 100
#     For Spearman/AUROC (higher):  gain = (moe - gcn) / gcn * 100
# ══════════════════════════════════════════════════════════════════════════════

TDC_ROOT = os.path.join("tdc_data", "admet_group")

# folder name inside admet_group  →  dataset label
TDC_FOLDER_MAP = {
    "caco2_wang":                    "Caco2_Wang",
    "lipophilicity_astrazeneca":     "Lipophilicity_AZ",
    "solubility_aqsoldb":            "Solubility_AqSolDB",
    "ppbr_az":                       "PPBR_AZ",
    "ld50_zhu":                      "LD50_Zhu",
    "vdss_lombardo":                 "VDss_Lombardo",
    "half_life_obach":               "Half_Life_Obach",
    "clearance_microsome_az":        "CL_Microsome",
    "clearance_hepatocyte_az":       "CL_Hepatocyte",
    "hia_hou":                       "HIA_Hou",
    "pgp_broccatelli":               "Pgp_Broccatelli",
    "bioavailability_ma":            "Bioavailability_Ma",
    "bbb_martins":                   "BBB_Martins",
    "cyp2d6_veith":                  "CYP2D6_Veith",
    "cyp3a4_veith":                  "CYP3A4_Veith",
    "cyp2d6_substrate_carbonmangels":"CYP2D6_Substrate",
    "cyp3a4_substrate_carbonmangels":"CYP3A4_Substrate",
    "cyp2c9_substrate_carbonmangels":"CYP2C9_Substrate",
    "herg":                          "hERG",
    "ames":                          "AMES",
    "dili":                          "DILI",
}

# MoE results from tech report — (gcn, moe, higher_better, task_type)
RESULTS = {
    # MoleculeNet (no folder — will use DeepChem/MoleculeNet CSV if available)
    "ESOL":             (1.3236, 1.067,  False, "MAE_reg"),
    "FreeSolv":         (4.9266, 3.591,  False, "MAE_reg"),
    "Lipophilicity_MN": (0.8120, 0.722,  False, "MAE_reg"),
    # TDC MAE regression
    "Caco2_Wang":       (0.4605, 0.366,  False, "MAE_reg"),
    "Lipophilicity_AZ": (0.5942, 0.542,  False, "MAE_reg"),
    "Solubility_AqSolDB":(1.0478,0.956,  False, "MAE_reg"),
    "PPBR_AZ":          (9.7396, 9.614,  False, "MAE_reg"),
    "LD50_Zhu":         (0.6968, 0.664,  False, "MAE_reg"),
    # TDC Spearman
    "VDss_Lombardo":    (0.3844, 0.409,  True,  "Spearman_reg"),
    "Half_Life_Obach":  (0.3124, 0.198,  True,  "Spearman_reg"),
    "CL_Microsome":     (0.4739, 0.535,  True,  "Spearman_reg"),
    "CL_Hepatocyte":    (0.3817, 0.338,  True,  "Spearman_reg"),
    # TDC Classification
    "HIA_Hou":          (0.9348, 0.946,  True,  "clf"),
    "Pgp_Broccatelli":  (0.8885, 0.894,  True,  "clf"),
    "Bioavailability_Ma":(0.5972,0.622,  True,  "clf"),
    "BBB_Martins":      (0.8587, 0.863,  True,  "clf"),
    "CYP2D6_Veith":     (0.8534, 0.838,  True,  "clf"),
    "CYP3A4_Veith":     (0.8739, 0.890,  True,  "clf"),
    "CYP2D6_Substrate": (0.8023, 0.786,  True,  "clf"),
    "CYP3A4_Substrate": (0.5900, 0.579,  True,  "clf"),
    "CYP2C9_Substrate": (0.6260, 0.604,  True,  "clf"),
    "hERG":             (0.7358, 0.709,  True,  "clf"),
    "AMES":             (0.8410, 0.838,  True,  "clf"),
    "DILI":             (0.9494, 0.900,  True,  "clf"),
}

def moe_gain(gcn, moe, higher_better):
    if higher_better:
        return (moe - gcn) / abs(gcn) * 100
    else:
        return (gcn - moe) / abs(gcn) * 100

# ══════════════════════════════════════════════════════════════════════════════
# 2.  SMILES LOADING — direct CSV read from admet_group folders
# ══════════════════════════════════════════════════════════════════════════════

def load_smiles_from_folder(folder_name):
    folder = os.path.join(TDC_ROOT, folder_name)
    smiles_all = []
    for fname in ["train_val.csv", "test.csv"]:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            col = "Drug" if "Drug" in df.columns else df.columns[1]
            smiles_all.extend(df[col].dropna().tolist())
    return smiles_all

# Reverse map: label → folder
LABEL_TO_FOLDER = {v: k for k, v in TDC_FOLDER_MAP.items()}

def get_smiles(label):
    folder = LABEL_TO_FOLDER.get(label)
    if folder:
        return load_smiles_from_folder(folder)
    # MoleculeNet: try common CSV locations
    for path in [
        f"data/{label.lower()}.csv",
        f"data/{label}.csv",
        f"{label.lower()}.csv",
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in ["smiles", "SMILES", "Drug"]:
                if col in df.columns:
                    return df[col].dropna().tolist()
    return []

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DESCRIPTOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_descriptors(smiles_list):
    rows, valid = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            rows.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAromaticRings(mol),
            ])
            valid.append(smi)
        except Exception:
            continue
    if not rows:
        return 0, np.array([]), []
    return len(rows), np.array(rows, dtype=float), valid

# ══════════════════════════════════════════════════════════════════════════════
# 4.  DIVERSITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def logp_std(desc_matrix):
    if desc_matrix.shape[0] == 0:
        return np.nan
    return float(np.std(desc_matrix[:, 1]))

def scaffold_diversity(smiles_list):
    scaffolds, valid = set(), 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.add(sc)
            valid += 1
        except Exception:
            continue
    return len(scaffolds) / valid if valid > 0 else np.nan

def silhouette_k5(desc_matrix, k=5):
    if desc_matrix.shape[0] < 50:
        return np.nan
    scaler = StandardScaler()
    X = scaler.fit_transform(desc_matrix)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    if X.shape[0] < 50:
        return np.nan
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(X, labels,
                                  sample_size=min(2000, X.shape[0]),
                                  random_state=42))

def tanimoto_diversity(smiles_list, n_sample=300, radius=2, n_bits=1024):
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 10:
        return np.nan
    rng = np.random.RandomState(42)
    if len(mols) > n_sample:
        idx = rng.choice(len(mols), n_sample, replace=False)
        mols = [mols[i] for i in idx]
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.array(fps)
    n = len(fps)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.dot(fps[i], fps[j])
            union = fps[i].sum() + fps[j].sum() - inter
            total += (1.0 - inter / union) if union > 0 else 0.0
            count += 1
    return float(total / count) if count > 0 else np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("=" * 65)
    print("  Chemical Diversity -> MoE Utility Analysis")
    print("=" * 65)
    records = []

    for label, (gcn, moe, higher, task_type) in RESULTS.items():
        gain = moe_gain(gcn, moe, higher)
        print(f"\n[{label}]  MoE gain: {gain:+.1f}%")

        smiles = get_smiles(label)
        print(f"  Loaded {len(smiles)} SMILES")

        if len(smiles) < 20:
            print("  [SKIP] no SMILES found")
            records.append({
                "dataset": label, "task_type": task_type,
                "n_molecules": 0, "moe_gain_pct": gain,
                "gcn": gcn, "moe": moe,
                "logp_std": np.nan, "scaffold_diversity": np.nan,
                "silhouette_k5": np.nan, "tanimoto_diversity": np.nan,
            })
            continue

        n, desc, valid_smi = compute_descriptors(smiles)
        print(f"  Valid molecules: {n}")

        lp  = logp_std(desc)
        sc  = scaffold_diversity(valid_smi)
        sil = silhouette_k5(desc)
        tan = tanimoto_diversity(valid_smi)

        print(f"  LogP std:           {lp:.3f}")
        print(f"  Scaffold diversity: {sc:.3f}")
        print(f"  Silhouette (k=5):   {sil:.3f}")
        print(f"  Tanimoto diversity: {tan:.3f}")

        records.append({
            "dataset": label, "task_type": task_type,
            "n_molecules": n, "moe_gain_pct": gain,
            "gcn": gcn, "moe": moe,
            "logp_std": lp, "scaffold_diversity": sc,
            "silhouette_k5": sil, "tanimoto_diversity": tan,
        })

    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_correlations(df):
    print("\n" + "=" * 65)
    print("  SPEARMAN CORRELATIONS: Diversity -> MoE Gain")
    print("=" * 65)
    metrics = ["logp_std", "scaffold_diversity", "silhouette_k5", "tanimoto_diversity"]
    corr = {}

    for scope, subset in [("All datasets", df),
                           ("Regression only", df[df["task_type"].isin(["MAE_reg","Spearman_reg"])])]:
        print(f"\n  [{scope}]")
        for m in metrics:
            sub = subset[["moe_gain_pct", m]].dropna()
            if len(sub) < 4:
                print(f"  {m:28s}: insufficient data (n={len(sub)})")
                continue
            rho, p = stats.spearmanr(sub["moe_gain_pct"], sub[m])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {m:28s}: rho={rho:+.3f}  p={p:.4f}  n={len(sub)}  {sig}")
            if scope == "All datasets":
                corr[m] = {"rho": rho, "p": p, "n": len(sub)}
    return corr

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {"MAE_reg": "#2196F3", "Spearman_reg": "#FF9800", "clf": "#9E9E9E"}

def scatter_plot(df, x_col, xlabel, outpath):
    sub = df[["dataset", "task_type", x_col, "moe_gain_pct"]].dropna()
    if sub.empty:
        print(f"[SKIP plot] no data for {x_col}")
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for task, grp in sub.groupby("task_type"):
        c = COLORS.get(task, "black")
        ax.scatter(grp[x_col], grp["moe_gain_pct"], color=c, s=80,
                   label=task.replace("_"," "), zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row["dataset"], (row[x_col], row["moe_gain_pct"]),
                        fontsize=7, xytext=(4,4), textcoords="offset points", color=c)
    reg = sub[sub["task_type"].isin(["MAE_reg","Spearman_reg"])]
    if len(reg) >= 4:
        rho, p = stats.spearmanr(reg["moe_gain_pct"], reg[x_col])
        m, b, r, _, _ = stats.linregress(reg[x_col], reg["moe_gain_pct"])
        xs = np.linspace(reg[x_col].min(), reg[x_col].max(), 100)
        ax.plot(xs, m*xs+b, "k--", lw=1.2, alpha=0.6,
                label=f"Regression trend (rho={rho:+.2f}, p={p:.3f})")
    ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("MoE Gain (%)  [positive = MoE wins]", fontsize=12)
    ax.set_title(f"{xlabel} vs MoE Performance Gain\n(25 datasets: 3 MolNet + 22 TDC)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {outpath}")
    plt.close()

def panel_plot(df):
    pairs = [
        ("logp_std",          "LogP Std"),
        ("scaffold_diversity","Scaffold Diversity"),
        ("silhouette_k5",     "Silhouette Score (k=5)"),
        ("tanimoto_diversity","Tanimoto Diversity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, pairs):
        sub = df[["dataset","task_type",col,"moe_gain_pct"]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        for task, grp in sub.groupby("task_type"):
            c = COLORS.get(task, "black")
            ax.scatter(grp[col], grp["moe_gain_pct"], color=c, s=55, zorder=3)
            for _, row in grp.iterrows():
                ax.annotate(row["dataset"], (row[col], row["moe_gain_pct"]),
                            fontsize=6, xytext=(3,3), textcoords="offset points", color=c)
        reg = sub[sub["task_type"].isin(["MAE_reg","Spearman_reg"])]
        title = label
        if len(reg) >= 4:
            rho, p = stats.spearmanr(reg["moe_gain_pct"], reg[col])
            m, b, _, _, _ = stats.linregress(reg[col], reg["moe_gain_pct"])
            xs = np.linspace(reg[col].min(), reg[col].max(), 100)
            ax.plot(xs, m*xs+b, "k--", lw=1.0, alpha=0.6)
            title = f"{label}\nrho={rho:+.2f} p={p:.3f} (regression)"
        ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("MoE Gain (%)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
    patches = [mpatches.Patch(color=c, label=t.replace("_"," "))
               for t, c in COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Chemical Diversity Metrics vs MoE Performance Gain", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("diversity_all_metrics.png", dpi=150, bbox_inches="tight")
    print("[SAVED] diversity_all_metrics.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df, corr):
    df.to_json("diversity_results.json", orient="records", indent=2)
    print("[SAVED] diversity_results.json")

    lines = [
        "# Chemical Diversity Analysis -- Summary\n",
        "## Per-Dataset Results\n",
        "| Dataset | N | MoE Gain% | LogP Std | Scaffold Div | Silhouette | Tanimoto | Task |",
        "|---------|---|-----------|----------|--------------|------------|----------|------|",
    ]
    for _, row in df.sort_values("moe_gain_pct", ascending=False).iterrows():
        def f(v): return f"{v:.3f}" if pd.notna(v) else "-"
        n = int(row["n_molecules"]) if pd.notna(row["n_molecules"]) and row["n_molecules"] > 0 else "-"
        lines.append(
            f"| {row['dataset']} | {n} | {row['moe_gain_pct']:+.1f}% "
            f"| {f(row['logp_std'])} | {f(row['scaffold_diversity'])} "
            f"| {f(row['silhouette_k5'])} | {f(row['tanimoto_diversity'])} "
            f"| {row['task_type']} |"
        )
    lines += [
        "\n## Spearman Correlations (Diversity -> MoE Gain)\n",
        "| Metric | rho | p-value | n | Sig |",
        "|--------|-----|---------|---|-----|",
    ]
    for m, r in corr.items():
        sig = "***" if r["p"]<0.001 else "**" if r["p"]<0.01 else "*" if r["p"]<0.05 else "n.s."
        lines.append(f"| {m} | {r['rho']:+.3f} | {r['p']:.4f} | {r['n']} | {sig} |")

    with open("diversity_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[SAVED] diversity_summary.md")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    df = run_analysis()

    print("\n\n" + "=" * 65)
    print("  RESULTS TABLE")
    print("=" * 65)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(df[["dataset","task_type","n_molecules","moe_gain_pct",
              "logp_std","scaffold_diversity","silhouette_k5","tanimoto_diversity"]
             ].to_string(index=False))

    corr = run_correlations(df)

    scatter_plot(df, "silhouette_k5",     "Silhouette Score (k=5)",  "diversity_vs_gain_silhouette.png")
    scatter_plot(df, "logp_std",          "LogP Std",                "diversity_vs_gain_logpstd.png")
    scatter_plot(df, "tanimoto_diversity","Tanimoto Diversity",       "diversity_vs_gain_tanimoto.png")
    scatter_plot(df, "scaffold_diversity","Scaffold Diversity",       "diversity_vs_gain_scaffold.png")
    panel_plot(df)

    save_outputs(df, corr)

    print("\n" + "=" * 65)
    print("  DONE")
    print("  KEY CHECK: Do Half_Life_Obach and PPBR_AZ have the")
    print("  LOWEST silhouette scores among regression datasets?")
    print("  If yes -- hypothesis holds. Paste results here.")
    print("=" * 65)
