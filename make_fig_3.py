"""
Fig 3 | Expert routing vs physicochemical descriptors
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

def load_dataset(tab_path, routing_path):
    df = pd.read_csv(tab_path, sep="\t")
    routing = np.load(routing_path)
    n = min(len(df), len(routing))
    df = df.iloc[:n].copy()
    routing = routing[:n]
    logp, mw, tpsa = [], [], []
    for smi in df["Drug"]:
        try:
            mol = Chem.MolFromSmiles(smi)
            logp.append(Descriptors.MolLogP(mol))
            mw.append(Descriptors.MolWt(mol))
            tpsa.append(Descriptors.TPSA(mol))
        except:
            logp.append(None); mw.append(None); tpsa.append(None)
    df["LogP"]   = logp
    df["MW"]     = mw
    df["TPSA"]   = tpsa
    df["expert"] = routing
    df = df.dropna(subset=["LogP","MW","TPSA"])
    return df

df_lipo = load_dataset(
    "data/lipophilicity_astrazeneca.tab",
    "routing_lipophilicity_astrazeneca_seed1_v2.npy")

df_sol = load_dataset(
    "data/solubility_aqsoldb.tab",
    "routing_solubility_aqsoldb_seed1_v2.npy")

# clip outliers
df_sol = df_sol[df_sol["LogP"].between(-10, 15)]
df_sol = df_sol[df_sol["MW"] < 1000]

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

def scatter_panel(ax, df, xcol, xlabel, title, show_legend=False):
    experts = sorted(df["expert"].unique())
    cmap = {e: COLORS[i] for i, e in enumerate(experts)}
    for e in experts:
        sub = df[df["expert"] == e]
        ax.scatter(sub[xcol], sub["Y"], c=cmap[e], s=5, alpha=0.4,
                   label=f"E{e} (n={len(sub)})")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(fontsize=6, markerscale=2, frameon=False)

def violin_panel(ax, df, title):
    experts = sorted(df["expert"].unique())
    df_clean = df[df["LogP"].between(-5, 8)]
    data = [df_clean[df_clean["expert"]==e]["LogP"].values for e in experts]
    parts = ax.violinplot(data, positions=list(range(len(experts))),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels([f"E{e}" for e in experts], fontsize=7)
    ax.set_xlabel("Expert", fontsize=8)
    ax.set_ylabel("LogP", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Row 1 – Lipophilicity
scatter_panel(axes[0,0], df_lipo, "LogP", "LogP",
              "A  Lipo-AZ: routing vs LogP", show_legend=True)
scatter_panel(axes[0,1], df_lipo, "MW", "MW (Da)",
              "B  Lipo-AZ: routing vs MW")
violin_panel(axes[0,2], df_lipo, "C  Lipo-AZ: LogP per expert")

# Row 2 – Solubility replication
scatter_panel(axes[1,0], df_sol, "LogP", "LogP",
              "D  Solubility: routing vs LogP", show_legend=True)
scatter_panel(axes[1,1], df_sol, "MW", "MW (Da)",
              "E  Solubility: routing vs MW")
violin_panel(axes[1,2], df_sol, "F  Solubility: LogP per expert")

fig.suptitle("Figure 3 | Expert routing partitions chemical space (seed 1)",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("fig3_final.png", dpi=300, bbox_inches="tight")
plt.savefig("fig3_final.pdf", bbox_inches="tight")
print("Saved: fig3_final.png / fig3_final.pdf")