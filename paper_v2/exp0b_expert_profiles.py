"""
exp0b_expert_profiles.py  — Table 4 and Fig. 3B from ONE named seed of the v2 routing files.
Per active expert (>= --min_n molecules): n, mean MW/LogP/TPSA/HBD/HBA/ArRings, and z-score of each mean
against the whole dataset (so the heatmap shows what makes an expert different).
Run (repo root):  python exp0b_expert_profiles.py --dataset solubility_aqsoldb --seed 0
Offline test:     --smiles_csv f.csv  (column 'smiles', same row order as the routing file)
Writes profiles_{dataset}_seed{seed}.json / .md / .png.  Name the seed in the Table 4 caption.
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
DESC = {"MW": Descriptors.ExactMolWt, "LogP": Descriptors.MolLogP, "TPSA": Descriptors.TPSA, "HBD": rdMolDescriptors.CalcNumHBD,
        "HBA": rdMolDescriptors.CalcNumHBA, "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "RingCount": rdMolDescriptors.CalcNumRings, "ArRings": rdMolDescriptors.CalcNumAromaticRings}
TDC = {"solubility_aqsoldb": "Solubility_AqSolDB", "caco2_wang": "Caco2_Wang", "ld50_zhu": "LD50_Zhu",
       "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca"}

def smiles_for(ds):
    from tdc.single_pred import ADME, Tox
    for C in (ADME, Tox):
        try:
            sp = C(name=TDC[ds]).get_split(method="scaffold", seed=42)
            return list(sp["train"]["Drug"]) + list(sp["valid"]["Drug"]) + list(sp["test"]["Drug"])
        except Exception: pass
    raise RuntimeError(ds)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="solubility_aqsoldb"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_n", type=int, default=30); ap.add_argument("--smiles_csv"); ap.add_argument("--routing"); a = ap.parse_args()
    smi = __import__("pandas").read_csv(a.smiles_csv)["smiles"].tolist() if a.smiles_csv else smiles_for(a.dataset)
    lab = np.load(a.routing or f"routing_{a.dataset}_seed{a.seed}_v2.npy"); assert len(lab) == len(smi), (len(lab), len(smi))
    V = {k: [] for k in DESC}; ok = []
    for s in smi:
        m = Chem.MolFromSmiles(str(s)); ok.append(m is not None)
        for k, f in DESC.items(): V[k].append(float(f(m)) if m else np.nan)
    ok = np.array(ok); V = {k: np.array(v) for k, v in V.items()}
    ids = [e for e in np.unique(lab[ok]) if (lab[ok] == e).sum() >= a.min_n]
    ids.sort(key=lambda e: -(lab == e).sum()); rows = []
    for e in ids:
        mk = (lab == e) & ok; r = dict(expert=int(e), n=int(mk.sum()))
        for k in DESC: r[k] = float(np.nanmean(V[k][mk])); r["z_" + k] = float((r[k] - np.nanmean(V[k][ok])) / np.nanstd(V[k][ok]))
        rows.append(r)
    tag = f"{a.dataset}_seed{a.seed}"; json.dump(rows, open(f"profiles_{tag}.json", "w"), indent=1)
    cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "ArRings"]
    md = [f"Dataset {a.dataset}, seed {a.seed}, active experts (n >= {a.min_n}); all {len(lab)} molecules, {len(ids)} experts shown.", "",
          "| Expert | n | " + " | ".join(cols) + " | Largest |z| |", "|---|---|" + "---|" * (len(cols) + 1)]
    for r in rows:
        top = max(DESC, key=lambda k: abs(r["z_" + k]))
        md.append(f"| E{r['expert']} | {r['n']:,} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + f" | {top} ({r['z_'+top]:+.2f}) |")
    open(f"profiles_{tag}.md", "w", encoding="utf-8").write("\n".join(md)); print("\n".join(md))
    Z = np.array([[r["z_" + k] for k in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(rows) + 1.6), dpi=300)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]): ax.text(j, i, f"{rows[i][cols[j]]:.1f}", ha="center", va="center", fontsize=8, color="#0b0b0b" if abs(Z[i, j]) < 1.3 else "#fff")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"E{r['expert']} (n={r['n']:,})" for r in rows], fontsize=8)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.set_label("z-score of expert mean vs all molecules"); cb.outline.set_visible(False)
    ax.set_title(f"Fig. 3B | Per-expert descriptor means, {a.dataset}, seed {a.seed}\n(cell text = raw mean; colour = z-score)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(f"Fig3B_profiles_{tag}.png"); print("saved Fig3B_profiles_" + tag + ".png")
if __name__ == "__main__": main()
