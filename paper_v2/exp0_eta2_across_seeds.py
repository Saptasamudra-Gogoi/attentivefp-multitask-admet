"""
exp0_eta2_across_seeds.py
=========================
Recompute expert-specialization eta^2 for EVERY saved seed (routing_{ds}_seed{s}_v2.npy)
and report mean +/- SD across seeds, plus cross-seed partition stability (ARI/NMI).

Why: Table 5 (expert_specialization_SUMMARY.json) and Table S3 / p1_stats_results.json
give different eta^2 for the same dataset/descriptor (e.g. Solubility LogP 0.325 vs 0.290,
Lipophilicity LogP 0.051 vs 0.019). They come from different runs. This script produces ONE
reproducible number per cell from the v2 routing files, with seed variability.

Needs: rdkit, numpy, scipy, scikit-learn, pytdc (or --smiles_csv for offline use).
Run (from repo root):
    python exp0_eta2_across_seeds.py
    python exp0_eta2_across_seeds.py --datasets solubility_aqsoldb caco2_wang ld50_zhu lipophilicity_astrazeneca
Offline test:  --smiles_csv my.csv (column 'smiles'; row order must equal routing order)

Molecule order = scaffold split (seed=42) train+valid+test concatenated, exactly as in
extract_smiles_and_specialize.py / retrain_and_extract.py.
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import f_oneway, kruskal
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

TDC = {
    "solubility_aqsoldb": "Solubility_AqSolDB", "caco2_wang": "Caco2_Wang",
    "ld50_zhu": "LD50_Zhu", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
}
DESC = {
    "MW": Descriptors.ExactMolWt, "LogP": Descriptors.MolLogP,
    "HBA": rdMolDescriptors.CalcNumHBA, "HBD": rdMolDescriptors.CalcNumHBD,
    "TPSA": Descriptors.TPSA, "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings, "ArRings": rdMolDescriptors.CalcNumAromaticRings,
}


def load_smiles(tdc_name):
    from tdc.single_pred import ADME, Tox
    for Cls in (ADME, Tox):
        try:
            sp = Cls(name=tdc_name).get_split(method="scaffold", seed=42)
            return list(sp["train"]["Drug"]) + list(sp["valid"]["Drug"]) + list(sp["test"]["Drug"])
        except Exception:
            continue
    raise RuntimeError(f"cannot load {tdc_name}")


def descriptors(smiles):
    out = {k: [] for k in DESC}
    ok = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        ok.append(m is not None)
        for k, fn in DESC.items():
            out[k].append(float(fn(m)) if m is not None else np.nan)
    return {k: np.array(v) for k, v in out.items()}, np.array(ok)


def eta2_omega2(groups):
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    sst = ((allv - gm) ** 2).sum()
    k, n = len(groups), len(allv)
    msw = (sst - ssb) / (n - k)
    eta = ssb / sst if sst > 0 else 0.0
    omega = (ssb - (k - 1) * msw) / (sst + msw) if sst > 0 else 0.0
    return float(eta), float(omega)


def analyse(labels, D, ok, min_group=3):
    lab = labels[ok]
    res = {}
    ids = [e for e in np.unique(lab) if (lab == e).sum() >= min_group]
    if len(ids) < 2:
        return None
    for name, v in D.items():
        v = v[ok]; fin = ~np.isnan(v)
        groups = [v[(lab == e) & fin] for e in ids]
        groups = [g for g in groups if len(g) >= min_group]
        if len(groups) < 2:
            continue
        eta, omega = eta2_omega2(groups)
        res[name] = dict(eta2=eta, omega2=omega,
                         p_anova=float(f_oneway(*groups)[1]), p_kw=float(kruskal(*groups)[1]))
    return dict(n_groups=len(ids), stats=res)


def bh(p):
    p = np.asarray(p); o = np.argsort(p); r = np.empty_like(o); r[o] = np.arange(1, len(p) + 1)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(TDC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smiles_csv", default=None)
    ap.add_argument("--out", default="eta2_across_seeds.json")
    a = ap.parse_args()
    report = {}
    for ds in a.datasets:
        smi = (__import__("pandas").read_csv(a.smiles_csv)["smiles"].tolist()
               if a.smiles_csv else load_smiles(TDC[ds]))
        D, ok = descriptors(smi)
        labs = [np.load(f"routing_{ds}_seed{s}_v2.npy") for s in a.seeds]
        n = len(labs[0])
        assert len(smi) == n, f"{ds}: {len(smi)} SMILES vs {n} routing labels - ordering/size mismatch"
        per_seed = []
        for s, l in zip(a.seeds, labs):
            r = analyse(l, D, ok)
            per_seed.append(r)
        names = list(DESC)
        table = {}
        for d in names:
            vals = [r["stats"][d]["eta2"] for r in per_seed if r and d in r["stats"]]
            table[d] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            per_seed=vals, n_seeds=len(vals))
        # BH over all (descriptor x seed) tests for this dataset
        ps = [r["stats"][d]["p_anova"] for r in per_seed if r for d in r["stats"]]
        pairs = list(itertools.combinations(range(len(labs)), 2))
        ari = [adjusted_rand_score(labs[i], labs[j]) for i, j in pairs]
        nmi = [normalized_mutual_info_score(labs[i], labs[j]) for i, j in pairs]
        report[ds] = dict(n=n, eta2=table, max_bh_adj_p=float(bh(ps).max()) if ps else None,
                          ari_mean=float(np.mean(ari)), ari_min=float(np.min(ari)), nmi_mean=float(np.mean(nmi)),
                          active_groups_per_seed=[r["n_groups"] if r else 1 for r in per_seed])
        print(f"\n== {ds} (n={n}) | cross-seed ARI mean {np.mean(ari):.3f} (min {np.min(ari):.3f}), NMI {np.mean(nmi):.3f}")
        print(f"   active experts/seed: {report[ds]['active_groups_per_seed']}")
        for d in names:
            t = table[d]; print(f"   {d:9s} eta2 = {t['mean']:.3f} +/- {t['sd']:.3f}   seeds: {[round(x,3) for x in t['per_seed']]}")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")
    print("Use mean +/- SD across seeds in Table 5, Fig 3A and Table S3; delete any cell not reproduced here.")


if __name__ == "__main__":
    main()
