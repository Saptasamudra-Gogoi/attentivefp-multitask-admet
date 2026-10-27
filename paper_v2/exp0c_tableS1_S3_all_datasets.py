"""
exp0c_tableS1_S3_all_datasets.py — recompute Table S1 (mutual information) and Table S3 (endpoint-descriptor
correlation vs specialization) from the v2 routing files, mean over seeds, for every dataset that has them.
Also gives the pooled Spearman rho with a CLUSTER bootstrap over datasets (pairs are nested in datasets, so a
plain P-value is anti-conservative) and the per-dataset rho.
Run (repo root):  python exp0c_tableS1_S3_all_datasets.py
Needs routing_{ds}_seed{0..4}_v2.npy and pytdc. Datasets with < 2 active experts in every seed are skipped.
Outputs: tableS1_S3.json, tableS3.csv
Offline test: --smiles_csv f.csv (columns smiles,y) --datasets NAME  (routing files must exist for NAME)
"""
import argparse, json, csv
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
import retrain_and_extract as R
from exp0_eta2_across_seeds import descriptors, analyse, DESC

def smiles_y(ds, offline=None):
    if offline:
        d = __import__("pandas").read_csv(offline); return d["smiles"].tolist(), d["y"].values.astype(float)
    P = R.BEST_PARAMS.get(ds)
    name = P["tdc_name"] if P else {"ames": "AMES", "bbb_martins": "BBB_Martins", "dili": "DILI", "herg": "hERG"}[ds]
    sp = R.load_tdc(name)
    S = [x for k in ("train", "valid", "test") for x in sp[k]["Drug"]]; Y = np.concatenate([sp[k]["Y"].values for k in ("train", "valid", "test")])
    return S, Y.astype(float)

def mi(lab, v, bins=10):
    mk = ~np.isnan(v); q = np.unique(np.quantile(v[mk], np.linspace(0, 1, bins + 1)))
    return float(mutual_info_score(lab[mk], np.digitize(v[mk], q[1:-1])))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+"); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smiles_csv"); a = ap.parse_args()
    import glob, re
    dss = a.datasets or sorted({re.match(r"routing_(.+)_seed0_v2\.npy", f).group(1) for f in glob.glob("routing_*_seed0_v2.npy") if "weights" not in f})
    out = {}; pairs = []
    for ds in dss:
        try: smi, y = smiles_y(ds, a.smiles_csv)
        except Exception as e: print("skip", ds, e); continue
        Dd, ok = descriptors(smi); labs = [np.load(f"routing_{ds}_seed{s}_v2.npy") for s in a.seeds]
        if len(labs[0]) != len(smi): print("size mismatch, skip", ds); continue
        res = [analyse(l, Dd, ok) for l in labs]; good = [r for r in res if r]
        if not good: print("collapsed, skip", ds); continue
        mis = {d: [] for d in DESC}
        for l in labs:
            for d in DESC: mis[d].append(mi(l[ok], Dd[d][ok]))
        row = {}
        for d in DESC:
            e2 = float(np.mean([r["stats"][d]["eta2"] for r in good if d in r["stats"]]))
            v = Dd[d]; mk = ok & ~np.isnan(v) & ~np.isnan(y)
            r_ = abs(float(np.corrcoef(v[mk], y[mk])[0, 1])) if v[mk].std() > 0 and y[mk].std() > 0 else np.nan
            row[d] = dict(eta2=e2, abs_r=r_, mi=float(np.mean(mis[d])))
            pairs.append((ds, d, r_, e2))
        out[ds] = dict(n=len(smi), active_per_seed=[r["n_groups"] if r else 1 for r in res], desc=row)
        print(ds, "active", out[ds]["active_per_seed"])
    P = [(r, e) for _, _, r, e in pairs if not np.isnan(r)]
    rho = spearmanr([p[0] for p in P], [p[1] for p in P])[0]
    names = sorted(out); rng = np.random.default_rng(0); boots = []
    by = {n: [(p[2], p[3]) for p in pairs if p[0] == n and not np.isnan(p[2])] for n in names}
    for _ in range(2000):
        pick = rng.choice(names, len(names)); xs = [q for n in pick for q in by[n]]
        if len(xs) > 5: boots.append(spearmanr([x[0] for x in xs], [x[1] for x in xs])[0])
    per = {n: float(spearmanr(*zip(*by[n]))[0]) for n in names if len(by[n]) > 3}
    summ = dict(n_datasets=len(names), n_pairs=len(P), pooled_rho=float(rho), cluster_boot_ci95=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                per_dataset_rho_median=float(np.median(list(per.values()))), per_dataset_rho_positive=int(sum(v > 0 for v in per.values())), per_dataset_rho=per)
    json.dump(dict(datasets=out, summary=summ), open("tableS1_S3.json", "w"), indent=1)
    with open("tableS3.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "descriptor", "abs_endpoint_corr", "eta2_mean_over_seeds", "mutual_info"])
        for n in names:
            for d, v in out[n]["desc"].items(): w.writerow([n, d, v["abs_r"], v["eta2"], v["mi"]])
    print(json.dumps(summ, indent=1))
if __name__ == "__main__": main()
