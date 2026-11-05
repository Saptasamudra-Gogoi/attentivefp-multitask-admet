#!/usr/bin/env python
"""
randomized_scaffold.py - randomized Bemis-Murcko scaffold splits (chemprop
`scaffold_balanced` logic) with saved, auditable indices.

Replaces two broken splitters in run_gcn_matched_classif.py:

  scaffold_split()            - deterministic, sorted by group size, and it checks
                                the cutoff BEFORE extending, so the final group can
                                overshoot train/val by its full size.
  stratified_scaffold_split() - moves minority-class scaffolds out of train until
                                val/test contain both classes. That selects the
                                split using the labels being predicted. Withdrawn.

Protocol here (chemprop scaffold_balanced):
  1. Group molecules by Murcko scaffold (includeChirality=False).
  2. Groups larger than half the val set or half the test set go to train first,
     so no single large scaffold can dominate the evaluation sets.
  3. Remaining groups are shuffled with the split seed.
  4. Greedy fill, checking the cutoff BEFORE adding (no overshoot).

Each split is saved as JSON with the indices, the scaffold count, the positive
rate per split, and an md5 of the sorted train indices so a split can be matched
across the GCN and MoE-GCN arms.

Usage as a library:
    from randomized_scaffold import randomized_scaffold_split, save_split

    tr, va, te = randomized_scaffold_split(dataset, split_seed=0)
    save_split("splits/BACE_seed0.json", tr, va, te, dataset, name="BACE")

Usage as a script (generate and audit splits without training anything):
    python randomized_scaffold.py --dataset BACE --seeds 0 1 2 3 4 5 6 7 8 9
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- split
def _scaffold_groups(smiles_list):
    """Map Murcko scaffold -> list of row indices. Unparsable SMILES get their own group."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups = defaultdict(list)
    n_invalid = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            groups[f"__invalid_{i}"].append(i)
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            sc = f"__failed_{i}"
        groups[sc].append(i)
    return groups, n_invalid


def randomized_scaffold_indices(smiles_list, split_seed, frac_train=0.8, frac_val=0.1):
    """
    chemprop scaffold_balanced. Returns (train_idx, val_idx, test_idx) as lists.

    Deterministic given (smiles_list, split_seed): the same seed always reproduces
    the same split, which is what makes the saved indices meaningful.
    """
    groups, n_invalid = _scaffold_groups(smiles_list)
    n = len(smiles_list)
    train_size = int(frac_train * n)
    val_size = int(frac_val * n)
    test_size = n - train_size - val_size

    # Big groups to train first, so one large scaffold can't dominate val/test.
    big, small = [], []
    for g in groups.values():
        if len(g) > val_size / 2 or len(g) > test_size / 2:
            big.append(g)
        else:
            small.append(g)

    rng = np.random.RandomState(split_seed)
    rng.shuffle(small)
    ordered = big + small

    train_idx, val_idx, test_idx = [], [], []
    for g in ordered:
        # Check BEFORE extending - this is the overshoot bug in the old splitter.
        if len(train_idx) + len(g) <= train_size:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= val_size:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "split lost molecules"
    assert not (set(train_idx) & set(val_idx)), "train/val index overlap"
    assert not (set(train_idx) & set(test_idx)), "train/test index overlap"
    assert not (set(val_idx) & set(test_idx)), "val/test index overlap"
    return train_idx, val_idx, test_idx, len(groups), n_invalid


def randomized_scaffold_split(dataset, split_seed, frac_train=0.8, frac_val=0.1):
    """Same, returning torch Subsets. `dataset.smiles` must be the SMILES list."""
    import torch

    tr, va, te, _, _ = randomized_scaffold_indices(
        list(dataset.smiles), split_seed, frac_train, frac_val
    )
    return (
        torch.utils.data.Subset(dataset, tr),
        torch.utils.data.Subset(dataset, va),
        torch.utils.data.Subset(dataset, te),
    )


# --------------------------------------------------------------------------- saving
def md5_idx(idx):
    return hashlib.md5(",".join(map(str, sorted(idx))).encode()).hexdigest()


def _positive_rate(dataset, idx):
    """Mean positive rate over tasks, ignoring NaN labels. None if unavailable."""
    try:
        ys = []
        for i in idx:
            y = dataset[i].y.numpy().flatten()
            ys.append(y)
        arr = np.array(ys, dtype=float)
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(arr))
    except Exception:
        return None


def save_split(path, train_idx, val_idx, test_idx, dataset=None, name="", split_seed=None,
               n_scaffolds=None, n_invalid=None):
    """
    Write a split as JSON. Keys train/val/test are the format check_splits.py reads.
    Also stores md5s and positive rates so two arms can be verified as matched.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {
        "dataset": name,
        "split_seed": split_seed,
        "protocol": "randomized_scaffold_balanced_80_10_10",
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
        "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "md5": {"train": md5_idx(train_idx), "val": md5_idx(val_idx), "test": md5_idx(test_idx)},
        "n_scaffolds": n_scaffolds,
        "n_invalid_smiles": n_invalid,
    }
    if dataset is not None:
        rec["positive_rate"] = {
            "train": _positive_rate(dataset, train_idx),
            "val": _positive_rate(dataset, val_idx),
            "test": _positive_rate(dataset, test_idx),
        }
        # Store SMILES order hash: proves both arms indexed the same dataset object.
        try:
            smi = ",".join(list(dataset.smiles))
            rec["smiles_order_md5"] = hashlib.md5(smi.encode()).hexdigest()
        except Exception:
            pass
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def load_split(path):
    with open(path) as f:
        rec = json.load(f)
    return rec["train"], rec["val"], rec["test"]


# --------------------------------------------------------------------------- script
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Generate and audit randomized scaffold splits.")
    ap.add_argument("--dataset", required=True, help="MoleculeNet name, e.g. BACE")
    ap.add_argument("--root", default="data/MoleculeNet")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--rf", action="store_true",
                    help="also run a Morgan-fingerprint RandomForest on each split")
    a = ap.parse_args()

    from torch_geometric.datasets import MoleculeNet

    ds = MoleculeNet(root=a.root, name=a.dataset)
    smiles = list(ds.smiles)
    print(f"{a.dataset}: {len(smiles)} molecules")

    rf_scores = []
    for s in a.seeds:
        tr, va, te, n_scaf, n_bad = randomized_scaffold_indices(smiles, s)
        path = os.path.join(a.outdir, f"{a.dataset}_seed{s}.json")
        rec = save_split(path, tr, va, te, ds, name=a.dataset, split_seed=s,
                         n_scaffolds=n_scaf, n_invalid=n_bad)
        pr = rec.get("positive_rate") or {}
        pr_s = "  ".join(
            f"{k}={pr[k]:.3f}" for k in ("train", "val", "test")
            if pr.get(k) is not None
        )
        print(f"  seed {s:>2}: {len(tr)}/{len(va)}/{len(te)}  "
              f"scaffolds={n_scaf}  invalid={n_bad}  {pr_s}  -> {path}")

        if a.rf:
            auc = _rf_auc(smiles, ds, tr, te)
            rf_scores.append(auc)
            print(f"           RF test AUC: {auc:.4f}")

    if rf_scores:
        arr = np.array([x for x in rf_scores if not np.isnan(x)])
        print(f"\n  RF across {len(arr)} splits: {arr.mean():.4f} +/- {arr.std():.4f}")
        print("  (A trivial fingerprint model scoring very high means the split is easy,")
        print("   not that a GNN trained on it is good. Compare against the withdrawn")
        print("   stratified split's 0.984 on BACE.)")


def _rf_auc(smiles, dataset, train_idx, test_idx, seed=0):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def fp(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        v = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(v, arr)
        return arr

    X = [fp(s) for s in smiles]
    Y = np.array([dataset[i].y.numpy().flatten() for i in range(len(smiles))], dtype=float)

    aucs = []
    for t in range(Y.shape[1]):
        tr = [i for i in train_idx if X[i] is not None and not np.isnan(Y[i, t])]
        te = [i for i in test_idx if X[i] is not None and not np.isnan(Y[i, t])]
        if len(tr) == 0 or len(te) == 0:
            continue
        ytr, yte = Y[tr, t], Y[te, t]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed,
                                     class_weight="balanced")
        clf.fit(np.stack([X[i] for i in tr]), ytr)
        p = clf.predict_proba(np.stack([X[i] for i in te]))[:, 1]
        aucs.append(roc_auc_score(yte, p))
    return float(np.mean(aucs)) if aucs else float("nan")


if __name__ == "__main__":
    main()
