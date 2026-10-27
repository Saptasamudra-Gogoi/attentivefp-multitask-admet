#!/usr/bin/env python
"""
check_splits.py - split-integrity audit for MoleculeNet-style classification datasets.

What it checks (per dataset):
  1. Split sizes and invalid SMILES
  2. Murcko scaffold overlap train/val/test   (must be 0 for a true scaffold split)
  3. Exact-duplicate molecule overlap         (canonical SMILES; must be 0)
  4. Positive rate per split, per task        (large train/test gaps distort AUC)
  5. MD5 hash of each split's sorted indices  (identical hash across pipelines = matched split)
  6. Optional: Morgan-fingerprint RandomForest test AUC on the same split
     -> if a trivial RF also scores very high, the split is easy/leaky, not your GCN "good"
  7. Optional: compare against a second split (e.g. the MoE-GCN run) -> MATCH / MISMATCH

USAGE
  # A) Audit the split your pipeline actually used (recommended):
  python check_splits.py --csv data/bace.csv --split my_runs/BACE_split.json

  # B) Separate index files:
  python check_splits.py --csv data/bace.csv --train tr.npy --val va.npy --test te.npy

  # C) No saved split: audit a reference deterministic scaffold split (80/10/10)
  python check_splits.py --csv data/bace.csv

  # D) Check that GCN and MoE-GCN used the identical split:
  python check_splits.py --csv data/bace.csv --split gcn_split.json --compare moe_split.json

Accepted split formats (--split / --compare): .json, .npz, .pt/.pth, .pkl
  containing a dict with keys train / valid (or val) / test -> lists of row indices.
Accepted index files (--train/--val/--test): .npy, .txt/.csv (one index per line), .json, .pt
Indices must be ROW POSITIONS in --csv after the script drops nothing (it never drops rows;
invalid SMILES are reported, not removed). If your pipeline dropped invalid rows BEFORE
splitting, pass --drop-invalid so positions line up.
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SMILES_CANDIDATES = ["smiles", "SMILES", "mol", "Smiles", "canonical_smiles", "Drug"]
NON_LABEL_COLS = {"mol_id", "id", "ID", "CID", "Model", "pIC50", "MW", "AlogP", "HBA", "HBD",
                  "RB", "HeavyAtomCount", "ChiralCenterCount", "ChiralCenterCountAllPossible",
                  "RingCount", "PSA", "Estate", "MR", "Polar", "sLogP", "sMR", "sPSA", "Unnamed: 0"}


# ----------------------------------------------------------------------------- loading
def load_dataset(csv_path, smiles_col=None, label_cols=None):
    df = pd.read_csv(csv_path)
    if smiles_col is None:
        smiles_col = next((c for c in SMILES_CANDIDATES if c in df.columns), None)
        if smiles_col is None:
            sys.exit(f"Could not find a SMILES column in {list(df.columns)}. Use --smiles-col.")
    if label_cols is None:
        label_cols = []
        for c in df.columns:
            if c == smiles_col or c in NON_LABEL_COLS:
                continue
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(vals) > 0 and set(np.round(vals, 6)).issubset({0.0, 1.0}):
                label_cols.append(c)
        if not label_cols:
            sys.exit("No binary label columns detected. Use --label-cols.")
    y = df[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return df[smiles_col].astype(str).tolist(), y, smiles_col, label_cols


def _to_int_list(obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
    except ImportError:
        pass
    return [int(i) for i in np.asarray(obj).ravel().tolist()]


def _load_any(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        return np.load(p, allow_pickle=True)
    if suf == ".npz":
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}
    if suf == ".json":
        return json.loads(p.read_text())
    if suf in (".pt", ".pth"):
        import torch
        return torch.load(p, map_location="cpu", weights_only=False)
    if suf in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            return pickle.load(f)
    if suf in (".txt", ".csv"):
        return [int(float(x)) for x in p.read_text().split() if x.strip()]
    sys.exit(f"Unsupported file type: {path}")


def load_split_dict(path):
    obj = _load_any(path)
    if not isinstance(obj, dict):
        sys.exit(f"{path} must contain a dict with train/valid/test keys.")
    keymap = {}
    for k in obj:
        kl = str(k).lower()
        if kl.startswith("train"):
            keymap["train"] = k
        elif kl in ("valid", "val", "validation", "dev"):
            keymap["val"] = k
        elif kl.startswith("test"):
            keymap["test"] = k
    missing = {"train", "val", "test"} - set(keymap)
    if missing:
        sys.exit(f"{path}: missing keys {missing}; found {list(obj)}")
    return {s: _to_int_list(obj[keymap[s]]) for s in ("train", "val", "test")}


# ----------------------------------------------------------------------------- chemistry
def mol_info(smiles):
    canon, scaf = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            canon.append(None)
            scaf.append(None)
            continue
        canon.append(Chem.MolToSmiles(m))
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
        except Exception:
            scaf.append(None)
    return canon, scaf


def reference_scaffold_split(scaffolds, frac=(0.8, 0.1, 0.1)):
    """DeepChem/MoleculeNet-style deterministic scaffold split: largest scaffold sets -> train."""
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s if s is not None else f"__invalid_{i}"].append(i)
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(scaffolds)
    tr_cut, va_cut = frac[0] * n, (frac[0] + frac[1]) * n
    out = {"train": [], "val": [], "test": []}
    for g in sets:
        if len(out["train"]) + len(g) <= tr_cut:
            out["train"] += g
        elif len(out["train"]) + len(out["val"]) + len(g) <= va_cut:
            out["val"] += g
        else:
            out["test"] += g
    return out


def md5_idx(idx):
    return hashlib.md5(",".join(map(str, sorted(idx))).encode()).hexdigest()


# ----------------------------------------------------------------------------- RF baseline
def rf_test_auc(smiles, y, split, seed=0):
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def fp(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        v = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        a = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(v, a)
        return a

    X = [fp(s) for s in smiles]
    aucs = []
    for t in range(y.shape[1]):
        tr = [i for i in split["train"] if X[i] is not None and not np.isnan(y[i, t])]
        te = [i for i in split["test"] if X[i] is not None and not np.isnan(y[i, t])]
        ytr, yte = y[tr, t], y[te, t]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed,
                                     class_weight="balanced")
        clf.fit(np.stack([X[i] for i in tr]), ytr)
        p = clf.predict_proba(np.stack([X[i] for i in te]))[:, 1]
        aucs.append(roc_auc_score(yte, p))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)


# ----------------------------------------------------------------------------- report
def audit(name, smiles, y, label_cols, split, canon, scaf, run_rf):
    n = len(smiles)
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}")

    all_idx = split["train"] + split["val"] + split["test"]
    bad = [i for i in all_idx if i < 0 or i >= n]
    if bad:
        sys.exit(f"  Indices out of range (n={n}): e.g. {bad[:5]}. Wrong CSV or --drop-invalid needed?")
    dup_rows = len(all_idx) - len(set(all_idx))
    print(f"  Sizes  train/val/test : {len(split['train'])}/{len(split['val'])}/{len(split['test'])}"
          f"  (total {len(all_idx)} of {n} rows)")
    if dup_rows:
        print(f"  !! {dup_rows} row indices appear in more than one split")
    unused = n - len(set(all_idx))
    if unused:
        print(f"  Note: {unused} rows not assigned to any split")
    n_invalid = sum(canon[i] is None for i in set(all_idx))
    print(f"  Invalid SMILES in split: {n_invalid}")

    def sset(idx):
        return {scaf[i] for i in idx if scaf[i] not in (None, "")}

    def cset(idx):
        return {canon[i] for i in idx if canon[i] is not None}

    S = {k: sset(v) for k, v in split.items()}
    C = {k: cset(v) for k, v in split.items()}
    flags = []
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        so, co = len(S[a] & S[b]), len(C[a] & C[b])
        print(f"  {a:>5}/{b:<5} scaffold overlap: {so:5d}   exact-duplicate overlap: {co:5d}")
        if so:
            flags.append(f"{a}/{b} share {so} scaffolds -> NOT a scaffold split")
        if co:
            flags.append(f"{a}/{b} share {co} identical molecules -> LEAKAGE")
    acyclic = sum(1 for i in split["test"] if scaf[i] == "")
    if acyclic:
        print(f"  (test has {acyclic} acyclic molecules with empty scaffold; excluded from overlap)")

    print("  Positive rate (train / val / test):")
    gaps = []
    for t, col in enumerate(label_cols):
        r = [np.nanmean(y[split[s], t]) if np.any(~np.isnan(y[split[s], t])) else np.nan
             for s in ("train", "val", "test")]
        gaps.append(abs(r[0] - r[2]))
        if len(label_cols) <= 4:
            print(f"    {col:<22} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}")
    if len(label_cols) > 4:
        print(f"    {len(label_cols)} tasks; max |train-test| gap = {np.nanmax(gaps):.3f} "
              f"(task '{label_cols[int(np.nanargmax(gaps))]}'), mean gap = {np.nanmean(gaps):.3f}")
    if np.nanmax(gaps) > 0.15:
        flags.append(f"train/test positive-rate gap up to {np.nanmax(gaps):.2f}")

    for s in ("train", "val", "test"):
        print(f"  md5 {s:<5}: {md5_idx(split[s])}")

    if run_rf:
        auc, k = rf_test_auc(smiles, y, split)
        print(f"  RF (Morgan r2, 2048b) test ROC-AUC: {auc:.4f}  ({k} tasks scored)")

    print("  VERDICT: " + ("; ".join(flags) if flags else "no scaffold/duplicate leakage, balanced splits"))
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="dataset CSV (raw MoleculeNet file)")
    ap.add_argument("--smiles-col")
    ap.add_argument("--label-cols", nargs="+")
    ap.add_argument("--split", help="file with dict train/valid/test indices")
    ap.add_argument("--train"), ap.add_argument("--val"), ap.add_argument("--test")
    ap.add_argument("--compare", help="second split file to check for identity (e.g. MoE-GCN run)")
    ap.add_argument("--drop-invalid", action="store_true",
                    help="drop unparsable SMILES before indexing (match pipelines that did this)")
    ap.add_argument("--no-rf", action="store_true", help="skip RandomForest baseline")
    a = ap.parse_args()

    smiles, y, scol, lcols = load_dataset(a.csv, a.smiles_col, a.label_cols)
    if a.drop_invalid:
        keep = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
        print(f"--drop-invalid: removed {len(smiles) - len(keep)} rows")
        smiles, y = [smiles[i] for i in keep], y[keep]
    print(f"Loaded {a.csv}: {len(smiles)} molecules | SMILES col '{scol}' | {len(lcols)} task(s): "
          f"{lcols if len(lcols) <= 6 else lcols[:6] + ['...']}")
    canon, scaf = mol_info(smiles)

    if a.split:
        split, label = load_split_dict(a.split), f"Split from {a.split}"
    elif a.train and a.val and a.test:
        split = {"train": _to_int_list(_load_any(a.train)), "val": _to_int_list(_load_any(a.val)),
                 "test": _to_int_list(_load_any(a.test))}
        label = "Split from index files"
    else:
        split, label = reference_scaffold_split(scaf), "REFERENCE deterministic scaffold split 80/10/10"

    audit(label, smiles, y, lcols, split, canon, scaf, not a.no_rf)

    if a.compare:
        other = load_split_dict(a.compare)
        print(f"\n{'=' * 64}\n  Identity check vs {a.compare}\n{'=' * 64}")
        all_same = True
        for s in ("train", "val", "test"):
            same = md5_idx(split[s]) == md5_idx(other[s])
            all_same &= same
            extra = "" if same else (f"  (only-in-first {len(set(split[s]) - set(other[s]))}, "
                                     f"only-in-second {len(set(other[s]) - set(split[s]))})")
            print(f"  {s:<5}: {'MATCH' if same else 'MISMATCH'}{extra}")
        print("  RESULT: " + ("identical splits - comparison is matched" if all_same
                              else "splits differ - GCN vs MoE-GCN comparison is NOT matched"))


if __name__ == "__main__":
    main()
