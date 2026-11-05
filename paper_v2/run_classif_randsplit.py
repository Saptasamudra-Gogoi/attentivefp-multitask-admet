#!/usr/bin/env python
"""
run_classif_randsplit.py - GCN vs MoE-GCN on MoleculeNet classification over TEN
randomized scaffold splits (saved by randomized_scaffold.py), replacing the
withdrawn stratified split and the overshooting deterministic split.

Protocol
  * Optuna TPE (seed 42), 30 trials, tuned ONCE per arm on split seed 0 (train/val
    of split 0 only). Search spaces match moegcn_classif.py; the plain GCN omits only
    num_experts and top_k.
  * best_params frozen, then each arm is trained on split seeds 0-9. Model init and
    batch order use torch.manual_seed(split_seed), so the arms are paired per split.
  * Training: Adam, ReduceLROnPlateau(patience 5, factor 0.5), early stopping on val
    AUC (patience 15), max 100 epochs, batch 64, loss BCE + 0.01 * balance loss.
    Test AUC is taken at the best-val epoch.
  * Splits are loaded from JSON and their smiles_order_md5 is checked against the
    dataset, so both arms provably index the same molecules.

Caveat (state in the paper): hyperparameters are tuned on split 0's validation set.
Molecules in that set can appear in other splits' test sets, so splits 1-9 are not
perfectly independent of tuning. Split 0's own test set is untouched by tuning.

Run (from paper_v2):
  python run_classif_randsplit.py --datasets BACE BBBP ClinTox SIDER Tox21
  python run_classif_randsplit.py --summarize
Resumable: finished (dataset, arm) pairs are skipped.
"""
import argparse
import copy
import hashlib
import json
import os
import time
import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH_SIZE, N_TRIALS = 100, 15, 64, 30
TASKS = {"BACE": 1, "BBBP": 1, "ClinTox": 2, "SIDER": 27, "Tox21": 12}
OUT_DIR = "results_randsplit"


# ------------------------------------------------------------------ models (verbatim logic)
class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_out).sum(dim=1), balance_loss


class Net(nn.Module):
    """MoE-GCN if num_experts is not None, else plain GCN. Always returns (logits, bal_loss)."""
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks, num_experts=None, top_k=None):
        super().__init__()
        self.convs, self.bns, self.dropout = nn.ModuleList(), nn.ModuleList(), dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k) if num_experts else None
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        if self.moe is None:
            return self.head(x), torch.zeros((), device=x.device)
        x, bal = self.moe(x)
        return self.head(x), bal


def ToFloat(data):
    data.x = data.x.float()
    return data


def build(arm, in_dim, p, num_tasks):
    if arm == "moe":
        return Net(in_dim, p["hidden"], p["num_layers"], p["dropout"], num_tasks,
                   p["num_experts"], p["top_k"])
    return Net(in_dim, p["hidden"], p["num_layers"], p["dropout"], num_tasks)


# ------------------------------------------------------------------ train / eval
def train_epoch(model, loader, opt, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float()
        if y.ndim == 1:
            y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01 * bal
        loss.backward()
        opt.step()


def evaluate(model, loader, num_tasks):
    model.eval()
    P, L = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            y = batch.y.cpu().numpy()
            if y.ndim == 1:
                y = y.reshape(-1, num_tasks)
            P.append(torch.sigmoid(out).cpu().numpy())
            L.append(y)
    P, L = np.vstack(P), np.vstack(L)
    aucs = []
    for t in range(num_tasks):
        m = ~np.isnan(L[:, t])
        if m.sum() < 2 or len(np.unique(L[m, t])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(L[m, t], P[m, t]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def fit(model, train_data, val_data, lr, wd, num_tasks):
    """Train with early stopping on val AUC. Returns (best_val, best_state)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(val_data, batch_size=BATCH_SIZE)
    best, pat, state = 0.0, 0, copy.deepcopy(model.state_dict())
    for _ in range(EPOCHS):
        train_epoch(model, tl, opt, num_tasks)
        v = evaluate(model, vl, num_tasks)
        sched.step(-v)
        if v > best:
            best, pat, state = v, 0, copy.deepcopy(model.state_dict())
        else:
            pat += 1
        if pat >= PATIENCE:
            break
    return best, state


def make_objective(arm, tr, va, in_dim, num_tasks):
    def objective(trial):
        p = {
            "hidden": trial.suggest_categorical("hidden", [128, 256]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
        if arm == "moe":
            p["num_experts"] = trial.suggest_categorical("num_experts", [4, 8, 16])
            p["top_k"] = trial.suggest_int("top_k", 1, min(4, p["num_experts"]))
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        torch.manual_seed(0)
        model = build(arm, in_dim, p, num_tasks).to(DEVICE)
        best, _ = fit(model, tr, va, lr, wd, num_tasks)
        return best
    return objective


# ------------------------------------------------------------------ splits
def load_split(root_splits, name, seed, dataset):
    path = os.path.join(root_splits, f"{name}_seed{seed}.json")
    with open(path) as f:
        rec = json.load(f)
    want = rec.get("smiles_order_md5")
    got = hashlib.md5(",".join(list(dataset.smiles)).encode()).hexdigest()
    if want and want != got:
        raise RuntimeError(f"{path}: saved split indexes a different molecule order than this "
                           f"dataset object (PyG version or filtering changed). Regenerate splits.")
    sub = lambda idx: torch.utils.data.Subset(dataset, idx)
    return sub(rec["train"]), sub(rec["val"]), sub(rec["test"]), rec["md5"]


# ------------------------------------------------------------------ main loop
def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in args.datasets:
        T = TASKS[name]
        ds = MoleculeNet(root=args.root, name=name, transform=ToFloat)
        in_dim = ds.num_node_features
        for arm in args.arms:
            out_path = os.path.join(OUT_DIR, f"{name}_{arm}.json")
            if os.path.exists(out_path):
                print(f"skip {name}/{arm} (done)")
                continue
            t0 = time.time()
            print(f"\n=== {name} | {arm} | {T} tasks ===")
            tr0, va0, _, _ = load_split(args.splits, name, 0, ds)
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            study.optimize(make_objective(arm, tr0, va0, in_dim, T), n_trials=N_TRIALS)
            bp = study.best_params
            print(f"  tuned on split 0: val AUC {study.best_value:.4f} | {bp}")

            test_aucs, val_aucs, md5s = [], [], []
            for s in args.seeds:
                tr, va, te, md5 = load_split(args.splits, name, s, ds)
                torch.manual_seed(s)
                np.random.seed(s)
                model = build(arm, in_dim, bp, T).to(DEVICE)
                bv, state = fit(model, tr, va, bp["lr"], bp["weight_decay"], T)
                model.load_state_dict(state)
                ta = evaluate(model, DataLoader(te, batch_size=BATCH_SIZE), T)
                test_aucs.append(ta); val_aucs.append(bv); md5s.append(md5)
                print(f"  split {s}: val {bv:.4f}  test {ta:.4f}")

            rec = {"dataset": name, "arm": arm, "protocol": "tuned_once_split0_eval_10_randomized_scaffold",
                   "split_seeds": args.seeds, "test_auc": test_aucs, "val_auc": val_aucs,
                   "mean": float(np.mean(test_aucs)), "std": float(np.std(test_aucs, ddof=1)),
                   "best_params": bp, "tuning_val_auc": study.best_value,
                   "split_md5": md5s, "time_min": round((time.time() - t0) / 60, 1)}
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"  {name}/{arm}: {rec['mean']:.4f} +/- {rec['std']:.4f} ({rec['time_min']} min)")


# ------------------------------------------------------------------ summary
def summarize():
    from scipy import stats
    rows, d_means = [], []
    print(f"{'Dataset':9}{'GCN':>16}{'MoE-GCN':>16}{'paired diff':>14}{'95% CI':>22}{'wins':>6}")
    for name in TASKS:
        pg, pm = (os.path.join(OUT_DIR, f"{name}_{a}.json") for a in ("gcn", "moe"))
        if not (os.path.exists(pg) and os.path.exists(pm)):
            continue
        g, m = json.load(open(pg)), json.load(open(pm))
        if g["split_md5"] != m["split_md5"]:
            print(f"{name}: SPLIT MISMATCH between arms - not a paired comparison"); continue
        a, b = np.array(g["test_auc"]), np.array(m["test_auc"])
        d = b - a
        n = len(d); se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = d.mean() - stats.t.ppf(0.975, n - 1) * se, d.mean() + stats.t.ppf(0.975, n - 1) * se
        d_means.append(d.mean())
        print(f"{name:9}{a.mean():>9.4f}+/-{a.std(ddof=1):.3f}{b.mean():>9.4f}+/-{b.std(ddof=1):.3f}"
              f"{d.mean():>+14.4f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{int((d > 0).sum()):>4}/{n}")
    k = len(d_means)
    if k:
        dm = np.array(d_means)
        wins = int((dm > 0).sum())
        p_sign = stats.binomtest(wins, k, 0.5, alternative="greater").pvalue
        try:
            p_w = stats.wilcoxon(dm, alternative="greater").pvalue
        except Exception:
            p_w = float("nan")
        se = dm.std(ddof=1) / np.sqrt(k) if k > 1 else float("nan")
        tcrit = stats.t.ppf(0.975, k - 1) if k > 1 else float("nan")
        print(f"\nPooled over {k} datasets: mean diff {dm.mean():+.4f} "
              f"(95% CI {dm.mean() - tcrit * se:+.3f} to {dm.mean() + tcrit * se:+.3f}), "
              f"MoE wins {wins}/{k}, exact one-sided sign P={p_sign:.3f}, Wilcoxon P={p_w:.3f}")
        print("These are the numbers that replace the withdrawn five-dataset statistics in v2.31.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--arms", nargs="+", default=["gcn", "moe"], choices=["gcn", "moe"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--root", default="data/MoleculeNet")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a)
