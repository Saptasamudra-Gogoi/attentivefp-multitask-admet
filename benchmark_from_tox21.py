"""
AttentiveFP + Optuna Benchmark — SIDER, ClinTox, HIV
(Tox21 done. ToxCast skipped — AWS S3 blocked on CN networks)
All classification → metric: ROC-AUC (higher is better)

Run: python benchmark_from_tox21.py
Results appended to: results_from_tox21.json
"""

import os
import json
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import AttentiveFP
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS    = 30
N_SEEDS     = 3
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 64
DATA_ROOT   = "./data"
SAVE_PATH   = "results_from_tox21.json"

# Tox21 already done — continuing from SIDER
# ToxCast skipped (617 tasks + AWS S3 blocked on CN networks)
DATASETS = [
    {"name": "SIDER",   "tasks": 27, "metric": "roc_auc"},
    {"name": "ClinTox", "tasks": 2,  "metric": "roc_auc"},
    {"name": "HIV",     "tasks": 1,  "metric": "roc_auc"},
]

print(f"Device: {DEVICE}")
print(f"Datasets queued: {[d['name'] for d in DATASETS]}")
print(f"Tox21 results will be loaded and preserved in {SAVE_PATH}\n")

# ── Load existing results (preserve Tox21) ──────────────────────────────────────
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        results = json.load(f)
    print(f"Loaded existing results: {list(results.keys())}")
else:
    results = {}

# ── Scaffold Split ──────────────────────────────────────────────────────────────
def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        try:
            mol = Chem.MolFromSmiles(data.smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        scaffolds[scaffold].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n       = len(dataset)
    n_train = int(n * frac_train)
    n_val   = int(n * frac_val)

    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) + len(s) <= n_train:
            train_idx.extend(s)
        elif len(val_idx) + len(s) <= n_val:
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    return train_idx, val_idx, test_idx


# ── Masked BCE Loss ─────────────────────────────────────────────────────────────
def masked_bce_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred.device)
    return F.binary_cross_entropy_with_logits(pred[mask], target[mask])


# ── ROC-AUC (multi-task, masked) ───────────────────────────────────────────────
def compute_roc_auc(preds, targets):
    preds   = np.array(preds)
    targets = np.array(targets)
    aucs = []
    for t in range(targets.shape[1]):
        mask = ~np.isnan(targets[:, t])
        if mask.sum() < 2 or len(np.unique(targets[mask, t])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(targets[mask, t], preds[mask, t]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


# ── Train / Eval ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        loss = masked_bce_loss(out, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
    return compute_roc_auc(
        np.concatenate(all_preds,   axis=0),
        np.concatenate(all_targets, axis=0),
    )


# ── Full Training Run (one seed) ────────────────────────────────────────────────
def run_training(train_loader, val_loader, test_loader, params, n_tasks, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AttentiveFP(
        in_channels     = 9,
        hidden_channels = params["hidden"],
        out_channels    = n_tasks,
        edge_dim        = 3,
        num_layers      = params["num_layers"],
        num_timesteps   = params["num_timesteps"],
        dropout         = params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6
    )

    best_val_auc, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        val_auc = eval_epoch(model, val_loader)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return eval_epoch(model, test_loader)


# ── Optuna Objective ────────────────────────────────────────────────────────────
def make_objective(train_loader, val_loader, n_tasks):
    def objective(trial):
        params = {
            "hidden"       : trial.suggest_categorical("hidden", [64, 128, 200, 256]),
            "num_layers"   : trial.suggest_int("num_layers",    2, 5),
            "num_timesteps": trial.suggest_int("num_timesteps", 2, 4),
            "dropout"      : trial.suggest_float("dropout",     0.0, 0.5),
            "lr"           : trial.suggest_float("lr",          1e-4, 1e-2, log=True),
            "weight_decay" : trial.suggest_float("weight_decay",1e-6, 1e-3, log=True),
        }

        torch.manual_seed(42)
        model = AttentiveFP(
            in_channels     = 9,
            hidden_channels = params["hidden"],
            out_channels    = n_tasks,
            edge_dim        = 3,
            num_layers      = params["num_layers"],
            num_timesteps   = params["num_timesteps"],
            dropout         = params["dropout"],
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        best_val, patience_ctr = 0.0, 0

        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer)
            val_auc = eval_epoch(model, val_loader)

            if val_auc > best_val:
                best_val, patience_ctr = val_auc, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val

    return objective


# ── Main Loop ───────────────────────────────────────────────────────────────────
for ds_cfg in DATASETS:
    name    = ds_cfg["name"]
    n_tasks = ds_cfg["tasks"]
    t0      = time.time()

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}  |  Tasks : {n_tasks}  |  Metric : ROC-AUC")
    print(f"{'='*60}")

    dataset = MoleculeNet(root=DATA_ROOT, name=name)

    train_idx, val_idx, test_idx = scaffold_split(dataset)
    train_data = [dataset[i] for i in train_idx]
    val_data   = [dataset[i] for i in val_idx]
    test_data  = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Split  → Train:{len(train_data)}  Val:{len(val_data)}  Test:{len(test_data)}")

    print(f"  Running Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction = "maximize",
        sampler   = TPESampler(seed=42),
        pruner    = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(
        make_objective(train_loader, val_loader, n_tasks),
        n_trials          = N_TRIALS,
        timeout           = 3600,
        show_progress_bar = False,
    )
    best_params  = study.best_params
    best_val_auc = study.best_value
    print(f"  Best val AUC : {best_val_auc:.4f}")
    print(f"  Best params  : {best_params}")

    print(f"  Final eval ({N_SEEDS} seeds)...")
    seed_aucs = []
    for seed in [0, 1, 2]:
        auc = run_training(train_loader, val_loader, test_loader, best_params, n_tasks, seed)
        seed_aucs.append(auc)
        print(f"    Seed {seed} → Test AUC: {auc:.4f}")

    mean_auc = float(np.mean(seed_aucs))
    std_auc  = float(np.std(seed_aucs))
    elapsed  = time.time() - t0

    results[name] = {
        "metric"      : "ROC-AUC",
        "n_tasks"     : n_tasks,
        "mean"        : mean_auc,
        "std"         : std_auc,
        "seeds"       : seed_aucs,
        "best_params" : best_params,
        "time_min"    : round(elapsed / 60, 1),
    }

    print(f"\n  ✓ {name}: {mean_auc:.4f} ± {std_auc:.4f}  ({elapsed/60:.1f} min)")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {SAVE_PATH}")

# ── Summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Dataset':<12} {'ROC-AUC Mean':>14} {'±Std':>8} {'Time(min)':>10}")
print("-" * 50)
for name, r in results.items():
    print(f"{name:<12} {r['mean']:>14.4f} {r['std']:>8.4f} {r['time_min']:>10.1f}")

print(f"\nAll results saved to: {SAVE_PATH}")
