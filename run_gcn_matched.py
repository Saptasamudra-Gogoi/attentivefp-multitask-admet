"""
run_gcn_matched.py
-------------------
Plain GCN on ESOL / FreeSolv / Lipo, matched to moegcn_regr.py's protocol:
  - Same scaffold_split() (identical function, no seed arg -> same split for all runs)
  - Same Optuna HPO: 30 trials, TPE(seed=42), same search space minus num_experts/top_k
  - Same EPOCHS/PATIENCE/BATCH_SIZE
  - 5 final seeds (0,1,2,3,4) to match MoE-GCN after extend_seeds.py

This directly answers P0-2: GCN's column in Table 1 must carry mean +/- std
over the same seed set, produced under the same tuning budget, as MoE-GCN.

Run: python run_gcn_matched.py
Output: results_gcn_matched.json
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Config (identical to moegcn_regr.py where applicable) --
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 5                      # matched to MoE-GCN post extend_seeds.py
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_gcn_matched.json"

DATASETS = [
    {"name": "ESOL",     "tasks": 1},
    {"name": "FreeSolv", "tasks": 1},
    {"name": "Lipo",     "tasks": 1},
]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -- Plain GCN model (no MoE layer; everything else matches MoEGCN) --
class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


# -- Scaffold split: byte-identical to moegcn_regr.py's version --
def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for i, d in enumerate(dataset):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            else:
                sc = smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff = int(n * (frac_train + frac_val))

    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            preds = out.squeeze().cpu().numpy()
            labels = batch.y.squeeze().cpu().numpy()
            all_preds.extend(np.atleast_1d(preds).flatten())
            all_labels.extend(np.atleast_1d(labels).flatten())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    mask = ~np.isnan(labels)
    rmse = float(np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2)))
    return rmse


def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch.y.float().squeeze()
        mask = ~torch.isnan(labels)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(out.squeeze()[mask], labels[mask])
        loss.backward()
        optimizer.step()


def make_objective(train_data, val_data, in_dim):
    def objective(trial):
        hidden     = trial.suggest_categorical("hidden", [128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout    = trial.suggest_float("dropout", 0.0, 0.3)
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        model = PlainGCN(in_dim, hidden, num_layers, dropout, 1).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)

        best_val = float("inf")
        patience_count = 0
        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer)
            val_rmse = evaluate(model, val_loader)
            scheduler.step(val_rmse)
            if val_rmse < best_val:
                best_val = val_rmse
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                break
        return best_val
    return objective


# -- Main loop --
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    print(f"Resuming from {SAVE_PATH} - {len(results)} datasets done")
else:
    results = {}

for ds in DATASETS:
    name = ds["name"]

    if name in results:
        print(f"  Skipping {name} (already done)")
        continue

    print(f"\n{'='*50}")
    print(f"  Plain GCN (matched HPO) | {name} | RMSE")
    print(f"{'='*50}")
    t0 = time.time()

    dataset = MoleculeNet(root=DATA_ROOT, name=name)
    in_dim = dataset.num_node_features
    train_data, val_data, test_data = scaffold_split(dataset)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(make_objective(train_data, val_data, in_dim), n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"  Best val RMSE: {study.best_value:.4f} | {best_params}")

    seed_scores = []
    val_loader  = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PlainGCN(
            in_dim,
            best_params["hidden"],
            best_params["num_layers"],
            best_params["dropout"],
            1
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val, patience_count, best_state = float("inf"), 0, None
        for epoch in range(EPOCHS):
            train_epoch(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), optimizer)
            val_rmse = evaluate(model, val_loader)
            scheduler.step(val_rmse)
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                break

        model.load_state_dict(best_state)
        test_rmse = evaluate(model, test_loader)
        seed_scores.append(test_rmse)
        print(f"    Seed {seed} -> RMSE: {test_rmse:.4f}")

    mean_s = float(np.mean(seed_scores))
    std_s  = float(np.std(seed_scores))
    elapsed = time.time() - t0

    results[name] = {
        "mean": mean_s, "std": std_s,
        "metric": "rmse", "seeds": seed_scores,
        "best_params": best_params, "time_min": round(elapsed / 60, 1),
        "model": "plain_GCN_matched_hpo"
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Done: {name}: {mean_s:.4f} +/- {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*50}")
print("  PLAIN GCN (MATCHED HPO) SUMMARY")
print(f"{'='*50}")
print(f"{'Dataset':12} {'RMSE':>10} {'+/-Std':>8}")
print("-" * 33)
for name, r in results.items():
    print(f"{name:12} {r['mean']:>10.4f} {r['std']:>8.4f}")
print(f"\nSaved -> {SAVE_PATH}")
