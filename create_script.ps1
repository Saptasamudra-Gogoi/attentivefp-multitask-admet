@'
"""
ablation_add_plain_gcn.py
==========================
P0-5b fix. ablation_optuna.py already runs MoE-GCN, Dense-uniform, and
Dense-wide fairly (separate 15-trial Optuna HPO per mode, 3 seeds) on
ESOL/FreeSolv/Lipo -- but plain GCN was never added to that table. As a
result the ablation shows three equal-capacity models tying, without
ever comparing against the plain backbone under the SAME fixed
protocol, so the Discussion's claim that "additional capacity remains
a plausible explanation for the gain over the plain backbone" was
unsupported by the table it cites.

This script adds a 4th mode, "plain_gcn" (just the GCN backbone + head,
no MoE/dense/wide layer at all), run under the IDENTICAL protocol:
same scaffold_split, same 15-trial Optuna (TPESampler seed=42), same
search space minus num_experts/top_k, same EPOCHS/PATIENCE/BATCH, same
3 final seeds. Result is merged into the existing
ablation_optuna_results.json under a new "plain_gcn" key per dataset,
so Table 6 gets its 4th column without disturbing the other three.

Run: python ablation_add_plain_gcn.py
"""

import json, os, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from torch_geometric.data import DataLoader as GeoLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from collections import defaultdict

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS  = 15
N_SEEDS   = 3
EPOCHS    = 100
PATIENCE  = 15
BATCH     = 64
DATA_ROOT = "./data"
SAVE_PATH = "ablation_optuna_results.json"  # same file, adds plain_gcn key

DATASETS = ["ESOL", "FreeSolv", "Lipo"]

print(f"Device: {DEVICE}")


class PlainGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers_unused=None):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x), torch.tensor(0.0, device=x.device)


class GCNWithLayer(nn.Module):
    """Same backbone as ablation_optuna.py's GCNWithLayer, but the
    'layer' after pooling is just a linear passthrough -- i.e. plain
    GCN with no expert/MoE/dense-ensemble structure at all."""
    def __init__(self, in_dim, hidden, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.layer = PlainGCNLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal = self.layer(x)
        return self.head(x), bal

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False) if mol else str(i)
        except Exception:
            sc = str(i)
        scaffolds[sc].append(i)

    sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    t_cut = int(n * frac_train)
    v_cut = int(n * (frac_train + frac_val))
    train, val, test = [], [], []
    for s in sets:
        if len(train) < t_cut: train.extend(s)
        elif len(val) < v_cut - t_cut: val.extend(s)
        else: test.extend(s)

    return (torch.utils.data.Subset(dataset, train),
            torch.utils.data.Subset(dataset, val),
            torch.utils.data.Subset(dataset, test))


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    p = np.array(preds); l = np.array(labels)
    mask = ~np.isnan(l)
    return float(np.sqrt(np.mean((p[mask] - l[mask]) ** 2)))


def train_model(train_data, val_data, in_dim, params, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)

    model = GCNWithLayer(
        in_dim=in_dim,
        hidden=params['hidden'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    loader = GeoLoader(train_data, batch_size=BATCH, shuffle=True)
    val_l = GeoLoader(val_data, batch_size=BATCH)

    best_val, best_state, pat = float('inf'), None, 0
    model.train()
    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out, bal = model(batch)
            lab = batch.y.float().squeeze()
            mask = ~torch.isnan(lab)
            if mask.sum() == 0: continue
            loss = F.mse_loss(out.squeeze()[mask], lab[mask])  # no bal loss, plain GCN
            loss.backward(); opt.step()

        val_rmse = evaluate(model, val_l)
        sched.step(val_rmse)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE: break

    model.load_state_dict(best_state)
    return model, best_val


def make_objective(train_data, val_data, in_dim):
    def objective(trial):
        hidden = trial.suggest_categorical('hidden', [128, 256])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        wd = trial.suggest_float('wd', 1e-6, 1e-4, log=True)

        params = dict(hidden=hidden, num_layers=num_layers, dropout=dropout, lr=lr, wd=wd)
        _, val_rmse = train_model(train_data, val_data, in_dim, params, seed=42, epochs=EPOCHS)
        return val_rmse
    return objective


def main():
    print("\n" + "=" * 70)
    print("  Adding plain_gcn to ablation -- same protocol as other 3 modes")
    print(f"  {N_TRIALS} trials x {N_SEEDS} seeds x {len(DATASETS)} datasets")
    print("=" * 70)

    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded existing {SAVE_PATH} -- {len(all_results)} datasets present")
    else:
        all_results = {}
        print(f"  WARNING: {SAVE_PATH} not found -- creating fresh "
              f"(other 3 modes will be missing until ablation_optuna.py runs)")

    for ds_name in DATASETS:
        if ds_name in all_results and "plain_gcn" in all_results[ds_name]:
            print(f"\n  Skipping {ds_name} (plain_gcn already done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {ds_name}  [plain_gcn]")
        print(f"{'=' * 60}")
        t0 = time.time()

        dataset = MoleculeNet(root=DATA_ROOT, name=ds_name)
        in_dim = dataset.num_node_features
        train_d, val_d, test_d = scaffold_split(dataset)
        test_l = GeoLoader(test_d, batch_size=BATCH)

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(make_objective(train_d, val_d, in_dim), n_trials=N_TRIALS,
                        show_progress_bar=False)
        best_params = study.best_params
        best_val = study.best_value
        print(f"  Best val RMSE: {best_val:.4f} | {best_params}")

        seed_scores = []
        for seed in range(N_SEEDS):
            model, _ = train_model(train_d, val_d, in_dim, best_params, seed, EPOCHS)
            test_rmse = evaluate(model, test_l)
            seed_scores.append(test_rmse)
            print(f"    seed={seed} test RMSE={test_rmse:.4f}")

        mean_s = float(np.mean(seed_scores))
        std_s = float(np.std(seed_scores))
        n_p = GCNWithLayer(in_dim=in_dim, hidden=best_params['hidden'],
                            num_layers=best_params['num_layers'],
                            dropout=best_params['dropout']).n_params()

        if ds_name not in all_results:
            all_results[ds_name] = {}

        all_results[ds_name]["plain_gcn"] = {
            'label': 'Plain GCN (no routing, no MoE)',
            'mean': round(mean_s, 4),
            'std': round(std_s, 4),
            'seeds': [round(s, 4) for s in seed_scores],
            'best_params': best_params,
            'best_val': round(best_val, 4),
            'n_params': n_p,
        }
        elapsed = (time.time() - t0) / 60
        print(f"  -> plain_gcn: {mean_s:.4f} +/- {std_s:.4f}  (params={n_p:,})  ({elapsed:.1f} min)")

        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [SAVED] {SAVE_PATH}")

    # -- Updated 4-way summary --
    print(f"\n{'=' * 70}")
    print("  UPDATED ABLATION SUMMARY (4 modes)")
    print(f"{'=' * 70}")
    modes = ["moe", "dense_uniform", "dense_wide", "plain_gcn"]
    print(f"\n  {'Dataset':<12} {'MoE-GCN':>18} {'Dense-unif':>18} {'Dense-wide':>18} {'Plain-GCN':>18}  Winner")
    print(f"  {'-' * 95}")
    for ds in DATASETS:
        if ds not in all_results:
            continue
        res = all_results[ds]
        scores = {m: res[m]['mean'] for m in modes if m in res}
        if not scores:
            continue
        best_m = min(scores, key=scores.get)
        row = f"  {ds:<12}"
        for m in modes:
            if m in res:
                v = res[m]
                tag = " *" if m == best_m else "  "
                row += f"  {v['mean']:.4f}+/-{v['std']:.4f}{tag}"
            else:
                row += f"  {'N/A':>16}  "
        print(row + f"  {best_m}")

    print(f"\nSaved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
'@ | Set-Content -Path D:\molprop_projectblation_add_plain_gcn.py -Encoding UTF8
