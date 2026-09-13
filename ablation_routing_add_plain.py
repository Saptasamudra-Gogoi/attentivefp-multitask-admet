"""
ablation_routing_add_plain.py
===============================
Adds a Plain-GCN mode to the SAME protocol as ablation_routing.py --
the script that actually produced ablation_routing_results.json, which
is what Table 6's published MoE-GCN/Dense-uniform/Dense-wide numbers
come from (verified: ESOL 1.1176, FreeSolv 3.0673, Lipo 0.7826, Caco-2
0.5395, Solubility 1.1355 -- all match Table 6 exactly).

This uses the IDENTICAL fixed hyperparameters, loss, seed count, and
data loading as ablation_routing.py (not the Optuna-tuned protocol from
ablation_optuna.py / ablation_add_plain_gcn.py, which produced a
DIFFERENT, non-comparable set of numbers that were mistakenly used
in the manuscript's Plain-GCN column previously -- this corrects that).

FIXED_HP = hidden=256, num_layers=3, dropout=0.1, lr=5e-4, wd=1e-5
Loss = MSE + 0.01 * balance_loss (balance_loss=0 for plain, so this
reduces to plain MSE -- kept for exact structural parity with the
train_and_eval function signature).
5 seeds, 100 epochs, patience 15, batch 64.
MoleculeNet: scaffold split. TDC: random 80/10/10 split (matching the
original script's TDC loader, including its 6-dim atom featurization --
this is a known inconsistency in the original protocol, not introduced
here; flagged for the record).

Merges into ablation_routing_results.json under the key
"Plain-GCN (no layer)" so Table 6 can add a fourth column consistent
with the other three.

Run: python ablation_routing_add_plain.py
"""

import json, os, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

from torch_geometric.data import DataLoader as GeoLoader
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS   = 5
EPOCHS    = 100
PATIENCE  = 15
BATCH     = 64
DATA_ROOT = "./data"

print(f"Device: {DEVICE}")

FIXED_HP = {
    'hidden': 256, 'num_layers': 3, 'dropout': 0.1,
    'lr': 5e-4, 'wd': 1e-5
}

MOLNET_DATASETS = ['ESOL', 'FreeSolv', 'Lipo']
TDC_DATASETS = ['caco2_wang', 'solubility_aqsoldb']

MODE_LABEL = 'Plain-GCN (no layer)'
SAVE_PATH = 'ablation_routing_results.json'


class PlainLayer(nn.Module):
    """No expert/dense/wide layer at all -- linear passthrough, matching
    the structural slot MoELayer/DenseLayer/WiderDenseLayer occupy in
    GCNWithLayer, but doing nothing beyond a single linear map."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x), torch.tensor(0.0, device=x.device)


class GCNPlain(nn.Module):
    """Same backbone shape as ablation_routing.py's GCNWithLayer, with
    the layer slot filled by PlainLayer instead of MoE/Dense/WiderDense."""
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.layer = PlainLayer(hidden, hidden)
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal = self.layer(x)
        return self.head(x), bal

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i, d in enumerate(dataset):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False) if mol else smi
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


def load_tdc_dataset(tdc_name):
    """Identical to ablation_routing.py's loader -- same 6-dim atom
    features, same random 80/10/10 split -- so results are comparable
    to the existing MoE-GCN/Dense-uniform/Dense-wide rows for this
    dataset in ablation_routing_results.json."""
    from tdc.single_pred import ADME, Tox
    try:
        data = ADME(name=tdc_name)
    except Exception:
        data = Tox(name=tdc_name)
    df = data.get_data()
    smiles = df['Drug'].tolist()
    labels = df['Y'].values.astype(float)

    dataset = []
    for smi, lab in zip(smiles, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats = []
        for atom in mol.GetAtoms():
            feats.append([
                atom.GetAtomicNum(), atom.GetDegree(),
                atom.GetFormalCharge(), int(atom.GetHybridization()),
                int(atom.GetIsAromatic()), atom.GetTotalNumHs(),
            ])
        x = torch.tensor(feats, dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[u, v], [v, u]]
        if not edges:
            continue
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        dataset.append(Data(x=x, edge_index=ei,
                             y=torch.tensor([lab], dtype=torch.float)))

    n = len(dataset)
    idx = np.random.permutation(n)
    t = int(0.8 * n); v = int(0.9 * n)
    train = torch.utils.data.Subset(dataset, idx[:t].tolist())
    val = torch.utils.data.Subset(dataset, idx[t:v].tolist())
    test = torch.utils.data.Subset(dataset, idx[v:].tolist())
    in_dim = dataset[0].x.shape[1]
    return train, val, test, in_dim


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds = np.array(preds)
    labels = np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2)))


def train_and_eval(train_data, val_data, test_data, in_dim,
                    hidden, num_layers, dropout, lr, wd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GCNPlain(in_dim=in_dim, hidden=hidden, num_layers=num_layers,
                      dropout=dropout, num_tasks=1).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    train_loader = GeoLoader(train_data, batch_size=BATCH, shuffle=True)
    val_loader = GeoLoader(val_data, batch_size=BATCH)
    test_loader = GeoLoader(test_data, batch_size=BATCH)

    best_val, best_state, patience_count = float('inf'), None, 0

    model.train()
    for ep in range(EPOCHS):
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out, bal = model(batch)
            labels = batch.y.float().squeeze()
            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue
            loss = F.mse_loss(out.squeeze()[mask], labels[mask]) + 0.01 * bal
            loss.backward()
            opt.step()

        val_rmse = evaluate(model, val_loader)
        sched.step(val_rmse)
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
    n_params = model.count_params()
    return test_rmse, n_params


def main():
    print("\n" + "=" * 70)
    print("  Adding Plain-GCN to ablation_routing_results.json")
    print(f"  Fixed protocol: {FIXED_HP}, {N_SEEDS} seeds")
    print("=" * 70)

    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded {SAVE_PATH} -- {len(all_results)} datasets present")
    else:
        print(f"  ERROR: {SAVE_PATH} not found. This script only ADDS a mode "
              f"to the existing file; run ablation_routing.py first.")
        return

    # -- MoleculeNet --
    for ds_name in MOLNET_DATASETS:
        if ds_name in all_results and MODE_LABEL in all_results[ds_name]:
            print(f"\n  Skipping {ds_name} (Plain-GCN already done)")
            continue

        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")
        dataset = MoleculeNet(root=DATA_ROOT, name=ds_name)
        in_dim = dataset.num_node_features
        train_d, val_d, test_d = scaffold_split(dataset)

        scores = []
        for seed in range(N_SEEDS):
            t0 = time.time()
            rmse, n_p = train_and_eval(
                train_d, val_d, test_d, in_dim,
                FIXED_HP['hidden'], FIXED_HP['num_layers'],
                FIXED_HP['dropout'], FIXED_HP['lr'], FIXED_HP['wd'], seed
            )
            scores.append(rmse)
            print(f"  {MODE_LABEL:<30} seed={seed} RMSE={rmse:.4f} "
                  f"params={n_p:,} ({time.time()-t0:.1f}s)")

        if ds_name not in all_results:
            all_results[ds_name] = {}
        all_results[ds_name][MODE_LABEL] = {
            'mean': round(float(np.mean(scores)), 4),
            'std': round(float(np.std(scores)), 4),
            'seeds': [round(s, 4) for s in scores],
            'n_params': n_p,
        }
        print(f"  -> {MODE_LABEL}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [SAVED] {SAVE_PATH}")

    # -- TDC --
    for ds_name in TDC_DATASETS:
        if ds_name in all_results and MODE_LABEL in all_results[ds_name]:
            print(f"\n  Skipping {ds_name} (Plain-GCN already done)")
            continue

        print(f"\n{'='*60}\n  {ds_name} (TDC)\n{'='*60}")
        train_d, val_d, test_d, in_dim = load_tdc_dataset(ds_name)

        scores = []
        for seed in range(N_SEEDS):
            t0 = time.time()
            rmse, n_p = train_and_eval(
                train_d, val_d, test_d, in_dim,
                FIXED_HP['hidden'], FIXED_HP['num_layers'],
                FIXED_HP['dropout'], FIXED_HP['lr'], FIXED_HP['wd'], seed
            )
            scores.append(rmse)
            print(f"  {MODE_LABEL:<30} seed={seed} RMSE={rmse:.4f} "
                  f"({time.time()-t0:.1f}s)")

        if ds_name not in all_results:
            all_results[ds_name] = {}
        all_results[ds_name][MODE_LABEL] = {
            'mean': round(float(np.mean(scores)), 4),
            'std': round(float(np.std(scores)), 4),
            'seeds': [round(s, 4) for s in scores],
            'n_params': n_p,
        }
        print(f"  -> {MODE_LABEL}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [SAVED] {SAVE_PATH}")

    print(f"\n{'='*70}")
    print("  ALL 5 DATASETS NOW HAVE Plain-GCN UNDER THE MATCHED PROTOCOL")
    print(f"{'='*70}")
    for ds in MOLNET_DATASETS + TDC_DATASETS:
        if ds in all_results and MODE_LABEL in all_results[ds]:
            v = all_results[ds][MODE_LABEL]
            print(f"  {ds:<22} {v['mean']:.4f} +/- {v['std']:.4f}")


if __name__ == "__main__":
    main()
