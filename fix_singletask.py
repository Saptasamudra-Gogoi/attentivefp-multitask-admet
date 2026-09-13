"""
Fix script for single-task classification datasets: BBBP, BACE, HIV
These failed in phase3_fixed.py due to (N,1) vs (N,) shape mismatch in final eval.
Results merged into results_phase3_fixed.json
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import NNConv
from torch.nn import GRUCell
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 3
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_phase3_fixed.json"
LB_WEIGHT  = 0.01

DATASETS = [
    {"name": "BBBP", "tasks": 1, "metric": "roc_auc"},
    {"name": "BACE", "tasks": 1, "metric": "roc_auc"},
    {"name": "HIV",  "tasks": 1, "metric": "roc_auc"},
]
BACKBONES = ["MoE-GIN", "MoE-GCN", "MoE-DMPNN"]

print(f"Device : {DEVICE}")
print(f"Fixing single-task datasets: {[d['name'] for d in DATASETS]}")
print(f"Backbones: {BACKBONES}\n")

# Load existing results
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    # Remove broken 0.0 entries so they rerun
    for bb in list(results.keys()):
        for ds in list(results[bb].keys()):
            if results[bb][ds].get("mean", 1) == 0.0:
                del results[bb][ds]
                print(f"Cleared bad result: {bb}/{ds}")
else:
    results = {}


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE-SAFE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def flatten(t):
    """Flatten any tensor to 1D by squeezing all size-1 dims."""
    while t.dim() > 1 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.dim() > 1:
        t = t.squeeze()
    return t

def masked_bce(out, tgt):
    out, tgt = flatten(out), flatten(tgt)
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.binary_cross_entropy_with_logits(out[mask], tgt[mask])

def roc_auc_safe(probs, targets):
    probs   = np.array(probs).flatten()
    targets = np.array(targets).flatten()
    mask    = ~np.isnan(targets)
    if mask.sum() < 2 or len(np.unique(targets[mask])) < 2:
        return 0.0
    try:
        return float(roc_auc_score(targets[mask], probs[mask]))
    except:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE MOE
# ══════════════════════════════════════════════════════════════════════════════

class SparseMoE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = torch.nn.ModuleList([
            Sequential(Linear(in_dim, out_dim), torch.nn.GELU(), Linear(out_dim, out_dim))
            for _ in range(num_experts)
        ])
        self.gate = Linear(in_dim, num_experts, bias=False)
        self._lb  = torch.tensor(0.0)

    def forward(self, x):
        logits       = self.gate(x)
        topk_v, topk_i = torch.topk(logits, self.top_k, dim=-1)
        weights      = F.softmax(topk_v, dim=-1)
        out          = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for k in range(self.top_k):
            idx = topk_i[:, k]
            w   = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])
        probs    = F.softmax(logits, dim=-1)
        self._lb = (probs.mean(0) * probs.mean(0)).sum() * self.num_experts
        return out

    def lb_loss(self):
        return self._lb


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class MoEGIN(torch.nn.Module):
    def __init__(self, in_ch, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = in_ch if i == 0 else hidden
            mlp = Sequential(Linear(inc, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden))
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, 1)  # always 1 output, flatten later

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.moe(x)
        return self.lin(x).squeeze(-1)   # always (N,)

    def lb_loss(self):
        return self.moe.lb_loss()


class MoEGCN(torch.nn.Module):
    def __init__(self, in_ch, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = in_ch if i == 0 else hidden
            self.convs.append(GCNConv(inc, hidden))
            self.bns.append(BatchNorm1d(hidden))
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.moe(x)
        return self.lin(x).squeeze(-1)   # always (N,)

    def lb_loss(self):
        return self.moe.lb_loss()


class MoEDMPNN(torch.nn.Module):
    def __init__(self, in_ch, edge_ch, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.input_proj = Linear(in_ch, hidden)
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = Sequential(Linear(edge_ch, hidden * hidden))
            self.convs.append(NNConv(hidden, hidden, nn_edge, aggr='mean'))
            self.bns.append(BatchNorm1d(hidden))
        self.gru     = GRUCell(hidden, hidden)
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.input_proj(x))
        h = x
        for conv, bn in zip(self.convs, self.bns):
            m = F.relu(bn(conv(h, edge_index, edge_attr)))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
        x = global_mean_pool(h, batch)
        x = self.moe(x)
        return self.lin(x).squeeze(-1)   # always (N,)

    def lb_loss(self):
        return self.moe.lb_loss()


def build_model(backbone, in_ch, edge_ch, hidden, num_layers, dropout, num_experts, top_k):
    if backbone == "MoE-GIN":
        return MoEGIN(in_ch, hidden, num_layers, dropout, num_experts, top_k).to(DEVICE)
    elif backbone == "MoE-GCN":
        return MoEGCN(in_ch, hidden, num_layers, dropout, num_experts, top_k).to(DEVICE)
    elif backbone == "MoE-DMPNN":
        return MoEDMPNN(in_ch, edge_ch, hidden, num_layers, dropout, num_experts, top_k).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# SCAFFOLD SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        try:
            mol = Chem.MolFromSmiles(data.smiles)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except:
            sc = ""
        scaffolds[sc].append(i)
    sets    = sorted(scaffolds.values(), key=len, reverse=True)
    n       = len(dataset)
    n_train = int(n * frac_train)
    n_val   = int(n * frac_val)
    train_idx, val_idx, test_idx = [], [], []
    for s in sets:
        if   len(train_idx) + len(s) <= n_train: train_idx.extend(s)
        elif len(val_idx)   + len(s) <= n_val:   val_idx.extend(s)
        else:                                      test_idx.extend(s)
    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt  = flatten(batch.y.float())
        loss = masked_bce(out, tgt) + LB_WEIGHT * model.lb_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt   = flatten(batch.y).cpu().numpy()
        probs = torch.sigmoid(out).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(tgt.tolist())
    return roc_auc_safe(all_probs, all_targets)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_training(train_loader, val_loader, test_loader, params,
                 backbone, in_ch, edge_ch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(backbone, in_ch, edge_ch,
                        params["hidden"], params["num_layers"],
                        params["dropout"], params["num_experts"], params["top_k"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    best_val, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        val_auc = eval_epoch(model, val_loader)
        scheduler.step(val_auc)
        if val_auc > best_val:
            best_val     = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: break
    model.load_state_dict(best_state)
    return eval_epoch(model, test_loader)


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(train_loader, val_loader, backbone, in_ch, edge_ch):
    def objective(trial):
        params = {
            "hidden"      : trial.suggest_categorical("hidden",      [64, 128, 256]),
            "num_layers"  : trial.suggest_int("num_layers",          2, 5),
            "dropout"     : trial.suggest_float("dropout",           0.0, 0.5),
            "num_experts" : trial.suggest_categorical("num_experts", [4, 8, 16]),
            "top_k"       : trial.suggest_int("top_k",               1, 4),
            "lr"          : trial.suggest_float("lr",                1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay",      1e-6, 1e-3, log=True),
        }
        params["top_k"] = min(params["top_k"], params["num_experts"])
        torch.manual_seed(42)
        model = build_model(backbone, in_ch, edge_ch,
                            params["hidden"], params["num_layers"],
                            params["dropout"], params["num_experts"], params["top_k"])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params["lr"], weight_decay=params["weight_decay"])
        best_val, patience_ctr = 0.0, 0
        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer)
            val_auc = eval_epoch(model, val_loader)
            if val_auc > best_val:
                best_val, patience_ctr = val_auc, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE: break
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_val
    return objective


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

for backbone in BACKBONES:
    if backbone not in results:
        results[backbone] = {}

    for ds_cfg in DATASETS:
        name = ds_cfg["name"]

        if name in results[backbone]:
            print(f"  [{backbone}] {name} already done, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  {backbone} | {name}")
        print(f"{'='*60}")

        t0      = time.time()
        dataset = MoleculeNet(root=DATA_ROOT, name=name)
        in_ch   = dataset[0].x.shape[1]
        edge_ch = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3

        train_idx, val_idx, test_idx = scaffold_split(dataset)
        train_data = [dataset[i] for i in train_idx]
        val_data   = [dataset[i] for i in val_idx]
        test_data  = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

        print(f"  Split → Train:{len(train_data)} Val:{len(val_data)} Test:{len(test_data)}")

        study = optuna.create_study(
            study_name = f"{backbone}_{name}_{int(time.time())}",
            direction  = "maximize",
            sampler    = TPESampler(seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        )
        study.optimize(
            make_objective(train_loader, val_loader, backbone, in_ch, edge_ch),
            n_trials=N_TRIALS, timeout=3600, show_progress_bar=False,
        )
        best_params = study.best_params
        best_params["top_k"] = min(best_params["top_k"], best_params["num_experts"])
        print(f"  Best val AUC: {study.best_value:.4f}")
        print(f"  Params: {best_params}")

        seed_aucs = []
        for seed in [0, 1, 2]:
            auc = run_training(train_loader, val_loader, test_loader,
                               best_params, backbone, in_ch, edge_ch, seed)
            seed_aucs.append(auc)
            print(f"    Seed {seed} → AUC: {auc:.4f}")

        mean_auc = float(np.mean(seed_aucs))
        std_auc  = float(np.std(seed_aucs))
        elapsed  = time.time() - t0

        results[backbone][name] = {
            "metric"      : "roc_auc",
            "mean"        : mean_auc,
            "std"         : std_auc,
            "seeds"       : seed_aucs,
            "best_params" : best_params,
            "time_min"    : round(elapsed / 60, 1),
        }
        print(f"  ✓ [{backbone}] {name}: {mean_auc:.4f} ± {std_auc:.4f}  ({elapsed/60:.1f} min)")

        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {SAVE_PATH}")

print(f"\n{'='*60}")
print("SINGLE-TASK FIX COMPLETE")
print(f"{'='*60}")
for bb, datasets in results.items():
    for ds_name, r in datasets.items():
        if ds_name in ["BBBP", "BACE", "HIV"]:
            print(f"{bb:<14} {ds_name:<8} {r['mean']:.4f} ± {r['std']:.4f}")
