"""
run_gcn_matched_classif.py
--------------------------
Plain GCN on the MoleculeNet classification sets, matched to moegcn_classif.py:
  - same scaffold_split() (no random fallback), same ToFloat transform
  - same Optuna HPO: 30 trials, TPE(seed=42), same search space minus num_experts/top_k
  - same EPOCHS / PATIENCE / BATCH_SIZE / scheduler / early stopping
  - 5 final seeds (0-4), matching MoE-GCN after seed extension
Output: results_gcn_matched_classif.json   (resumable; one dataset at a time)

Run small sets first:  python run_gcn_matched_classif.py --datasets BACE BBBP ClinTox SIDER Tox21
Then the slow ones:    python run_gcn_matched_classif.py --datasets HIV ToxCast
"""
import os, json, time, warnings, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 5
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_gcn_matched_classif.json"

DATASETS = [
    {"name": "BBBP",    "tasks": 1},
    {"name": "BACE",    "tasks": 1},
    {"name": "ClinTox", "tasks": 2},
    {"name": "SIDER",   "tasks": 27},
    {"name": "Tox21",   "tasks": 12},
    {"name": "HIV",     "tasks": 1},
    {"name": "ToxCast", "tasks": 617},
]
_ap = argparse.ArgumentParser(); _ap.add_argument("--datasets", nargs="*", default=None)
_args = _ap.parse_args()
if _args.datasets:
    DATASETS = [d for d in DATASETS if d["name"] in _args.datasets]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Transform: cast x to float32 at load time ────────────────────────────────
def ToFloat(data):
    data.x = data.x.float()
    return data

# ── Plain GCN (no expert layer) ──────────────────────────────────────────────
class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x), torch.zeros((), device=x.device)

# ── Scaffold Split ────────────────────────────────────────────────────────────
def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))

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

# Copied verbatim from fix_bbbp_bace.py: the split used for MoE-GCN on BBBP and BACE
def stratified_scaffold_split(dataset, frac_train=0.8, frac_val=0.1, seed=42):
    """
    Groups molecules by scaffold, then ensures both classes appear in val/test
    by distributing minority-class scaffolds across splits.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    # Get scaffold for each molecule
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)

    # Get label for each molecule (single task binary)
    labels = []
    for i in range(len(dataset)):
        y = dataset[i].y.numpy().flatten()[0]
        labels.append(int(y) if not np.isnan(y) else 0)
    labels = np.array(labels)

    # Sort scaffold groups: large scaffolds first, then by minority class presence
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))

    train_idx, val_idx, test_idx = [], [], []

    # First pass: fill train
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    # Check if val has both classes, if not swap some scaffolds
    def has_both_classes(idx_list):
        if len(idx_list) == 0:
            return False
        y = labels[idx_list]
        return len(np.unique(y)) >= 2

    # If val missing a class, move some train scaffolds to fix it
    if not has_both_classes(val_idx):
        minority_class = 0 if (labels == 0).sum() < (labels == 1).sum() else 1
        # Find train indices with minority class and move smallest scaffold containing them
        rng = np.random.RandomState(seed)
        for s in scaffold_sets:
            if all(i in train_idx for i in s):
                if any(labels[i] == minority_class for i in s):
                    for i in s:
                        train_idx.remove(i)
                    val_idx.extend(s)
                    if has_both_classes(val_idx):
                        break

    if not has_both_classes(test_idx):
        minority_class = 0 if (labels == 0).sum() < (labels == 1).sum() else 1
        for s in scaffold_sets:
            if all(i in train_idx for i in s):
                if any(labels[i] == minority_class for i in s):
                    for i in s:
                        train_idx.remove(i)
                    test_idx.extend(s)
                    if has_both_classes(test_idx):
                        break

    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate(model, loader, num_tasks):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            out, _ = model(batch)                      # [B, num_tasks]
            probs  = torch.sigmoid(out).cpu().numpy()  # [B, num_tasks]
            y      = batch.y.cpu().numpy()             # [B, 1] or [B, T]
            if y.ndim == 1:
                y = y.reshape(-1, num_tasks)
            all_preds.append(probs)
            all_labels.append(y)

    preds  = np.vstack(all_preds)   # [N, num_tasks]
    labels = np.vstack(all_labels)  # [N, num_tasks]

    aucs = []
    for t in range(num_tasks):
        col  = labels[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 2 or len(np.unique(col[mask])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(col[mask], preds[mask, t]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0

# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y = batch.y.float()
        if y.ndim == 1:
            y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01 * bal_loss
        loss.backward()
        optimizer.step()

# ── Optuna Objective ──────────────────────────────────────────────────────────
def make_objective(train_data, val_data, in_dim, num_tasks):
    def objective(trial):
        hidden      = trial.suggest_categorical("hidden", [128, 256])
        num_layers  = trial.suggest_int("num_layers", 2, 4)
        dropout     = trial.suggest_float("dropout", 0.0, 0.3)
        lr          = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd          = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        model = PlainGCN(in_dim, hidden, num_layers, dropout, num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl    = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl    = DataLoader(val_data,   batch_size=BATCH_SIZE)

        best_val, pat = 0.0, 0
        for _ in range(EPOCHS):
            train_epoch(model, tl, opt, num_tasks)
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val, pat = val_auc, 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break
        return best_val
    return objective

# ── Main Loop ─────────────────────────────────────────────────────────────────
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} datasets already done")
else:
    results = {}

for ds in DATASETS:
    name      = ds["name"]
    num_tasks = ds["tasks"]

    if name in results:
        print(f"  Skipping {name} (already done)")
        continue

    print(f"\n{'='*55}")
    print(f"  Plain GCN | {name} | {num_tasks} tasks | ROC-AUC")
    print(f"{'='*55}")
    t0 = time.time()

    # KEY FIX: apply ToFloat transform so x is float32 from the start
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    if name in ("BBBP", "BACE"):
        train_data, val_data, test_data = stratified_scaffold_split(dataset)
    else:
        train_data, val_data, test_data = scaffold_split(dataset)

    # Optuna
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(make_objective(train_data, val_data, in_dim, num_tasks), n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"  Best val AUC: {study.best_value:.4f} | {best_params}")

    # Final eval
    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PlainGCN(
            in_dim,
            best_params["hidden"],
            best_params["num_layers"],
            best_params["dropout"],
            num_tasks,
        ).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_val, pat = 0.0, 0
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(EPOCHS):
            train_epoch(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt, num_tasks)
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val   = val_auc
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break

        model.load_state_dict(best_state)
        test_auc = evaluate(model, tl, num_tasks)
        seed_scores.append(test_auc)
        print(f"    Seed {seed} → AUC: {test_auc:.4f}")

    mean_s  = float(np.mean(seed_scores))
    std_s   = float(np.std(seed_scores))
    elapsed = time.time() - t0

    results[name] = {
        "mean": mean_s, "std": std_s,
        "metric": "roc_auc", "seeds": seed_scores,
        "best_params": best_params, "time_min": round(elapsed / 60, 1),
        "split": "stratified_scaffold" if name in ("BBBP", "BACE") else "scaffold",
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ GCN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  MoE-GCN CLASSIFICATION SUMMARY")
print(f"{'='*55}")
print(f"{'Dataset':14} {'ROC-AUC':>10} {'±Std':>8}")
print("-" * 35)
for name, r in results.items():
    print(f"{name:14} {r['mean']:>10.4f} {r['std']:>8.4f}")
print(f"\nSaved → {SAVE_PATH}")
