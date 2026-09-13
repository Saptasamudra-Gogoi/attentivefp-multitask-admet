"""
Lipo only — MoE-GIN, MoE-GCN, MoE-DMPNN
Results saved to: results_phase3_fixed.json
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, GRUCell
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, GCNConv, NNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
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

BACKBONES = ["MoE-GIN", "MoE-GCN", "MoE-DMPNN"]

print(f"Device : {DEVICE}")
print(f"Dataset: Lipo x {BACKBONES}\n")

results = {}
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    for bb in list(results.keys()):
        if "Lipo" in results.get(bb, {}):
            del results[bb]["Lipo"]
            print(f"Cleared {bb}/Lipo — will rerun")

def to1d(t): return t.reshape(-1)

def mse_loss(out, tgt):
    out, tgt = to1d(out), to1d(tgt)
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.mse_loss(out[mask], tgt[mask])

def rmse_score(preds, targets):
    p = np.array(preds).flatten()
    t = np.array(targets).flatten()
    mask = ~np.isnan(t)
    return float(np.sqrt(np.mean((p[mask] - t[mask])**2)))

class SparseMoE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, ne, k):
        super().__init__()
        self.E = ne; self.K = k
        self.exps = torch.nn.ModuleList([
            Sequential(Linear(in_dim, out_dim), torch.nn.GELU(), Linear(out_dim, out_dim))
            for _ in range(ne)
        ])
        self.gate = Linear(in_dim, ne, bias=False)
        self._lb  = torch.tensor(0.0)

    def forward(self, x):
        g = self.gate(x)
        v, i = torch.topk(g, self.K, dim=-1)
        w = F.softmax(v, dim=-1)
        out = torch.zeros(x.size(0), self.exps[0][-1].out_features, device=x.device)
        for k in range(self.K):
            idx = i[:, k]; wk = w[:, k:k+1]
            for e in range(self.E):
                m = (idx == e)
                if m.any():
                    out[m] = out[m] + wk[m] * self.exps[e](x[m])
        p = F.softmax(g, dim=-1)
        self._lb = (p.mean(0)**2).sum() * self.E
        return out

    def lb(self): return self._lb

class MoEGIN(torch.nn.Module):
    def __init__(self, ic, h, nl, dr, ne, k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(nl):
            c = ic if i == 0 else h
            self.convs.append(GINConv(Sequential(Linear(c,h), BatchNorm1d(h), ReLU(), Linear(h,h)), train_eps=True))
            self.bns.append(BatchNorm1d(h))
        self.moe = SparseMoE(h, h, ne, k)
        self.dr  = dr
        self.lin = Linear(h, 1)

    def forward(self, x, ei, ea, batch):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), p=self.dr, training=self.training)
        return self.lin(self.moe(global_add_pool(x, batch))).reshape(-1)

    def lb(self): return self.moe.lb()

class MoEGCN(torch.nn.Module):
    def __init__(self, ic, h, nl, dr, ne, k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(nl):
            c = ic if i == 0 else h
            self.convs.append(GCNConv(c, h))
            self.bns.append(BatchNorm1d(h))
        self.moe = SparseMoE(h, h, ne, k)
        self.dr  = dr
        self.lin = Linear(h, 1)

    def forward(self, x, ei, ea, batch):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), p=self.dr, training=self.training)
        return self.lin(self.moe(global_mean_pool(x, batch))).reshape(-1)

    def lb(self): return self.moe.lb()

class MoEDMPNN(torch.nn.Module):
    def __init__(self, ic, ec, h, nl, dr, ne, k):
        super().__init__()
        self.proj  = Linear(ic, h)
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for _ in range(nl):
            self.convs.append(NNConv(h, h, Sequential(Linear(ec, h*h)), aggr='mean'))
            self.bns.append(BatchNorm1d(h))
        self.gru = GRUCell(h, h)
        self.moe = SparseMoE(h, h, ne, k)
        self.dr  = dr
        self.lin = Linear(h, 1)

    def forward(self, x, ei, ea, batch):
        h = F.relu(self.proj(x))
        for c, b in zip(self.convs, self.bns):
            m = F.dropout(F.relu(b(c(h, ei, ea))), p=self.dr, training=self.training)
            h = self.gru(m, h)
        return self.lin(self.moe(global_mean_pool(h, batch))).reshape(-1)

    def lb(self): return self.moe.lb()

def build(bb, ic, ec, h, nl, dr, ne, k):
    if bb == "MoE-GIN":   return MoEGIN(ic, h, nl, dr, ne, k).to(DEVICE)
    if bb == "MoE-GCN":   return MoEGCN(ic, h, nl, dr, ne, k).to(DEVICE)
    if bb == "MoE-DMPNN": return MoEDMPNN(ic, ec, h, nl, dr, ne, k).to(DEVICE)

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        try:
            mol = Chem.MolFromSmiles(data.smiles)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except: sc = ""
        scaffolds[sc].append(i)
    sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset); nt = int(n*0.8); nv = int(n*0.1)
    tr, va, te = [], [], []
    for s in sets:
        if   len(tr)+len(s) <= nt: tr.extend(s)
        elif len(va)+len(s) <= nv: va.extend(s)
        else:                       te.extend(s)
    return tr, va, te

def train_one(model, loader, opt):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt  = to1d(batch.y.float())
        loss = mse_loss(out, tgt) + LB_WEIGHT * model.lb()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

@torch.no_grad()
def eval_one(model, loader):
    model.eval()
    ps, ts = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        ps.extend(out.cpu().numpy().flatten().tolist())
        ts.extend(to1d(batch.y).cpu().numpy().flatten().tolist())
    return rmse_score(ps, ts)

def full_run(trl, vl, tel, p, bb, ic, ec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ne = p["num_experts"]; k = min(p["top_k"], ne)
    model = build(bb, ic, ec, p["hidden"], p["num_layers"], p["dropout"], ne, k)
    opt   = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7, min_lr=1e-6)
    best_val, best_state, pat = 1e9, None, 0
    for ep in range(1, EPOCHS+1):
        train_one(model, trl, opt)
        v = eval_one(model, vl)
        sch.step(v)
        if v < best_val:
            best_val = v
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state is None:
        return 999.0
    model.load_state_dict(best_state)
    return eval_one(model, tel)

def make_obj(trl, vl, bb, ic, ec):
    def obj(trial):
        p = {
            "hidden"      : trial.suggest_categorical("hidden",      [64, 128, 256]),
            "num_layers"  : trial.suggest_int("num_layers",          2, 5),
            "dropout"     : trial.suggest_float("dropout",           0.0, 0.5),
            "num_experts" : trial.suggest_categorical("num_experts", [4, 8, 16]),
            "top_k"       : trial.suggest_int("top_k",               1, 4),
            "lr"          : trial.suggest_float("lr",                1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay",      1e-6, 1e-3, log=True),
        }
        p["top_k"] = min(p["top_k"], p["num_experts"])
        torch.manual_seed(trial.number * 7 + 13)
        model = build(bb, ic, ec, p["hidden"], p["num_layers"], p["dropout"],
                      p["num_experts"], p["top_k"])
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        best, pat = 1e9, 0
        for ep in range(1, EPOCHS+1):
            train_one(model, trl, opt)
            v = eval_one(model, vl)
            if v < best: best, pat = v, 0
            else:
                pat += 1
                if pat >= PATIENCE: break
            trial.report(-v, ep)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return -best
    return obj

# ── Load data once ────────────────────────────────────────────────────────────
dataset = MoleculeNet(root=DATA_ROOT, name="Lipo")
ic      = dataset[0].x.shape[1]
ec      = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3
tri, vai, tei = scaffold_split(dataset)
trl = DataLoader([dataset[i] for i in tri], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
vl  = DataLoader([dataset[i] for i in vai], batch_size=BATCH_SIZE, shuffle=False)
tel = DataLoader([dataset[i] for i in tei], batch_size=BATCH_SIZE, shuffle=False)
print(f"Lipo → Tr:{len(tri)} Va:{len(vai)} Te:{len(tei)}")

# ── Main ─────────────────────────────────────────────────────────────────────
for bb in BACKBONES:
    if bb not in results:
        results[bb] = {}

    if results[bb].get("Lipo", {}).get("mean", 999) < 999:
        print(f"  [{bb}] Lipo already done ({results[bb]['Lipo']['mean']:.4f}), skipping.")
        continue

    print(f"\n{'='*50}")
    print(f"  {bb} | Lipo")
    print(f"{'='*50}")
    t0 = time.time()

    study = optuna.create_study(
        study_name = f"{bb}_Lipo_{int(time.time())}",
        direction  = "maximize",
        sampler    = TPESampler(seed=int(time.time()) % 99991),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15),
    )
    study.optimize(make_obj(trl, vl, bb, ic, ec),
                   n_trials=N_TRIALS, timeout=3600, show_progress_bar=False)

    bp = study.best_params
    bp["top_k"] = min(bp["top_k"], bp["num_experts"])
    print(f"  Best val RMSE: {-study.best_value:.4f} | {bp}")

    rmses = []
    for seed in [0, 1, 2]:
        r = full_run(trl, vl, tel, bp, bb, ic, ec, seed)
        rmses.append(r)
        print(f"    Seed {seed} → RMSE: {r:.4f}")

    mean_r = float(np.mean(rmses))
    std_r  = float(np.std(rmses))
    elapsed = time.time() - t0

    results[bb]["Lipo"] = {
        "metric": "rmse", "mean": mean_r, "std": std_r,
        "seeds": rmses, "best_params": bp,
        "time_min": round(elapsed/60, 1),
    }
    print(f"  ✓ [{bb}] Lipo: {mean_r:.4f} ± {std_r:.4f}  ({elapsed/60:.1f} min)")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {SAVE_PATH}")

print(f"\n{'='*50}")
print("DONE — Lipo results:")
for bb in BACKBONES:
    r = results.get(bb, {}).get("Lipo", {})
    if r:
        print(f"  {bb:<14} RMSE: {r['mean']:.4f} ± {r['std']:.4f}")
