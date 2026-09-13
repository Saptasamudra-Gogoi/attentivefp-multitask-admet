"""
lambda_sweep.py
================
P0-6 fix. The manuscript reports one expert handling ~60% of molecules
while another collapses to a few percent, calling it "self-organization
through positive feedback" -- but lambda (the load-balancing coefficient)
was never varied, so collapse-vs-self-organization cannot currently be
distinguished. random_partition_null.py already showed ppbr_az collapses
to n_active=1 on ALL 5 seeds at lambda=0.01 (the current fixed value) --
that is direct evidence collapse actually happens, not a hypothetical.

This sweeps lambda in {0, 0.001, 0.01, 0.1, 1} on two datasets:
  - ppbr_az            : the confirmed collapse case (regression)
  - solubility_aqsoldb : a dataset with clean, robust specialization
                         at lambda=0.01 (16/16 seed x descriptor null
                         clearance from random_partition_null.py) --
                         the accuracy/specialization-preservation check

3 seeds each. For every (dataset, lambda, seed) run, records:
  - final test metric (mae)
  - final load distribution (fraction of molecules per expert)
  - load entropy (normalized, 1.0 = perfectly uniform, 0 = total collapse)
  - per-epoch entropy trace, to see whether an exploration-to-
    specialization transition occurs and at what lambda it survives
  - mean eta2 across 8 descriptors on the final routing (specialization
    strength), so we can see the accuracy/specialization tradeoff as
    lambda increases

Uses the SAME model class, same per-dataset hidden/num_layers/dropout/
lr/wd/num_experts/top_k as retrain_and_extract.py's BEST_PARAMS --
only the loss coefficient on balance_loss changes. This isolates
lambda as the only variable.

Run: python lambda_sweep.py
Output: lambda_sweep_results.json
"""

import json, os, time, warnings, copy
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0]
SEEDS = [0, 1, 2]
EPOCHS = 150
PATIENCE = 20
BATCH = 64
SAVE_PATH = "lambda_sweep_results.json"

DATASETS = {
    "ppbr_az": {
        "cls": "ADME", "tdc_name": "PPBR_AZ",
        "hidden": 256, "num_layers": 3, "dropout": 0.002,
        "num_experts": 16, "top_k": 2,
        "lr": 0.000200, "weight_decay": 2.89e-05,
        "metric": "mae",
    },
    "solubility_aqsoldb": {
        "cls": "ADME", "tdc_name": "Solubility_AqSolDB",
        "hidden": 256, "num_layers": 4, "dropout": 0.153,
        "num_experts": 8, "top_k": 3,
        "lr": 0.000896, "weight_decay": 1.99e-05,
        "metric": "mae",
    },
}

DESCRIPTOR_FNS = {
    "MW":        Descriptors.ExactMolWt,
    "LogP":      Descriptors.MolLogP,
    "HBA":       rdMolDescriptors.CalcNumHBA,
    "HBD":       rdMolDescriptors.CalcNumHBD,
    "TPSA":      Descriptors.TPSA,
    "RotBonds":  rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "ArRings":   rdMolDescriptors.CalcNumAromaticRings,
}


class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x, return_routing=False):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        if return_routing:
            return out, balance_loss, weights
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data, return_routing=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        if return_routing:
            out, bal, weights = self.moe(x, return_routing=True)
            pred = self.head(out).squeeze(-1)
            return pred, bal, weights
        out, bal = self.moe(x)
        return self.head(out).squeeze(-1), bal


def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(), int(atom.GetChiralTag()),
                atom.GetDegree(), atom.GetFormalCharge(),
                atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ])
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]
        edge_index = (torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                      if edge_index else torch.zeros((2, 0), dtype=torch.long))
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


def build_dataset(smiles_list, labels):
    data_list, valid_smiles = [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(str(smi))
        if g is None:
            continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        data_list.append(g)
        valid_smiles.append(str(smi))
    return data_list, valid_smiles


def load_tdc(tdc_name, cls_name):
    from tdc.single_pred import ADME, Tox
    Cls = ADME if cls_name == "ADME" else Tox
    data = Cls(name=tdc_name)
    return data.get_split(method="scaffold", seed=42)


def load_entropy(dominant, num_experts):
    counts = np.bincount(dominant, minlength=num_experts).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log(probs))
    max_ent = np.log(num_experts)
    return float(ent / max_ent) if max_ent > 0 else 0.0


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0


def train_one(params, lam, train_data, val_data, seed, num_experts):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MoEGCN(in_dim=9, hidden=params["hidden"], num_layers=params["num_layers"],
                   dropout=params["dropout"], num_experts=num_experts,
                   top_k=params["top_k"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                  weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    drop_last = len(train_data) % BATCH == 1
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=256)

    best_val, best_state, patience_count = float("inf"), None, 0
    entropy_trace = []

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out, bal = model(batch)
            y = batch.y.squeeze()
            if y.dim() == 0: y = y.unsqueeze(0)
            if out.dim() == 0: out = out.unsqueeze(0)
            loss = F.mse_loss(out, y) + lam * bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        dom_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                _, _, weights = model(batch, return_routing=True)
                dom_list.append(weights.argmax(dim=-1).cpu().numpy())
        dominant_val = np.concatenate(dom_list)
        entropy_trace.append(load_entropy(dominant_val, num_experts))

        val_preds, val_truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out, _ = model(batch)
                if out.dim() == 0: out = out.unsqueeze(0)
                val_preds.extend(out.cpu().numpy().tolist())
                val_truths.extend(batch.y.cpu().numpy().flatten().tolist())
        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_truths))))
        scheduler.step(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, entropy_trace


def evaluate_final(model, test_data, all_data, all_smi, num_experts):
    test_loader = DataLoader(test_data, batch_size=256)
    preds, truths = [], []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            if out.dim() == 0: out = out.unsqueeze(0)
            preds.extend(out.cpu().numpy().tolist())
            truths.extend(batch.y.cpu().numpy().flatten().tolist())
    test_mae = float(np.mean(np.abs(np.array(preds) - np.array(truths))))

    full_loader = DataLoader(all_data, batch_size=256)
    dom_list = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(DEVICE)
            _, _, weights = model(batch, return_routing=True)
            dom_list.append(weights.argmax(dim=-1).cpu().numpy())
    dominant = np.concatenate(dom_list)

    n_active = len(np.unique(dominant))
    entropy = load_entropy(dominant, num_experts)
    load_frac = (np.bincount(dominant, minlength=num_experts) / len(dominant)).tolist()

    desc_vals = {k: [] for k in DESCRIPTOR_FNS}
    for smi in all_smi:
        mol = Chem.MolFromSmiles(smi)
        for name, fn in DESCRIPTOR_FNS.items():
            try:
                desc_vals[name].append(float(fn(mol)))
            except Exception:
                desc_vals[name].append(np.nan)

    eta2_list = []
    for name, vals in desc_vals.items():
        vals = np.array(vals)
        mask = ~np.isnan(vals)
        groups = [vals[mask][dominant[mask] == e] for e in np.unique(dominant)
                  if (dominant[mask] == e).sum() >= 3]
        if len(groups) >= 2:
            eta2_list.append(eta_squared(groups))

    mean_eta2 = float(np.mean(eta2_list)) if eta2_list else 0.0

    return {
        "test_mae": test_mae,
        "n_active_experts": n_active,
        "load_entropy": entropy,
        "load_fractions": load_frac,
        "mean_eta2": mean_eta2,
    }


def main():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            results = json.load(f)
        print(f"Resuming -- {len(results)} entries done")
    else:
        results = {}

    for ds_key, cfg in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_key}")
        print(f"{'='*70}")

        split = load_tdc(cfg["tdc_name"], cfg["cls"])
        train_data, train_smi = build_dataset(split["train"]["Drug"], split["train"]["Y"])
        val_data, val_smi = build_dataset(split["valid"]["Drug"], split["valid"]["Y"])
        test_data, test_smi = build_dataset(split["test"]["Drug"], split["test"]["Y"])
        all_data = train_data + val_data + test_data
        all_smi = train_smi + val_smi + test_smi

        print(f"  n={len(all_data)} molecules")

        for lam in LAMBDAS:
            for seed in SEEDS:
                key = f"{ds_key}__lambda{lam}__seed{seed}"
                if key in results:
                    print(f"  Skipping {key} (done)")
                    continue

                print(f"\n  [{ds_key} | lambda={lam} | seed={seed}]")
                t0 = time.time()

                model, entropy_trace = train_one(
                    cfg, lam, train_data, val_data, seed, cfg["num_experts"])
                final = evaluate_final(model, test_data, all_data, all_smi, cfg["num_experts"])

                elapsed = (time.time() - t0) / 60
                entry = {
                    "dataset": ds_key,
                    "lambda": lam,
                    "seed": seed,
                    "test_mae": final["test_mae"],
                    "n_active_experts": final["n_active_experts"],
                    "num_experts": cfg["num_experts"],
                    "load_entropy": final["load_entropy"],
                    "load_fractions": final["load_fractions"],
                    "mean_eta2": final["mean_eta2"],
                    "entropy_trace": entropy_trace,
                    "time_min": round(elapsed, 1),
                }
                results[key] = entry
                print(f"    mae={final['test_mae']:.4f}  n_active={final['n_active_experts']}/"
                      f"{cfg['num_experts']}  entropy={final['load_entropy']:.3f}  "
                      f"eta2={final['mean_eta2']:.4f}  ({elapsed:.1f} min)")

                with open(SAVE_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\n{'='*90}")
    print("  LAMBDA SWEEP SUMMARY (mean across seeds)")
    print(f"{'='*90}")
    print(f"  {'Dataset':<22} {'Lambda':>8} {'MAE':>10} {'Entropy':>10} {'Eta2':>10} {'N_active':>10}")
    print(f"  {'-'*80}")
    for ds_key in DATASETS:
        for lam in LAMBDAS:
            entries = [v for k, v in results.items()
                       if v["dataset"] == ds_key and v["lambda"] == lam]
            if not entries:
                continue
            mae = np.mean([e["test_mae"] for e in entries])
            ent = np.mean([e["load_entropy"] for e in entries])
            eta2 = np.mean([e["mean_eta2"] for e in entries])
            n_act = np.mean([e["n_active_experts"] for e in entries])
            print(f"  {ds_key:<22} {lam:>8} {mae:>10.4f} {ent:>10.3f} {eta2:>10.4f} {n_act:>10.1f}")

    print(f"\nSaved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
