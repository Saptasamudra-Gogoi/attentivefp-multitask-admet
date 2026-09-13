"""
retrain_and_extract.py
=======================
Retrains MoE-GCN on all 22 TDC datasets and extracts routing
assignments from the SAME model that produces benchmark performance.

This gives internally consistent (routing, performance) pairs
needed for valid SSS computation.

Run: python retrain_and_extract.py

Outputs per dataset:
    models/moegcn_{dataset}_seed{s}_v2.pt   checkpoint
    routing_{dataset}_seed{s}_v2.npy        dominant expert per molecule
    routing_weights_{dataset}_seed{s}_v2.npy full routing matrix
    results_moegcn_v2.json                  performance + routing summary

Estimated time: 2-4 hours on GTX 1660 Ti
"""

import json
import copy
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import f_oneway, spearmanr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

SEEDS = [0, 1, 2, 3, 4]  # 5 seeds for robust results

BEST_PARAMS = {
    "solubility_aqsoldb": {
        "hidden":256,"num_layers":4,"dropout":0.153,"num_experts":8,
        "top_k":3,"lr":0.000896,"weight_decay":1.99e-05,
        "tdc_name":"Solubility_AqSolDB","task":"regression","metric":"mae"
    },
    "caco2_wang": {
        "hidden":256,"num_layers":3,"dropout":0.031,"num_experts":16,
        "top_k":4,"lr":0.000599,"weight_decay":2.01e-05,
        "tdc_name":"Caco2_Wang","task":"regression","metric":"mae"
    },
    "lipophilicity_astrazeneca": {
        "hidden":256,"num_layers":4,"dropout":0.038,"num_experts":4,
        "top_k":1,"lr":0.000972,"weight_decay":7.03e-05,
        "tdc_name":"Lipophilicity_AstraZeneca","task":"regression","metric":"mae"
    },
    "ppbr_az": {
        "hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,
        "top_k":2,"lr":0.000200,"weight_decay":2.89e-05,
        "tdc_name":"PPBR_AZ","task":"regression","metric":"mae"
    },
    "ld50_zhu": {
        "hidden":256,"num_layers":4,"dropout":0.099,"num_experts":16,
        "top_k":2,"lr":0.000470,"weight_decay":4.92e-05,
        "tdc_name":"LD50_Zhu","task":"regression","metric":"mae"
    },
    "vdss_lombardo": {
        "hidden":256,"num_layers":3,"dropout":0.222,"num_experts":8,
        "top_k":2,"lr":0.000905,"weight_decay":2.82e-05,
        "tdc_name":"VDss_Lombardo","task":"regression","metric":"spearman"
    },
    "half_life_obach": {
        "hidden":256,"num_layers":3,"dropout":0.220,"num_experts":16,
        "top_k":4,"lr":0.000539,"weight_decay":1.83e-05,
        "tdc_name":"Half_Life_Obach","task":"regression","metric":"spearman"
    },
    "clearance_microsome_az": {
        "hidden":256,"num_layers":3,"dropout":0.003,"num_experts":16,
        "top_k":2,"lr":0.000195,"weight_decay":2.91e-05,
        "tdc_name":"Clearance_Microsome_AZ","task":"regression","metric":"spearman"
    },
    "clearance_hepatocyte_az": {
        "hidden":256,"num_layers":3,"dropout":0.221,"num_experts":16,
        "top_k":2,"lr":0.000973,"weight_decay":1.05e-06,
        "tdc_name":"Clearance_Hepatocyte_AZ","task":"regression","metric":"spearman"
    },
    "hia_hou": {
        "hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,
        "top_k":4,"lr":0.000914,"weight_decay":2.15e-05,
        "tdc_name":"HIA_Hou","task":"classification","metric":"auroc"
    },
    "pgp_broccatelli": {
        "hidden":256,"num_layers":4,"dropout":0.002,"num_experts":4,
        "top_k":4,"lr":0.000127,"weight_decay":2.75e-05,
        "tdc_name":"Pgp_Broccatelli","task":"classification","metric":"auroc"
    },
    "bioavailability_ma": {
        "hidden":256,"num_layers":3,"dropout":0.209,"num_experts":4,
        "top_k":4,"lr":0.000808,"weight_decay":3.40e-05,
        "tdc_name":"Bioavailability_Ma","task":"classification","metric":"auroc"
    },
    "bbb_martins": {
        "hidden":256,"num_layers":4,"dropout":0.202,"num_experts":16,
        "top_k":4,"lr":0.000527,"weight_decay":1.27e-05,
        "tdc_name":"BBB_Martins","task":"classification","metric":"auroc"
    },
    "cyp2d6_veith": {
        "hidden":256,"num_layers":3,"dropout":0.209,"num_experts":4,
        "top_k":4,"lr":0.000624,"weight_decay":2.61e-05,
        "tdc_name":"CYP2D6_Veith","task":"classification","metric":"auroc"
    },
    "cyp3a4_veith": {
        "hidden":256,"num_layers":4,"dropout":0.036,"num_experts":4,
        "top_k":4,"lr":0.000329,"weight_decay":5.66e-06,
        "tdc_name":"CYP3A4_Veith","task":"classification","metric":"auroc"
    },
    "cyp2c9_veith": {
        "hidden":256,"num_layers":4,"dropout":0.057,"num_experts":4,
        "top_k":4,"lr":0.000520,"weight_decay":1.08e-05,
        "tdc_name":"CYP2C9_Veith","task":"classification","metric":"auroc"
    },
    "cyp2d6_substrate_carbonmangels": {
        "hidden":256,"num_layers":2,"dropout":0.207,"num_experts":8,
        "top_k":2,"lr":0.000996,"weight_decay":6.38e-05,
        "tdc_name":"CYP2D6_Substrate_CarbonMangels",
        "task":"classification","metric":"auroc"
    },
    "cyp3a4_substrate_carbonmangels": {
        "hidden":128,"num_layers":2,"dropout":0.136,"num_experts":4,
        "top_k":2,"lr":0.000998,"weight_decay":6.18e-06,
        "tdc_name":"CYP3A4_Substrate_CarbonMangels",
        "task":"classification","metric":"auroc"
    },
    "cyp2c9_substrate_carbonmangels": {
        "hidden":256,"num_layers":4,"dropout":0.064,"num_experts":16,
        "top_k":3,"lr":0.000270,"weight_decay":3.82e-06,
        "tdc_name":"CYP2C9_Substrate_CarbonMangels",
        "task":"classification","metric":"auroc"
    },
    "herg": {
        "hidden":256,"num_layers":2,"dropout":0.222,"num_experts":8,
        "top_k":4,"lr":0.000156,"weight_decay":2.01e-05,
        "tdc_name":"hERG","task":"classification","metric":"auroc"
    },
    "ames": {
        "hidden":256,"num_layers":4,"dropout":0.191,"num_experts":16,
        "top_k":2,"lr":0.000316,"weight_decay":3.76e-05,
        "tdc_name":"AMES","task":"classification","metric":"auroc"
    },
    "dili": {
        "hidden":128,"num_layers":2,"dropout":0.031,"num_experts":16,
        "top_k":3,"lr":0.000835,"weight_decay":1.01e-05,
        "tdc_name":"DILI","task":"classification","metric":"auroc"
    },
}

DISPLAY_NAMES = {
    "solubility_aqsoldb":"Solubility","caco2_wang":"Caco-2",
    "lipophilicity_astrazeneca":"Lipophilicity","ppbr_az":"PPBR",
    "ld50_zhu":"LD50","vdss_lombardo":"VDss",
    "half_life_obach":"Half-Life","clearance_microsome_az":"CL-Microsome",
    "clearance_hepatocyte_az":"CL-Hepatocyte","hia_hou":"HIA",
    "pgp_broccatelli":"PGP","bioavailability_ma":"Bioavailability",
    "bbb_martins":"BBB","cyp2d6_veith":"CYP2D6 Inh",
    "cyp3a4_veith":"CYP3A4 Inh","cyp2c9_veith":"CYP2C9 Inh",
    "cyp2d6_substrate_carbonmangels":"CYP2D6 Sub",
    "cyp3a4_substrate_carbonmangels":"CYP3A4 Sub",
    "cyp2c9_substrate_carbonmangels":"CYP2C9 Sub",
    "herg":"hERG","ames":"AMES","dili":"DILI",
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

# ══════════════════════════════════════════════════════════════════════════
# MODEL — exact copy from run_expert_specialization.py
# ══════════════════════════════════════════════════════════════════════════

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
            return out, balance_loss, weights, topk_idx
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout,
                 num_experts, top_k, num_outputs=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(
                GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_outputs)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal_loss = self.moe(x)
        return self.head(x).squeeze(-1), bal_loss

    def forward_with_routing(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x, bal_loss, routing_weights, topk_idx = \
            self.moe(x, return_routing=True)
        pred = self.head(x).squeeze(-1)
        dominant = routing_weights.argmax(dim=-1)
        return pred, routing_weights, dominant


# ══════════════════════════════════════════════════════════════════════════
# FEATURIZATION
# ══════════════════════════════════════════════════════════════════════════

def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(), int(atom.GetChiralTag()),
                atom.GetDegree(), atom.GetFormalCharge(),
                atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = [int(bond.GetBondTypeAsDouble()),
                  int(bond.GetStereo()),
                  int(bond.GetIsConjugated())]
            edge_index += [[i,j],[j,i]]
            edge_attr  += [bf, bf]
        if not edge_index:
            edge_index = torch.zeros((2,0), dtype=torch.long)
            edge_attr  = torch.zeros((0,3), dtype=torch.float)
        else:
            edge_index = torch.tensor(
                edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except:
        return None


def build_dataset(smiles_list, labels):
    data_list, valid_smiles = [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y      = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        data_list.append(g)
        valid_smiles.append(smi)
    return data_list, valid_smiles


def load_tdc(tdc_name):
    for Cls in _tdc_classes():
        try:
            data  = Cls(name=tdc_name)
            split = data.get_split(method="scaffold", seed=42)
            return split
        except:
            continue
    return None


def _tdc_classes():
    classes = []
    try:
        from tdc.single_pred import ADME
        classes.append(ADME)
    except: pass
    try:
        from tdc.single_pred import Tox
        classes.append(Tox)
    except: pass
    try:
        from tdc.single_pred import ADMET
        classes.append(ADMET)
    except: pass
    return classes


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, task):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal = model(batch)
        y = batch.y.squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        if task == "regression":
            loss = F.mse_loss(out, y) + 0.01 * bal
        else:
            loss = F.binary_cross_entropy_with_logits(
                out, y) + 0.01 * bal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, task, metric):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    preds  = np.array(preds)
    truths = np.array(truths)
    if metric == "mae":
        return float(np.mean(np.abs(preds - truths)))
    elif metric == "spearman":
        r, _ = spearmanr(preds, truths)
        return float(r)
    elif metric == "auroc":
        try:
            from scipy.special import expit
            return float(roc_auc_score(truths, expit(preds)))
        except:
            return 0.5
    return 0.0


def higher_is_better(metric):
    return metric in ("spearman", "auroc")


def train_model(params, train_data, val_data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    task   = params["task"]
    metric = params["metric"]

    model = MoEGCN(
        in_dim=9,
        hidden=params["hidden"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
        mode="min" if not higher_is_better(metric) else "max")

    drop_last = len(train_data) % 64 == 1
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, drop_last=drop_last)
    val_loader   = DataLoader(val_data, batch_size=256)

    best_val   = float("inf") if not higher_is_better(metric) \
                 else -float("inf")
    best_state = None
    patience   = 0

    for epoch in range(150):
        train_one_epoch(model, train_loader, optimizer, task)
        val_score = evaluate(model, val_loader, task, metric)
        scheduler.step(val_score)

        improved = (val_score < best_val
                    if not higher_is_better(metric)
                    else val_score > best_val)
        if improved:
            best_val   = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
        else:
            patience += 1
            if patience >= 20:
                break

    model.load_state_dict(best_state)
    return model, best_val


# ══════════════════════════════════════════════════════════════════════════
# ROUTING EXTRACTION + SPECIALIZATION
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_routing(model, all_data):
    model.eval()
    loader   = DataLoader(all_data, batch_size=256)
    dominant = []
    weights  = []
    for batch in loader:
        batch = batch.to(DEVICE)
        _, rw, dom = model.forward_with_routing(batch)
        dominant.extend(dom.cpu().numpy().tolist())
        weights.append(rw.cpu().numpy())
    dominant = np.array(dominant)
    weights  = np.vstack(weights)
    return dominant, weights


def compute_eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb   = sum(len(g)*(np.mean(g)-grand)**2 for g in groups)
    sst   = np.sum((all_v - grand)**2)
    return ssb/sst if sst > 0 else 0.0


def compute_routing_score(dominant, smiles_list):
    """Mean eta-squared across all 8 descriptors."""
    expert_ids = np.unique(dominant)
    if len(expert_ids) < 2:
        return 0.0, len(expert_ids)

    desc_vals = defaultdict(list)
    valid_mask = []
    for smi in smiles_list[:len(dominant)]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                valid_mask.append(False)
                for k in DESCRIPTOR_FNS:
                    desc_vals[k].append(np.nan)
                continue
            valid_mask.append(True)
            for name, fn in DESCRIPTOR_FNS.items():
                desc_vals[name].append(float(fn(mol)))
        except:
            valid_mask.append(False)
            for k in DESCRIPTOR_FNS:
                desc_vals[k].append(np.nan)

    valid_mask = np.array(valid_mask)
    dom_valid  = dominant[:len(valid_mask)][valid_mask]
    eta2_vals  = []

    for desc_name, vals in desc_vals.items():
        vals = np.array(vals)[valid_mask]
        not_nan = ~np.isnan(vals)
        if not_nan.sum() < 20:
            continue
        groups = [vals[(dom_valid==e) & not_nan]
                  for e in np.unique(dom_valid)
                  if ((dom_valid==e) & not_nan).sum() >= 3]
        if len(groups) < 2:
            continue
        try:
            eta2_vals.append(compute_eta_squared(groups))
        except:
            continue

    return (float(np.mean(eta2_vals)) if eta2_vals else 0.0,
            len(expert_ids))


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all 22)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=SEEDS)
    parser.add_argument("--skip_existing", action="store_true",
                        default=True)
    args = parser.parse_args()

    target = args.datasets or list(BEST_PARAMS.keys())
    seeds  = args.seeds

    Path("models").mkdir(exist_ok=True)
    results = {}

    print("=" * 65)
    print(f"  Retrain + Extract Routing — {len(target)} datasets "
          f"x {len(seeds)} seeds")
    print("=" * 65)

    for dataset in target:
        print(f"\n{'─'*65}")
        display = DISPLAY_NAMES.get(dataset, dataset)
        print(f"  [{display}]")

        params   = BEST_PARAMS[dataset]
        tdc_name = params["tdc_name"]
        task     = params["task"]
        metric   = params["metric"]

        # Load data
        print(f"  Loading {tdc_name}...")
        split = load_tdc(tdc_name)
        if split is None:
            print(f"  TDC load failed — skipping")
            continue

        train_data, train_smi = build_dataset(
            split["train"]["Drug"], split["train"]["Y"])
        val_data,   val_smi   = build_dataset(
            split["valid"]["Drug"], split["valid"]["Y"])
        test_data,  test_smi  = build_dataset(
            split["test"]["Drug"],  split["test"]["Y"])
        all_data  = train_data + val_data + test_data
        all_smi   = train_smi + val_smi + test_smi
        print(f"  n={len(all_data)} molecules")

        seed_perf    = []
        seed_routing = []
        seed_n_exp   = []

        for seed in seeds:
            ckpt = Path(f"models/moegcn_{dataset}_seed{seed}_v2.pt")
            npy  = Path(f"routing_{dataset}_seed{seed}_v2.npy")

            # Skip if both exist
            if args.skip_existing and ckpt.exists() and npy.exists():
                saved = torch.load(ckpt, map_location="cpu")
                test_score = saved.get("test_score", None)
                routing_score = saved.get("routing_score", None)
                n_exp = saved.get("n_experts", None)
                if all(v is not None
                       for v in [test_score, routing_score, n_exp]):
                    print(f"  Seed {seed}: loaded from cache "
                          f"({metric}={test_score:.4f}, "
                          f"routing={routing_score:.4f})")
                    seed_perf.append(test_score)
                    seed_routing.append(routing_score)
                    seed_n_exp.append(n_exp)
                    continue

            # Train
            print(f"  Seed {seed}: training...", end="", flush=True)
            model, best_val = train_model(
                params, train_data, val_data, seed)

            # Evaluate on test
            test_loader = DataLoader(test_data, batch_size=256)
            test_score  = evaluate(model, test_loader, task, metric)
            print(f" {metric}={test_score:.4f}", end="", flush=True)

            # Extract routing from SAME model
            dominant, weights = extract_routing(model, all_data)
            routing_score, n_exp = compute_routing_score(
                dominant, all_smi)
            print(f" | routing={routing_score:.4f} "
                  f"n_exp={n_exp}")

            # Save checkpoint WITH routing metadata
            torch.save({
                "model_state_dict": model.state_dict(),
                "params":           params,
                "test_score":       test_score,
                "metric":           metric,
                "routing_score":    routing_score,
                "n_experts":        n_exp,
                "seed":             seed,
            }, ckpt)

            # Save routing arrays
            np.save(npy, dominant)
            np.save(f"routing_weights_{dataset}_seed{seed}_v2.npy",
                    weights)

            seed_perf.append(test_score)
            seed_routing.append(routing_score)
            seed_n_exp.append(n_exp)

        if not seed_perf:
            continue

        # Aggregate across seeds
        mean_perf    = float(np.mean(seed_perf))
        std_perf     = float(np.std(seed_perf))
        mean_routing = float(np.mean(seed_routing))
        mean_n_exp   = float(np.mean(seed_n_exp))

        results[dataset] = {
            "display":       display,
            "task":          task,
            "metric":        metric,
            "mean_perf":     round(mean_perf, 4),
            "std_perf":      round(std_perf, 4),
            "seed_perfs":    [round(s, 4) for s in seed_perf],
            "routing_score": round(mean_routing, 4),
            "n_experts":     round(mean_n_exp, 1),
        }

        print(f"  RESULT: {metric}={mean_perf:.4f}±{std_perf:.4f} | "
              f"routing={mean_routing:.4f} | "
              f"n_exp={mean_n_exp:.1f}")

    # ── Save results ───────────────────────────────────────────────────
    with open("results_moegcn_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] results_moegcn_v2.json")

    # ── Quick SSS preview ──────────────────────────────────────────────
    if len(results) >= 5:
        from scipy.stats import spearmanr as sp

        TASK_TYPES = {d: p["task"] for d, p in BEST_PARAMS.items()}
        MOE_GAINS  = {
            "caco2_wang":20.521,"lipophilicity_astrazeneca":8.747,
            "solubility_aqsoldb":8.760,"ppbr_az":1.295,
            "vdss_lombardo":6.310,"half_life_obach":-36.701,
            "clearance_microsome_az":12.900,"clearance_hepatocyte_az":-11.407,
            "hia_hou":1.054,"pgp_broccatelli":0.463,
            "bioavailability_ma":3.705,"bbb_martins":-0.490,
            "cyp2d6_veith":-2.404,"cyp3a4_veith":1.830,"cyp2c9_veith":0.211,
            "cyp2d6_substrate_carbonmangels":-2.999,
            "cyp3a4_substrate_carbonmangels":-5.418,
            "cyp2c9_substrate_carbonmangels":-0.231,
            "herg":-7.368,"ames":-0.357,"dili":-5.203,"ld50_zhu":4.782,
        }

        common = [d for d in results if d in MOE_GAINS]
        gains  = [MOE_GAINS[d] for d in common]

        # SSS = routing_score for regression, 0 for classification
        sss = [results[d]["routing_score"]
               if TASK_TYPES.get(d) == "regression" else 0.0
               for d in common]

        if len(common) >= 5:
            r, p = sp(sss, gains)
            print(f"\n  SSS PREVIEW (n={len(common)}):")
            print(f"  task_x_routing Spearman r={r:.4f}, p={p:.4f}")
            if abs(r) >= 0.65:
                print(f"  Nature Methods level — SSS works!")
            elif abs(r) >= 0.45:
                print(f"  J. Cheminformatics level")
            else:
                print(f"  Weak — may need further analysis")

    print(f"\n{'='*65}")
    print(f"  DONE — {len(results)}/{len(target)} datasets completed")
    print(f"  Next: run sss_v2.py to compute final SSS")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
