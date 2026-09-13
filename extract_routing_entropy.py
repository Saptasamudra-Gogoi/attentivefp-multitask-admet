"""
extract_routing_entropy.py
==========================
PURPOSE: Retrain MoE-GCN using best hyperparams from existing results,
         save checkpoints, extract per-molecule routing weights,
         compute routing entropy per dataset per seed.

This is the foundation for:
  - Experiment B1: entropy vs performance correlation
  - Experiment B2: seed invariance test
  - Experiment B3: partial correlation (controlling confounds)

OUTPUT FILES:
  checkpoints/          <- model weights per dataset per seed
  entropy_results/
    routing_weights_{dataset}_{seed}.json   <- per-molecule weights
    entropy_summary.json                    <- dataset-level entropy stats
    molecule_metadata.json                  <- SMILES + descriptors

USAGE:
  python extract_routing_entropy.py
  python extract_routing_entropy.py --datasets esol freesolv  (test subset)
  python extract_routing_entropy.py --skip_existing
"""

import os
import json
import copy
import argparse
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Output directories ───────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
ENTROPY_DIR    = Path("entropy_results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
ENTROPY_DIR.mkdir(exist_ok=True)

# ── Dataset configs (MoleculeNet + TDC) ─────────────────────────────────────
# We pull best_params from your existing result JSONs.
# task: clf or reg
# metric: how to evaluate
# source: moleculenet or tdc
# result_file: where best_params live

DATASET_CONFIGS = {
    # ── MoleculeNet Regression (3) ──
    "ESOL":     {"source": "moleculenet", "task": "reg", "metric": "rmse",
                 "result_file": "results_moegcn_regr.json"},
    "FreeSolv": {"source": "moleculenet", "task": "reg", "metric": "rmse",
                 "result_file": "results_moegcn_regr.json"},
    "Lipo":     {"source": "moleculenet", "task": "reg", "metric": "rmse",
                 "result_file": "results_moegcn_regr.json"},

    # ── TDC Classification (13) ──
    "hia_hou":                        {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "HIA_Hou",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "pgp_broccatelli":                {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "Pgp_Broccatelli",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "bioavailability_ma":             {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "Bioavailability_Ma",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "bbb_martins":                    {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "BBB_Martins",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp2c9_veith":                   {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "CYP2C9_Veith",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp3a4_veith":                   {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "CYP3A4_Veith",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp2d6_veith":                   {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "CYP2D6_Veith",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp2d6_substrate_carbonmangels": {"source": "tdc", "task": "clf",
                                       "metric": "auroc",
                                       "tdc_name": "CYP2D6_Substrate_CarbonMangels",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp3a4_substrate_carbonmangels": {"source": "tdc", "task": "clf",
                                       "metric": "auroc",
                                       "tdc_name": "CYP3A4_Substrate_CarbonMangels",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "cyp2c9_substrate_carbonmangels": {"source": "tdc", "task": "clf",
                                       "metric": "auroc",
                                       "tdc_name": "CYP2C9_Substrate_CarbonMangels",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "herg":                           {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "hERG",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "ames":                           {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "AMES",
                                       "result_file": "results_moegcn_tdc_v2.json"},
    "dili":                           {"source": "tdc", "task": "clf",
                                       "metric": "auroc", "tdc_name": "DILI",
                                       "result_file": "results_moegcn_tdc_v2.json"},

    # ── TDC Regression (9) ──
    "caco2_wang":                {"source": "tdc", "task": "reg",
                                  "metric": "mae", "tdc_name": "Caco2_Wang",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "lipophilicity_astrazeneca": {"source": "tdc", "task": "reg",
                                  "metric": "mae",
                                  "tdc_name": "Lipophilicity_AstraZeneca",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "solubility_aqsoldb":        {"source": "tdc", "task": "reg",
                                  "metric": "mae", "tdc_name": "Solubility_AqSolDB",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "ppbr_az":                   {"source": "tdc", "task": "reg",
                                  "metric": "mae", "tdc_name": "PPBR_AZ",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "vdss_lombardo":             {"source": "tdc", "task": "reg",
                                  "metric": "spearman", "tdc_name": "VDss_Lombardo",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "half_life_obach":           {"source": "tdc", "task": "reg",
                                  "metric": "spearman", "tdc_name": "Half_Life_Obach",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "clearance_microsome_az":    {"source": "tdc", "task": "reg",
                                  "metric": "spearman",
                                  "tdc_name": "Clearance_Microsome_AZ",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "clearance_hepatocyte_az":   {"source": "tdc", "task": "reg",
                                  "metric": "spearman",
                                  "tdc_name": "Clearance_Hepatocyte_AZ",
                                  "result_file": "results_moegcn_tdc_v2.json"},
    "ld50_zhu":                  {"source": "tdc", "task": "reg",
                                  "metric": "mae", "tdc_name": "LD50_Zhu",
                                  "result_file": "results_moegcn_tdc_v2.json"},
}

# ── GCN baseline results for performance gain computation ────────────────────
GCN_RESULTS_FILE = "results_gcn_tdc.json"

# ── Fallback hyperparams if no saved results found ───────────────────────────
DEFAULT_PARAMS = {
    "hidden": 128, "num_layers": 2, "dropout": 0.1,
    "num_experts": 8, "top_k": 2,
    "lr": 5e-4, "weight_decay": 1e-5
}


# ════════════════════════════════════════════════════════════════════════════
# MODEL — same as your run_moegcn_tdc_benchmark.py but routing weights EXPOSED
# ════════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x, return_weights=False):
        gate_logits           = self.gate(x)
        topk_vals, topk_idx   = torch.topk(gate_logits, self.top_k, dim=-1)

        # Full weight vector (zeros for non-topk experts)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1)
        )
        load         = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()

        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out        = (weights.unsqueeze(-1) * expert_out).sum(dim=1)

        if return_weights:
            return out, balance_loss, weights   # weights: [B, num_experts]
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.dropout  = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data, return_routing=False):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)

        if return_routing:
            x, bal_loss, routing_weights = self.moe(x, return_weights=True)
            return self.head(x).squeeze(-1), bal_loss, routing_weights
        else:
            x, bal_loss = self.moe(x)
            return self.head(x).squeeze(-1), bal_loss


# ════════════════════════════════════════════════════════════════════════════
# FEATURIZATION — identical to your run_moegcn_tdc_benchmark.py
# ════════════════════════════════════════════════════════════════════════════

def mol_to_graph(smiles):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(),
                int(atom.GetChiralTag()),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = [
                int(bond.GetBondTypeAsDouble()),
                int(bond.GetStereo()),
                int(bond.GetIsConjugated()),
            ]
            edge_index += [[i, j], [j, i]]
            edge_attr  += [bond_feat, bond_feat]
        if not edge_index:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


def smiles_to_dataset(smiles_list, labels):
    data_list, valid_smiles = [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y     = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi          # store SMILES on graph object for later
        data_list.append(g)
        valid_smiles.append(smi)
    return data_list, valid_smiles


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_moleculenet_dataset(name):
    from torch_geometric.datasets import MoleculeNet
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    dataset   = MoleculeNet(root="./data", name=name)
    smiles_list = dataset.smiles

    # Scaffold split
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol,
                                                       includeChirality=False)
        except Exception:
            sc = str(i)
        scaffolds[sc].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cut = int(n * 0.8)
    val_cut   = int(n * 0.9)

    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if   len(train_idx) < train_cut:          train_idx.extend(s)
        elif len(val_idx)   < (val_cut-train_cut): val_idx.extend(s)
        else:                                       test_idx.extend(s)

    def to_data_list(indices):
        items, smis = [], []
        for i in indices:
            g = mol_to_graph(smiles_list[i])
            if g is None:
                continue
            g.y      = dataset[i].y.float()
            g.smiles = smiles_list[i]
            items.append(g)
            smis.append(smiles_list[i])
        return items, smis

    train_data, train_smiles = to_data_list(train_idx)
    val_data,   val_smiles   = to_data_list(val_idx)
    test_data,  test_smiles  = to_data_list(test_idx)

    all_data   = train_data + val_data + test_data
    all_smiles = train_smiles + val_smiles + test_smiles
    return train_data, val_data, test_data, all_data, all_smiles


def load_tdc_dataset(tdc_name, task):
    from tdc.single_pred import ADME, Tox
    data_obj = None
    for Loader in [ADME, Tox]:
        try:
            data_obj = Loader(name=tdc_name)
            break
        except Exception:
            continue
    if data_obj is None:
        raise ValueError(f"Could not load TDC: {tdc_name}")

    split = data_obj.get_split(method="scaffold", seed=42)
    train_d, train_s = smiles_to_dataset(
        split["train"]["Drug"], split["train"]["Y"])
    val_d,   val_s   = smiles_to_dataset(
        split["valid"]["Drug"], split["valid"]["Y"])
    test_d,  test_s  = smiles_to_dataset(
        split["test"]["Drug"],  split["test"]["Y"])

    all_data   = train_d + val_d + test_d
    all_smiles = train_s + val_s + test_s
    return train_d, val_d, test_d, all_data, all_smiles


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, task):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y    = batch.y.squeeze().float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        if task == "clf":
            loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
        else:
            loss = F.mse_loss(out[mask], y[mask])
        loss = loss + 0.01 * bal_loss
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, task, metric):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    preds  = np.array(preds)
    truths = np.array(truths)
    mask   = ~np.isnan(truths)
    preds, truths = preds[mask], truths[mask]

    if metric == "auroc":
        if len(np.unique(truths)) < 2:
            return 0.5
        return float(roc_auc_score(truths, preds))
    elif metric == "mae":
        return float(np.mean(np.abs(preds - truths)))
    elif metric == "spearman":
        r, _ = spearmanr(preds, truths)
        return float(r) if not np.isnan(r) else 0.0
    else:  # rmse
        return float(np.sqrt(np.mean((preds - truths) ** 2)))


def train_with_checkpoint(train_data, val_data, test_data,
                          params, task, metric,
                          seed, dataset_key, max_epochs=100, patience=15):
    """Train model, save best checkpoint, return (test_score, model)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MoEGCN(
        in_dim=9,
        hidden=params["hidden"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_experts=params["num_experts"],
        top_k=min(params["top_k"], params["num_experts"]),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    drop_last    = len(train_data) % 64 == 1
    train_loader = DataLoader(train_data, batch_size=64,
                              shuffle=True, drop_last=drop_last)
    val_loader   = DataLoader(val_data,   batch_size=256)
    test_loader  = DataLoader(test_data,  batch_size=256)

    higher       = metric in ("auroc", "spearman")
    best_val     = -np.inf if higher else np.inf
    best_state   = None
    patience_cnt = 0

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, optimizer, task)
        val_score = evaluate(model, val_loader, task, metric)
        scheduler.step(-val_score if higher else val_score)

        improved = (val_score > best_val) if higher else (val_score < best_val)
        if improved:
            best_val   = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{dataset_key}_seed{seed}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "params":     params,
        "task":       task,
        "metric":     metric,
        "best_val":   best_val,
    }, ckpt_path)

    test_score = evaluate(model, test_loader, task, metric)
    return test_score, model


# ════════════════════════════════════════════════════════════════════════════
# ROUTING ENTROPY EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_routing_weights(model, all_data, all_smiles):
    """
    Forward pass all molecules through trained model.
    Returns:
        routing_weights: np.array [N, num_experts]
        predictions:     np.array [N]
        truths:          np.array [N]
        smiles_list:     list[str]
    """
    model.eval()
    loader = DataLoader(all_data, batch_size=256, shuffle=False)

    all_weights = []
    all_preds   = []
    all_truths  = []

    for batch in loader:
        batch = batch.to(DEVICE)
        preds, _, weights = model(batch, return_routing=True)
        all_weights.append(weights.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_truths.append(batch.y.cpu().numpy().flatten())

    routing_weights = np.vstack(all_weights)     # [N, num_experts]
    predictions     = np.concatenate(all_preds)  # [N]
    truths          = np.concatenate(all_truths) # [N]

    return routing_weights, predictions, truths, all_smiles


def compute_entropy(routing_weights):
    """
    Shannon entropy of routing distribution per molecule.
    H = -sum(w * log(w))  for non-zero weights
    Returns: np.array [N]
    """
    # Clip to avoid log(0)
    w   = np.clip(routing_weights, 1e-10, 1.0)
    H   = -np.sum(w * np.log(w), axis=1)
    return H


def compute_rdkit_descriptors(smiles_list):
    """
    Compute 8 physicochemical descriptors for each molecule.
    Returns: dict {smiles: {MW, LogP, HBA, HBD, TPSA, RotBonds, Rings, ArRings}}
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        print("  WARNING: RDKit not available for descriptor computation")
        return {}

    desc_dict = {}
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            desc_dict[smi] = {
                "MW":       float(Descriptors.MolWt(mol)),
                "LogP":     float(Descriptors.MolLogP(mol)),
                "HBA":      float(rdMolDescriptors.CalcNumHBA(mol)),
                "HBD":      float(rdMolDescriptors.CalcNumHBD(mol)),
                "TPSA":     float(Descriptors.TPSA(mol)),
                "RotBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                "Rings":    float(rdMolDescriptors.CalcNumRings(mol)),
                "ArRings":  float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            }
        except Exception:
            continue
    return desc_dict


# ════════════════════════════════════════════════════════════════════════════
# LOAD BEST PARAMS FROM EXISTING RESULTS
# ════════════════════════════════════════════════════════════════════════════

def load_best_params(dataset_key, config):
    """Load best hyperparams from your existing result JSON files."""
    result_file = config.get("result_file", "")
    if not result_file or not Path(result_file).exists():
        print(f"  WARNING: {result_file} not found, using defaults")
        return DEFAULT_PARAMS.copy()

    with open(result_file) as f:
        results = json.load(f)

    # MoleculeNet results use dataset name directly
    # TDC results use dataset key
    key = dataset_key if dataset_key in results else config.get("tdc_name", "")

    if key in results and "best_params" in results[key]:
        params = results[key]["best_params"].copy()
        # Ensure all required keys exist
        for k, v in DEFAULT_PARAMS.items():
            if k not in params:
                params[k] = v
        print(f"  Loaded params from {result_file}: {params}")
        return params
    else:
        print(f"  WARNING: {dataset_key} not in {result_file}, using defaults")
        return DEFAULT_PARAMS.copy()


# ════════════════════════════════════════════════════════════════════════════
# PERFORMANCE GAIN COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def load_gcn_results():
    """Load GCN baseline scores for performance gain computation."""
    gcn_scores = {}

    # Try TDC GCN results
    if Path(GCN_RESULTS_FILE).exists():
        with open(GCN_RESULTS_FILE) as f:
            gcn_data = json.load(f)
        for k, v in gcn_data.items():
            if isinstance(v, dict) and "mean" in v:
                gcn_scores[k] = v["mean"]

    # Try MoleculeNet GCN results from baselines files
    baseline_files = [
        "results/baselines_20260522_0241.json",
        "results/baselines_20260521_2140.json",
    ]
    for bf in baseline_files:
        if Path(bf).exists():
            with open(bf) as f:
                baseline_data = json.load(f)
            # Look for GCN entries
            for k, v in baseline_data.items():
                if "GCN" in k or "gcn" in k:
                    if isinstance(v, dict) and "mean" in v:
                        gcn_scores[k] = v["mean"]

    return gcn_scores


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_dataset(dataset_key, config, n_seeds=5, skip_existing=False):
    """
    Full pipeline for one dataset:
    1. Load data
    2. Load best params
    3. Train n_seeds models with checkpoint saving
    4. Extract routing weights per seed
    5. Compute entropy
    6. Save results

    Returns: dict with entropy stats and performance scores
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_key}")
    print(f"{'='*60}")

    # Check if already done
    summary_path = ENTROPY_DIR / f"{dataset_key}_entropy_summary.json"
    if skip_existing and summary_path.exists():
        print(f"  Skipping {dataset_key} (already done)")
        with open(summary_path) as f:
            return json.load(f)

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        if config["source"] == "moleculenet":
            train_d, val_d, test_d, all_d, all_s = \
                load_moleculenet_dataset(dataset_key)
        else:
            train_d, val_d, test_d, all_d, all_s = \
                load_tdc_dataset(config["tdc_name"], config["task"])
    except Exception as e:
        print(f"  ERROR loading {dataset_key}: {e}")
        return None

    print(f"  Data: train={len(train_d)}, val={len(val_d)}, "
          f"test={len(test_d)}, total={len(all_d)}")

    if len(train_d) < 10:
        print(f"  SKIP: too few molecules")
        return None

    # ── Load best params ───────────────────────────────────────────────────
    params = load_best_params(dataset_key, config)
    task   = config["task"]
    metric = config["metric"]

    # ── Train + extract per seed ───────────────────────────────────────────
    per_seed_entropy  = []   # [n_seeds, N] — entropy per molecule per seed
    per_seed_weights  = []   # [n_seeds, N, E] — full routing weights
    per_seed_scores   = []   # test scores
    per_seed_preds    = []   # [n_seeds, N] — predictions

    for seed in range(n_seeds):
        print(f"\n  Seed {seed}/{n_seeds-1}...")

        # Check if checkpoint exists
        ckpt_path = CHECKPOINT_DIR / f"{dataset_key}_seed{seed}.pt"
        weight_path = ENTROPY_DIR / f"routing_weights_{dataset_key}_seed{seed}.npz"

        if ckpt_path.exists() and weight_path.exists():
            print(f"    Checkpoint exists — loading weights directly")
            # Load model from checkpoint
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model = MoEGCN(
                in_dim=9,
                hidden=params["hidden"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                num_experts=params["num_experts"],
                top_k=min(params["top_k"], params["num_experts"]),
            ).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])

            # Evaluate to get test score
            test_loader = DataLoader(test_d, batch_size=256)
            test_score  = evaluate(model, test_loader, task, metric)
        else:
            # Train from scratch
            test_score, model = train_with_checkpoint(
                train_d, val_d, test_d,
                params, task, metric,
                seed, dataset_key
            )

        per_seed_scores.append(test_score)
        print(f"    Test {metric}: {test_score:.4f}")

        # ── Extract routing weights ────────────────────────────────────────
        routing_weights, preds, truths, smiles = \
            extract_routing_weights(model, all_d, all_s)

        # Compute entropy
        entropy = compute_entropy(routing_weights)  # [N]

        per_seed_entropy.append(entropy)
        per_seed_weights.append(routing_weights)
        per_seed_preds.append(preds)

        # Save per-seed routing weights
        np.savez_compressed(
            ENTROPY_DIR / f"routing_weights_{dataset_key}_seed{seed}.npz",
            routing_weights=routing_weights,
            entropy=entropy,
            predictions=preds,
            truths=truths,
        )
        print(f"    Entropy: mean={entropy.mean():.4f}, "
              f"std={entropy.std():.4f}, "
              f"max={entropy.max():.4f}")

    # ── Compute seed-averaged entropy ──────────────────────────────────────
    per_seed_entropy = np.array(per_seed_entropy)   # [n_seeds, N]
    per_seed_weights = np.array(per_seed_weights)   # [n_seeds, N, E]

    mean_entropy    = per_seed_entropy.mean(axis=0)   # [N] — averaged over seeds
    dataset_entropy = float(mean_entropy.mean())       # scalar — dataset-level

    # ── Seed invariance metrics (B2) ───────────────────────────────────────
    # Spearman correlation between entropy rankings across seeds
    from scipy.stats import spearmanr as sp
    seed_corrs = []
    for i in range(n_seeds):
        for j in range(i+1, n_seeds):
            r, _ = sp(per_seed_entropy[i], per_seed_entropy[j])
            seed_corrs.append(float(r))

    mean_seed_corr = float(np.mean(seed_corrs)) if seed_corrs else 0.0
    cv = float(per_seed_entropy.std(axis=0).mean() /
               (per_seed_entropy.mean(axis=0).mean() + 1e-10))

    print(f"\n  ENTROPY SUMMARY for {dataset_key}:")
    print(f"    Dataset-level mean entropy: {dataset_entropy:.4f}")
    print(f"    Seed-pair Spearman corr:    {mean_seed_corr:.4f}  "
          f"({'INVARIANT ✓' if mean_seed_corr > 0.7 else 'variable ✗'})")
    print(f"    Coefficient of variation:   {cv:.4f}  "
          f"({'stable ✓' if cv < 0.1 else 'unstable ✗'})")
    print(f"    Test scores: {[f'{s:.4f}' for s in per_seed_scores]}")
    print(f"    Mean ± std:  {np.mean(per_seed_scores):.4f} "
          f"± {np.std(per_seed_scores):.4f}")

    # ── Identify ambiguous molecules (top 10% entropy) ─────────────────────
    threshold_90 = np.percentile(mean_entropy, 90)
    ambiguous_mask = mean_entropy >= threshold_90
    ambiguous_smiles = [s for s, m in zip(all_s, ambiguous_mask) if m]

    print(f"    Ambiguous molecules (top 10%): {ambiguous_mask.sum()} / {len(all_s)}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "dataset":            dataset_key,
        "n_molecules":        len(all_d),
        "n_seeds":            n_seeds,
        "task":               task,
        "metric":             metric,
        "dataset_entropy":    dataset_entropy,
        "entropy_mean":       float(mean_entropy.mean()),
        "entropy_std":        float(mean_entropy.std()),
        "entropy_median":     float(np.median(mean_entropy)),
        "entropy_p90":        float(threshold_90),
        "seed_corr_mean":     mean_seed_corr,
        "seed_corr_values":   seed_corrs,
        "seed_invariant":     mean_seed_corr > 0.7,
        "cv":                 cv,
        "test_scores":        per_seed_scores,
        "test_mean":          float(np.mean(per_seed_scores)),
        "test_std":           float(np.std(per_seed_scores)),
        "n_ambiguous":        int(ambiguous_mask.sum()),
        "ambiguous_smiles":   ambiguous_smiles[:50],  # save first 50
        "params":             params,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save mean routing weights across seeds ─────────────────────────────
    np.savez_compressed(
        ENTROPY_DIR / f"routing_weights_{dataset_key}_mean.npz",
        routing_weights_mean=per_seed_weights.mean(axis=0),
        entropy_mean=mean_entropy,
        smiles=np.array(all_s, dtype=object),
        truths=truths,
    )

    return summary


# ════════════════════════════════════════════════════════════════════════════
# SAVE GLOBAL ENTROPY SUMMARY (feeds directly into B1 analysis tomorrow)
# ════════════════════════════════════════════════════════════════════════════

def compile_global_summary(all_summaries, gcn_scores):
    """
    Compile all dataset summaries into one file for B1 correlation analysis.
    """
    global_summary = {}

    for key, summary in all_summaries.items():
        if summary is None:
            continue

        moe_score = summary["test_mean"]
        metric    = summary["metric"]

        # Get GCN baseline score
        gcn_score = gcn_scores.get(key)

        # Compute performance gain
        # For error metrics (mae, rmse): gain = (gcn - moe) / gcn  (positive = MoE better)
        # For score metrics (auroc, spearman): gain = (moe - gcn) / gcn (positive = MoE better)
        if gcn_score is not None and gcn_score != 0:
            if metric in ("auroc", "spearman"):
                perf_gain = (moe_score - gcn_score) / abs(gcn_score)
            else:
                perf_gain = (gcn_score - moe_score) / abs(gcn_score)
        else:
            perf_gain = None

        global_summary[key] = {
            "dataset_entropy":  summary["dataset_entropy"],
            "entropy_mean":     summary["entropy_mean"],
            "entropy_std":      summary["entropy_std"],
            "seed_invariant":   summary["seed_invariant"],
            "seed_corr":        summary["seed_corr_mean"],
            "cv":               summary["cv"],
            "moe_score":        moe_score,
            "gcn_score":        gcn_score,
            "perf_gain":        perf_gain,
            "metric":           metric,
            "task":             summary["task"],
            "n_molecules":      summary["n_molecules"],
            "n_ambiguous":      summary["n_ambiguous"],
            "test_scores":      summary["test_scores"],
        }

    # Save
    with open(ENTROPY_DIR / "global_entropy_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  GLOBAL ENTROPY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<35} {'Entropy':>8} {'SeedCorr':>9} "
          f"{'Invariant':>10} {'PerfGain':>9}")
    print("-" * 75)
    for key, v in global_summary.items():
        inv = "✓" if v["seed_invariant"] else "✗"
        pg  = f"{v['perf_gain']:.3f}" if v["perf_gain"] is not None else "N/A"
        print(f"  {key:<33} {v['dataset_entropy']:>8.4f} "
              f"{v['seed_corr']:>9.4f} {inv:>10} {pg:>9}")

    print(f"\n  Saved → {ENTROPY_DIR}/global_entropy_summary.json")
    return global_summary


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",      nargs="+", default=None,
                        help="Specific datasets to process (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip datasets with existing summary files")
    parser.add_argument("--seeds",         type=int, default=5,
                        help="Number of seeds per dataset (default=5)")
    parser.add_argument("--test",          action="store_true",
                        help="Quick test: run 3 small datasets with 2 seeds")
    args = parser.parse_args()

    # Test mode: just run ESOL, HIA, Half-life to verify pipeline
    if args.test:
        datasets_to_run = ["ESOL", "hia_hou", "half_life_obach"]
        n_seeds = 2
        print("TEST MODE: running 3 datasets with 2 seeds")
    else:
        datasets_to_run = args.datasets or list(DATASET_CONFIGS.keys())
        n_seeds = args.seeds

    print(f"\nRunning {len(datasets_to_run)} datasets × {n_seeds} seeds")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Entropy dir:    {ENTROPY_DIR}")

    # Load GCN baseline scores
    gcn_scores = load_gcn_results()
    print(f"Loaded GCN scores for {len(gcn_scores)} datasets")

    # Process each dataset
    all_summaries = {}
    for key in datasets_to_run:
        if key not in DATASET_CONFIGS:
            print(f"WARNING: {key} not in DATASET_CONFIGS, skipping")
            continue
        config  = DATASET_CONFIGS[key]
        summary = process_dataset(key, config, n_seeds=n_seeds,
                                  skip_existing=args.skip_existing)
        all_summaries[key] = summary

    # Compile global summary for B1 analysis
    compile_global_summary(all_summaries, gcn_scores)

    print(f"\n{'='*60}")
    print("  DONE. Next step:")
    print("  python analyze_entropy_correlation.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
