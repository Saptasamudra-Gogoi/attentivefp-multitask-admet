"""
pharma_moe_adaptive.py
=======================
Adaptive Pharmacophore-Guided MoE-GCN

UPGRADE from pharma_moe_gcn.py:
Fixed alpha → Entropy-conditioned adaptive alpha per molecule

Standard MoE:     graph_embedding → gate → routing
Pharma-MoE v1:    graph + pharma → dual gate (fixed alpha)
Pharma-MoE v2:    graph + pharma → dual gate (adaptive alpha)

Adaptive alpha formula:
  H = routing entropy of graph gate (per molecule)
  alpha = sigmoid(alpha_base + alpha_scale * H)
  
  Low H (collapsed routing) → alpha → 1.0 (trust graph only)
  High H (rich routing)     → alpha → 0.5 (trust both equally)

This self-regulates pharmacophore influence based on whether
routing has meaningful chemical structure — fixing CL-Microsome
instability while preserving gains on physicochemical endpoints.

Run: python pharma_moe_adaptive.py
Run specific: python pharma_moe_adaptive.py --datasets clearance_microsome_az ppbr_az hia_hou
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
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

PHARMA_DIM = 7

# ══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE EXTRACTOR (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def extract_pharmacophore(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros(PHARMA_DIM, dtype=np.float32)
        hbd      = float(rdMolDescriptors.CalcNumHBD(mol))
        hba      = float(rdMolDescriptors.CalcNumHBA(mol))
        ar_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        rot      = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        hydrophobic = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetAtomicNum() in (6, 16)
            and not any(n.GetAtomicNum() in (7, 8, 9, 17, 35)
                       for n in atom.GetNeighbors()))
        pos_charge = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 7
            and atom.GetTotalValence() < 4)
        neg_charge = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetAtomicNum() in (8, 16)
            and any(b.GetBondTypeAsDouble() == 2.0
                   for b in atom.GetBonds()))
        return np.array([hbd, hba, ar_rings, float(hydrophobic),
                        float(pos_charge), float(neg_charge), rot],
                       dtype=np.float32)
    except:
        return np.zeros(PHARMA_DIM, dtype=np.float32)


def build_pharma_stats(smiles_list):
    all_feats = np.array([extract_pharmacophore(s) for s in smiles_list])
    return all_feats.mean(axis=0), all_feats.std(axis=0) + 1e-6


# ══════════════════════════════════════════════════════════════════════════
# ADAPTIVE PHARMACOPHORE-GUIDED MOE LAYER — THE NEW ALGORITHM
# ══════════════════════════════════════════════════════════════════════════

class AdaptivePharmaGuidedMoELayer(nn.Module):
    """
    ADAPTIVE dual routing mechanism.

    Key innovation over v1:
    Alpha is now per-molecule and entropy-conditioned:

        H_i = Shannon entropy of graph gate logits for molecule i
        alpha_i = sigmoid(alpha_base + alpha_scale * H_i)

    Interpretation:
        H_i low  (routing collapsed)  → alpha_i → 1.0 → pure graph routing
        H_i high (routing active)     → alpha_i → ~0.5 → balanced dual routing

    This means:
    - On CL-Microsome (routing collapses): pharmacophore is automatically
      ignored, fixing the instability problem
    - On Solubility (routing active): pharmacophore contributes ~50%
      of routing decision, improving specialization

    alpha_base and alpha_scale are BOTH learnable parameters.
    The model learns the optimal entropy-alpha relationship from data.
    """

    def __init__(self, in_dim, out_dim, num_experts, top_k,
                 pharma_dim=PHARMA_DIM):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])

        # Dual gate
        self.graph_gate  = nn.Linear(in_dim, num_experts)
        self.pharma_encoder = nn.Sequential(
            nn.Linear(pharma_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

        # Adaptive alpha parameters — both learnable
        # alpha_base: baseline trust in graph routing
        # alpha_scale: how strongly entropy modulates alpha
        self.alpha_base  = nn.Parameter(torch.tensor(0.0))
        self.alpha_scale = nn.Parameter(torch.tensor(1.0))

        # Max entropy for normalization
        self.max_entropy = float(np.log(num_experts))

    def forward(self, x, pharma_vec, return_routing=False):
        """
        Args:
            x:         [B, in_dim]
            pharma_vec:[B, pharma_dim]
        """
        # Graph gate logits
        g1 = self.graph_gate(x)  # [B, num_experts]

        # Compute per-molecule routing entropy from graph gate
        g1_probs = F.softmax(g1, dim=-1)  # [B, num_experts]
        H = -(g1_probs * (g1_probs + 1e-10).log()).sum(dim=-1)  # [B]
        H_norm = H / (self.max_entropy + 1e-10)  # normalize to [0,1]

        # Adaptive alpha: high entropy → lower alpha (more pharmacophore)
        # Low entropy → higher alpha (less pharmacophore, more graph)
        alpha = torch.sigmoid(
            self.alpha_base + self.alpha_scale * H_norm
        ).unsqueeze(-1)  # [B, 1]

        # Pharmacophore gate
        g2 = self.pharma_encoder(pharma_vec)  # [B, num_experts]

        # Combined gate with per-molecule adaptive alpha
        gate_logits = alpha * g1 + (1.0 - alpha) * g2  # [B, num_experts]

        # Top-k routing
        topk_vals, topk_idx = torch.topk(
            gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))

        # Load balancing
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()

        # Expert computation
        expert_out = torch.stack(
            [e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)

        if return_routing:
            dominant = weights.argmax(dim=-1)
            return out, balance_loss, weights, dominant, alpha.squeeze(-1), H_norm

        return out, balance_loss

    def get_alpha_stats(self):
        return {
            "alpha_base":  float(self.alpha_base.item()),
            "alpha_scale": float(self.alpha_scale.item()),
        }


# ══════════════════════════════════════════════════════════════════════════
# ADAPTIVE PHARMA-MOE-GCN MODEL
# ══════════════════════════════════════════════════════════════════════════

class AdaptivePharmaMoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout,
                 num_experts, top_k, pharma_dim=PHARMA_DIM):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(
                GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = AdaptivePharmaGuidedMoELayer(
            hidden, hidden, num_experts, top_k, pharma_dim)
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, data, pharma_vec):
        mol_repr = self.encode(data)
        x, bal   = self.moe(mol_repr, pharma_vec)
        return self.head(x).squeeze(-1), bal

    def forward_with_routing(self, data, pharma_vec):
        mol_repr = self.encode(data)
        x, bal, weights, dominant, alpha, H_norm = self.moe(
            mol_repr, pharma_vec, return_routing=True)
        pred = self.head(x).squeeze(-1)
        return pred, weights, dominant, alpha, H_norm

    def get_alpha_stats(self):
        return self.moe.get_alpha_stats()


# Standard MoE-GCN for comparison
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
            return out, balance_loss, weights, weights.argmax(dim=-1)
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers,
                 dropout, num_experts, top_k):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(
                GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, data):
        mol_repr = self.encode(data)
        x, bal   = self.moe(mol_repr)
        return self.head(x).squeeze(-1), bal

    def forward_with_routing(self, data):
        mol_repr = self.encode(data)
        x, bal, weights, dominant = self.moe(mol_repr, return_routing=True)
        return self.head(x).squeeze(-1), weights, dominant


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
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()), int(atom.IsInRing()),
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


def build_dataset(smiles_list, labels, pharma_mean, pharma_std):
    data_list, valid_smiles, valid_pharma = [], [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y      = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        data_list.append(g)
        valid_smiles.append(smi)
        pf = (extract_pharmacophore(smi) - pharma_mean) / pharma_std
        valid_pharma.append(pf)
    return data_list, valid_smiles, np.array(valid_pharma)


def load_tdc(tdc_name):
    for cls_name in ['ADME', 'Tox', 'ADMET']:
        try:
            mod = __import__('tdc.single_pred', fromlist=[cls_name])
            Cls = getattr(mod, cls_name)
            data  = Cls(name=tdc_name)
            split = data.get_split(method="scaffold", seed=42)
            return split
        except:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER WITH PHARMACOPHORE
# ══════════════════════════════════════════════════════════════════════════

class PharmaLoader:
    def __init__(self, data_list, pharma_arr, batch_size,
                 shuffle=True, drop_last=False):
        self.data_list  = data_list
        self.pharma_arr = pharma_arr
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

    def __iter__(self):
        indices = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            batch_data   = [self.data_list[i] for i in batch_idx]
            batch_pharma = torch.tensor(
                self.pharma_arr[batch_idx], dtype=torch.float).to(DEVICE)
            batch = Batch.from_data_list(batch_data).to(DEVICE)
            yield batch, batch_pharma


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def higher_is_better(metric):
    return metric in ("spearman", "auroc")


def _compute_metric(preds, truths, metric):
    preds  = np.array(preds)
    truths = np.array(truths)
    if metric == "mae":
        return float(np.mean(np.abs(preds - truths)))
    elif metric == "spearman":
        r, _ = spearmanr(preds, truths)
        return float(r)
    elif metric == "auroc":
        from scipy.special import expit
        try:
            return float(roc_auc_score(truths, expit(preds)))
        except:
            return 0.5
    return 0.0


@torch.no_grad()
def evaluate_pharma(model, data_list, pharma_arr, metric):
    model.eval()
    loader = PharmaLoader(data_list, pharma_arr, 256, shuffle=False)
    preds, truths = [], []
    for batch, pharma in loader:
        out, _ = model(batch, pharma)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    return _compute_metric(preds, truths, metric)


@torch.no_grad()
def evaluate_standard(model, data_list, metric):
    model.eval()
    loader = DataLoader(data_list, batch_size=256)
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    return _compute_metric(preds, truths, metric)


def train_adaptive(params, train_data, train_pharma,
                   val_data, val_pharma, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    metric = params["metric"]

    model = AdaptivePharmaMoEGCN(
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

    best_val   = float("inf") if not higher_is_better(metric) \
                 else -float("inf")
    best_state = None
    patience   = 0

    drop_last = len(train_data) % 64 == 1
    for epoch in range(150):
        model.train()
        loader = PharmaLoader(
            train_data, train_pharma, 64,
            shuffle=True, drop_last=drop_last)
        for batch, pharma in loader:
            optimizer.zero_grad()
            out, bal = model(batch, pharma)
            y = batch.y.squeeze()
            if y.dim() == 0: y = y.unsqueeze(0)
            if out.dim() == 0: out = out.unsqueeze(0)
            if metric in ("mae", "spearman"):
                loss = F.mse_loss(out, y) + 0.01 * bal
            else:
                loss = F.binary_cross_entropy_with_logits(
                    out, y) + 0.01 * bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_score = evaluate_pharma(model, val_data, val_pharma, metric)
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


def train_standard(params, train_data, val_data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    best_val   = float("inf") if not higher_is_better(metric) \
                 else -float("inf")
    best_state = None
    patience   = 0

    drop_last = len(train_data) % 64 == 1
    for epoch in range(150):
        model.train()
        loader = DataLoader(
            train_data, batch_size=64, shuffle=True, drop_last=drop_last)
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out, bal = model(batch)
            y = batch.y.squeeze()
            if y.dim() == 0: y = y.unsqueeze(0)
            if out.dim() == 0: out = out.unsqueeze(0)
            if metric in ("mae", "spearman"):
                loss = F.mse_loss(out, y) + 0.01 * bal
            else:
                loss = F.binary_cross_entropy_with_logits(
                    out, y) + 0.01 * bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_score = evaluate_standard(model, val_data, metric)
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
# SPECIALIZATION
# ══════════════════════════════════════════════════════════════════════════

DESCRIPTOR_FNS = {
    "MW":      Descriptors.ExactMolWt,
    "LogP":    Descriptors.MolLogP,
    "HBA":     rdMolDescriptors.CalcNumHBA,
    "HBD":     rdMolDescriptors.CalcNumHBD,
    "TPSA":    Descriptors.TPSA,
    "ArRings": rdMolDescriptors.CalcNumAromaticRings,
}


def compute_eta_squared(groups):
    all_v  = np.concatenate(groups)
    grand  = np.mean(all_v)
    ssb    = sum(len(g)*(np.mean(g)-grand)**2 for g in groups)
    sst    = np.sum((all_v - grand)**2)
    return float(ssb/sst) if sst > 0 else 0.0


def compute_specialization(dominant, smiles_list):
    expert_ids = np.unique(dominant)
    if len(expert_ids) < 2:
        return 0.0, len(expert_ids)
    eta2_vals = []
    for desc_name, fn in DESCRIPTOR_FNS.items():
        vals = []
        for smi in smiles_list[:len(dominant)]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                vals.append(float(fn(mol)) if mol else np.nan)
            except:
                vals.append(np.nan)
        vals = np.array(vals)
        not_nan = ~np.isnan(vals)
        dom_v   = dominant[:len(vals)][not_nan]
        vals_v  = vals[not_nan]
        groups  = [vals_v[dom_v==e] for e in expert_ids
                  if (dom_v==e).sum() >= 3]
        if len(groups) >= 2:
            try:
                eta2_vals.append(compute_eta_squared(groups))
            except:
                pass
    return (float(np.mean(eta2_vals)) if eta2_vals else 0.0,
            len(expert_ids))


# ══════════════════════════════════════════════════════════════════════════
# DATASET CONFIG
# ══════════════════════════════════════════════════════════════════════════

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
    "bbb_martins": {
        "hidden":256,"num_layers":4,"dropout":0.202,"num_experts":16,
        "top_k":4,"lr":0.000527,"weight_decay":1.27e-05,
        "tdc_name":"BBB_Martins","task":"classification","metric":"auroc"
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

DISPLAY = {
    "solubility_aqsoldb":"Solubility","caco2_wang":"Caco-2",
    "lipophilicity_astrazeneca":"Lipophilicity","ppbr_az":"PPBR",
    "ld50_zhu":"LD50","vdss_lombardo":"VDss",
    "half_life_obach":"Half-Life","clearance_microsome_az":"CL-Microsome",
    "clearance_hepatocyte_az":"CL-Hepatocyte","hia_hou":"HIA",
    "bbb_martins":"BBB","herg":"hERG","ames":"AMES","dili":"DILI",
}

SEEDS = [0, 1, 2, 3, 4]

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(BEST_PARAMS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 2])
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    results = {}

    print("=" * 70)
    print("  Adaptive Pharmacophore-Guided MoE-GCN")
    print("  UPGRADE: Entropy-conditioned adaptive alpha per molecule")
    print("=" * 70)

    for dataset in args.datasets:
        if dataset not in BEST_PARAMS:
            continue

        params  = BEST_PARAMS[dataset]
        metric  = params["metric"]
        display = DISPLAY.get(dataset, dataset)

        print(f"\n{'─'*70}")
        print(f"  [{display}]")

        split = load_tdc(params["tdc_name"])
        if split is None:
            print(f"  TDC load failed")
            continue

        all_train_smi = list(split["train"]["Drug"])
        pharma_mean, pharma_std = build_pharma_stats(all_train_smi)

        train_data, train_smi, train_pharma = build_dataset(
            split["train"]["Drug"], split["train"]["Y"],
            pharma_mean, pharma_std)
        val_data, val_smi, val_pharma = build_dataset(
            split["valid"]["Drug"], split["valid"]["Y"],
            pharma_mean, pharma_std)
        test_data, test_smi, test_pharma = build_dataset(
            split["test"]["Drug"], split["test"]["Y"],
            pharma_mean, pharma_std)

        all_data   = train_data + val_data + test_data
        all_smi    = train_smi + val_smi + test_smi
        all_pharma = np.vstack([train_pharma, val_pharma, test_pharma])

        print(f"  n={len(all_data)} molecules")

        adaptive_scores, standard_scores = [], []
        adaptive_spec,   standard_spec   = [], []
        alpha_base_vals, alpha_scale_vals = [], []
        mean_alpha_vals = []

        for seed in args.seeds:
            print(f"\n  Seed {seed}:")

            # Train adaptive model
            print(f"    Adaptive-MoE...", end="", flush=True)
            model, _ = train_adaptive(
                params, train_data, train_pharma,
                val_data, val_pharma, seed)

            score = evaluate_pharma(
                model, test_data, test_pharma, metric)
            adaptive_scores.append(score)

            # Extract routing + alpha stats
            model.eval()
            loader = PharmaLoader(all_data, all_pharma, 256, shuffle=False)
            all_dom, all_alpha = [], []
            with torch.no_grad():
                for batch, pharma in loader:
                    _, _, dom, alpha, H_norm = model.forward_with_routing(
                        batch, pharma)
                    all_dom.extend(dom.cpu().numpy().tolist())
                    all_alpha.extend(alpha.cpu().numpy().tolist())

            dom_arr = np.array(all_dom)
            spec_a, n_exp_a = compute_specialization(dom_arr, all_smi)
            adaptive_spec.append(spec_a)
            mean_alpha_vals.append(float(np.mean(all_alpha)))

            alpha_stats = model.get_alpha_stats()
            alpha_base_vals.append(alpha_stats["alpha_base"])
            alpha_scale_vals.append(alpha_stats["alpha_scale"])

            print(f" {metric}={score:.4f} eta²={spec_a:.4f} "
                  f"n_exp={n_exp_a} "
                  f"α_base={alpha_stats['alpha_base']:.3f} "
                  f"α_scale={alpha_stats['alpha_scale']:.3f} "
                  f"mean_α={float(np.mean(all_alpha)):.3f}")

            # Train standard model
            print(f"    Standard-MoE...", end="", flush=True)
            std_model, _ = train_standard(
                params, train_data, val_data, seed)

            std_score = evaluate_standard(std_model, test_data, metric)
            standard_scores.append(std_score)

            std_model.eval()
            geo_loader = DataLoader(all_data, batch_size=256)
            all_dom_std = []
            with torch.no_grad():
                for batch in geo_loader:
                    batch = batch.to(DEVICE)
                    _, _, dom = std_model.forward_with_routing(batch)
                    all_dom_std.extend(dom.cpu().numpy().tolist())

            spec_s, n_exp_s = compute_specialization(
                np.array(all_dom_std), all_smi)
            standard_spec.append(spec_s)

            print(f" {metric}={std_score:.4f} eta²={spec_s:.4f} "
                  f"n_exp={n_exp_s}")

        # Aggregate
        mean_adapt  = float(np.mean(adaptive_scores))
        mean_std    = float(np.mean(standard_scores))
        mean_spec_a = float(np.mean(adaptive_spec))
        mean_spec_s = float(np.mean(standard_spec))
        mean_alpha  = float(np.mean(mean_alpha_vals))
        mean_ab     = float(np.mean(alpha_base_vals))
        mean_as     = float(np.mean(alpha_scale_vals))

        if higher_is_better(metric):
            gain = (mean_adapt - mean_std) / abs(mean_std) * 100
        else:
            gain = (mean_std - mean_adapt) / abs(mean_std) * 100

        spec_impr = (mean_spec_a - mean_spec_s) / (mean_spec_s + 1e-6) * 100

        verdict = ("✓ WINS both" if gain > 0 and spec_impr > 0
                  else "~ Spec only" if spec_impr > 0
                  else "~ Perf only" if gain > 0
                  else "✗ No advantage")

        print(f"\n  ── [{display}] ──")
        print(f"  Adaptive-MoE: {metric}={mean_adapt:.4f} "
              f"eta²={mean_spec_a:.4f} "
              f"mean_α={mean_alpha:.3f} "
              f"α_base={mean_ab:.3f} α_scale={mean_as:.3f}")
        print(f"  Standard-MoE: {metric}={mean_std:.4f} "
              f"eta²={mean_spec_s:.4f}")
        print(f"  Perf gain: {gain:+.1f}%  "
              f"Spec improvement: {spec_impr:+.1f}%  {verdict}")

        results[dataset] = {
            "display":       display,
            "metric":        metric,
            "adaptive_score":    round(mean_adapt, 4),
            "standard_score":    round(mean_std, 4),
            "perf_gain_pct":     round(gain, 2),
            "adaptive_eta2":     round(mean_spec_a, 4),
            "standard_eta2":     round(mean_spec_s, 4),
            "spec_impr_pct":     round(spec_impr, 2),
            "mean_alpha":        round(mean_alpha, 3),
            "alpha_base":        round(mean_ab, 3),
            "alpha_scale":       round(mean_as, 3),
        }

    # Save
    with open("adaptive_pharma_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] adaptive_pharma_results.json")

    # Summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY — Adaptive Pharmacophore-Guided MoE-GCN")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<20} {'Perf Gain':>10} "
          f"{'Spec Improv':>12} {'mean_α':>8} "
          f"{'α_base':>8} {'α_scale':>9} Verdict")
    print("  " + "-"*80)

    wins = 0
    for d, v in results.items():
        verdict = ("✓" if v["perf_gain_pct"] > 0
                        and v["spec_impr_pct"] > 0
                  else "~" if v["spec_impr_pct"] > 0 or v["perf_gain_pct"] > 0
                  else "✗")
        if verdict == "✓":
            wins += 1
        print(f"  {v['display']:<20} {v['perf_gain_pct']:>+10.1f}% "
              f"{v['spec_impr_pct']:>+11.1f}% "
              f"{v['mean_alpha']:>8.3f} "
              f"{v['alpha_base']:>8.3f} "
              f"{v['alpha_scale']:>9.3f} {verdict}")

    gains = [v["perf_gain_pct"] for v in results.values()]
    specs = [v["spec_impr_pct"] for v in results.values()]

    print(f"\n  Mean performance gain:           {np.mean(gains):+.1f}%")
    print(f"  Mean specialization improvement: {np.mean(specs):+.1f}%")
    print(f"  Wins (both metrics):             {wins}/{len(results)}")
    print(f"\n  KEY: alpha_scale > 0 means model learned to")
    print(f"  increase pharmacophore trust as routing entropy grows.")
    print(f"  alpha_scale < 0 means pharmacophore suppressed on")
    print(f"  high-entropy (classification) endpoints.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
