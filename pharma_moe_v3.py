"""
pharma_moe_v3.py
=================
Adaptive Pharmacophore-Guided MoE-GCN — Version 3

Three-component loss:
  loss = task_loss + 0.01 * balance_loss + entropy_reg_loss

entropy_reg_loss = -lambda_ent * mean_routing_entropy
  → Penalizes routing collapse during training
  → Keeps experts active across seeds
  → Fixes CL-Microsome seed 2/3 instability

Combined with entropy-conditioned adaptive alpha:
  alpha_i = sigmoid(alpha_base + alpha_scale * H_i)

Run: python pharma_moe_v3.py --datasets clearance_microsome_az ppbr_az hia_hou --seeds 0 1 2 3 4
Run all: python pharma_moe_v3.py --seeds 0 1 2
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
ENTROPY_REG = 0.05  # entropy regularization weight

# ══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE
# ══════════════════════════════════════════════════════════════════════════

def extract_pharmacophore(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros(PHARMA_DIM, dtype=np.float32)
        hbd  = float(rdMolDescriptors.CalcNumHBD(mol))
        hba  = float(rdMolDescriptors.CalcNumHBA(mol))
        ar   = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        rot  = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        hydr = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() in (6,16)
                  and not any(n.GetAtomicNum() in (7,8,9,17,35)
                             for n in a.GetNeighbors()))
        pos  = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum()==7 and a.GetTotalValence()<4)
        neg  = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() in (8,16)
                  and any(b.GetBondTypeAsDouble()==2.0 for b in a.GetBonds()))
        return np.array([hbd,hba,ar,float(hydr),
                        float(pos),float(neg),rot], dtype=np.float32)
    except:
        return np.zeros(PHARMA_DIM, dtype=np.float32)


def build_pharma_stats(smiles_list):
    feats = np.array([extract_pharmacophore(s) for s in smiles_list])
    return feats.mean(0), feats.std(0) + 1e-6


# ══════════════════════════════════════════════════════════════════════════
# ADAPTIVE PHARMA MOE LAYER v3
# ══════════════════════════════════════════════════════════════════════════

class AdaptivePharmaGuidedMoELayerV3(nn.Module):
    """
    Version 3: adaptive alpha + entropy regularization in forward pass.

    Returns routing entropy H so training loop can add entropy reg loss.
    This prevents collapse at initialization level, not just inference.
    """
    def __init__(self, in_dim, out_dim, num_experts, top_k,
                 pharma_dim=PHARMA_DIM):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.graph_gate      = nn.Linear(in_dim, num_experts)
        self.pharma_encoder  = nn.Sequential(
            nn.Linear(pharma_dim, 32), nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        self.alpha_base  = nn.Parameter(torch.tensor(0.0))
        self.alpha_scale = nn.Parameter(torch.tensor(1.0))
        self.max_entropy = float(np.log(num_experts + 1e-10))

    def forward(self, x, pharma_vec, return_routing=False):
        g1       = self.graph_gate(x)                    # [B, E]
        g1_probs = F.softmax(g1, dim=-1)                 # [B, E]

        # Per-molecule entropy
        H      = -(g1_probs * (g1_probs + 1e-10).log()).sum(-1)  # [B]
        H_norm = H / (self.max_entropy + 1e-10)           # [B]

        # Adaptive alpha per molecule
        alpha  = torch.sigmoid(
            self.alpha_base + self.alpha_scale * H_norm
        ).unsqueeze(-1)                                   # [B, 1]

        g2           = self.pharma_encoder(pharma_vec)    # [B, E]
        gate_logits  = alpha * g1 + (1.0 - alpha) * g2   # [B, E]

        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))

        load         = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()

        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out        = (weights.unsqueeze(-1) * expert_out).sum(1)

        # Return H for entropy regularization in training loop
        if return_routing:
            dominant = weights.argmax(-1)
            return out, balance_loss, weights, dominant, \
                   alpha.squeeze(-1), H_norm
        return out, balance_loss, H.mean()   # H.mean() for reg loss


class AdaptivePharmaMoEGCNV3(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout,
                 num_experts, top_k, pharma_dim=PHARMA_DIM):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = AdaptivePharmaGuidedMoELayerV3(
            hidden, hidden, num_experts, top_k, pharma_dim)
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1: x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, data, pharma_vec):
        mol_repr        = self.encode(data)
        x, bal, H_mean  = self.moe(mol_repr, pharma_vec)
        return self.head(x).squeeze(-1), bal, H_mean

    def forward_with_routing(self, data, pharma_vec):
        mol_repr = self.encode(data)
        x, bal, weights, dominant, alpha, H_norm = self.moe(
            mol_repr, pharma_vec, return_routing=True)
        return self.head(x).squeeze(-1), weights, dominant, alpha, H_norm

    def get_alpha_stats(self):
        return {
            "alpha_base":  float(self.moe.alpha_base.item()),
            "alpha_scale": float(self.moe.alpha_scale.item()),
        }


# Standard MoE-GCN
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

    def forward(self, x, return_routing=False):
        gl = self.gate(x)
        tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        load = w.mean(0)
        bal  = self.num_experts * (load*load).sum()
        eo   = torch.stack([e(x) for e in self.experts], dim=1)
        out  = (w.unsqueeze(-1) * eo).sum(1)
        if return_routing:
            return out, bal, w, w.argmax(-1)
        return out, bal


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout,
                 num_experts, top_k):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            if x.size(0) > 1: x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, b)

    def forward(self, data):
        m = self.encode(data)
        x, bal = self.moe(m)
        return self.head(x).squeeze(-1), bal

    def forward_with_routing(self, data):
        m = self.encode(data)
        x, bal, w, dom = self.moe(m, return_routing=True)
        return self.head(x).squeeze(-1), w, dom


# ══════════════════════════════════════════════════════════════════════════
# FEATURIZATION
# ══════════════════════════════════════════════════════════════════════════

def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        af = []
        for atom in mol.GetAtoms():
            af.append([atom.GetAtomicNum(), int(atom.GetChiralTag()),
                       atom.GetDegree(), atom.GetFormalCharge(),
                       atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                       int(atom.GetHybridization()),
                       int(atom.GetIsAromatic()), int(atom.IsInRing())])
        x = torch.tensor(af, dtype=torch.float)
        ei, ea = [], []
        for bond in mol.GetBonds():
            i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = [int(bond.GetBondTypeAsDouble()),
                  int(bond.GetStereo()), int(bond.GetIsConjugated())]
            ei += [[i,j],[j,i]]; ea += [bf,bf]
        if not ei:
            ei = torch.zeros((2,0), dtype=torch.long)
            ea = torch.zeros((0,3), dtype=torch.float)
        else:
            ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
            ea = torch.tensor(ea, dtype=torch.float)
        return Data(x=x, edge_index=ei, edge_attr=ea)
    except: return None


def build_dataset(smiles_list, labels, pharma_mean, pharma_std):
    dl, vs, vp = [], [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None: continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        dl.append(g); vs.append(smi)
        vp.append((extract_pharmacophore(smi) - pharma_mean) / pharma_std)
    return dl, vs, np.array(vp)


def load_tdc(tdc_name):
    for cls in ['ADME','Tox','ADMET']:
        try:
            mod = __import__('tdc.single_pred', fromlist=[cls])
            C   = getattr(mod, cls)
            d   = C(name=tdc_name)
            return d.get_split(method="scaffold", seed=42)
        except: continue
    return None


# ══════════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════════

class PharmaLoader:
    def __init__(self, dl, pa, bs, shuffle=True, drop_last=False):
        self.dl=dl; self.pa=pa; self.bs=bs
        self.shuffle=shuffle; self.drop_last=drop_last

    def __iter__(self):
        idx = np.arange(len(self.dl))
        if self.shuffle: np.random.shuffle(idx)
        for s in range(0, len(idx), self.bs):
            bi = idx[s:s+self.bs]
            if self.drop_last and len(bi) < self.bs: continue
            bp = torch.tensor(self.pa[bi], dtype=torch.float).to(DEVICE)
            bb = Batch.from_data_list([self.dl[i] for i in bi]).to(DEVICE)
            yield bb, bp


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def higher_is_better(metric):
    return metric in ("spearman","auroc")


def metric_fn(preds, truths, metric):
    p,t = np.array(preds), np.array(truths)
    if metric=="mae":      return float(np.mean(np.abs(p-t)))
    if metric=="spearman": r,_=spearmanr(p,t); return float(r)
    if metric=="auroc":
        from scipy.special import expit
        try: return float(roc_auc_score(t, expit(p)))
        except: return 0.5
    return 0.0


@torch.no_grad()
def eval_pharma(model, dl, pa, metric):
    model.eval()
    ps,ts=[],[]
    for b,p in PharmaLoader(dl,pa,256,shuffle=False):
        out,_,_ = model(b,p)
        if out.dim()==0: out=out.unsqueeze(0)
        ps.extend(out.cpu().numpy().tolist())
        ts.extend(b.y.cpu().numpy().flatten().tolist())
    return metric_fn(ps,ts,metric)


@torch.no_grad()
def eval_std(model, dl, metric):
    model.eval()
    ps,ts=[],[]
    for b in DataLoader(dl,batch_size=256):
        b=b.to(DEVICE); out,_=model(b)
        if out.dim()==0: out=out.unsqueeze(0)
        ps.extend(out.cpu().numpy().tolist())
        ts.extend(b.y.cpu().numpy().flatten().tolist())
    return metric_fn(ps,ts,metric)


def train_adaptive_v3(params, train_dl, train_pa,
                      val_dl, val_pa, seed,
                      entropy_reg=ENTROPY_REG):
    torch.manual_seed(seed); np.random.seed(seed)
    metric = params["metric"]
    hib    = higher_is_better(metric)

    model = AdaptivePharmaMoEGCNV3(
        in_dim=9, hidden=params["hidden"],
        num_layers=params["num_layers"], dropout=params["dropout"],
        num_experts=params["num_experts"], top_k=params["top_k"],
    ).to(DEVICE)

    opt  = torch.optim.Adam(
        model.parameters(), lr=params["lr"],
        weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5,
        mode="min" if not hib else "max")

    best_val   = float("inf") if not hib else -float("inf")
    best_state = None
    patience   = 0
    drop_last  = len(train_dl) % 64 == 1

    for epoch in range(150):
        model.train()
        for batch, pharma in PharmaLoader(
                train_dl, train_pa, 64,
                shuffle=True, drop_last=drop_last):
            opt.zero_grad()
            out, bal, H_mean = model(batch, pharma)
            y = batch.y.squeeze()
            if y.dim()==0: y=y.unsqueeze(0)
            if out.dim()==0: out=out.unsqueeze(0)

            # THREE-COMPONENT LOSS
            if metric in ("mae","spearman"):
                task_loss = F.mse_loss(out, y)
            else:
                task_loss = F.binary_cross_entropy_with_logits(out, y)

            # Entropy regularization — REWARD high routing entropy
            # This prevents collapse during training
            entropy_loss = -entropy_reg * H_mean

            loss = task_loss + 0.01 * bal + entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_score = eval_pharma(model, val_dl, val_pa, metric)
        sched.step(val_score)

        improved = (val_score < best_val if not hib
                   else val_score > best_val)
        if improved:
            best_val=val_score; best_state=copy.deepcopy(model.state_dict())
            patience=0
        else:
            patience+=1
            if patience>=20: break

    model.load_state_dict(best_state)
    return model, best_val


def train_standard(params, train_dl, val_dl, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    metric = params["metric"]
    hib    = higher_is_better(metric)

    model = MoEGCN(
        in_dim=9, hidden=params["hidden"],
        num_layers=params["num_layers"], dropout=params["dropout"],
        num_experts=params["num_experts"], top_k=params["top_k"],
    ).to(DEVICE)

    opt   = torch.optim.Adam(
        model.parameters(), lr=params["lr"],
        weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5,
        mode="min" if not hib else "max")

    best_val=float("inf") if not hib else -float("inf")
    best_state=None; patience=0
    drop_last = len(train_dl)%64==1

    for epoch in range(150):
        model.train()
        for b in DataLoader(train_dl,batch_size=64,
                           shuffle=True,drop_last=drop_last):
            b=b.to(DEVICE); opt.zero_grad()
            out,bal=model(b)
            y=b.y.squeeze()
            if y.dim()==0: y=y.unsqueeze(0)
            if out.dim()==0: out=out.unsqueeze(0)
            if metric in ("mae","spearman"):
                loss=F.mse_loss(out,y)+0.01*bal
            else:
                loss=F.binary_cross_entropy_with_logits(out,y)+0.01*bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

        vs=eval_std(model,val_dl,metric); sched.step(vs)
        improved=(vs<best_val if not hib else vs>best_val)
        if improved: best_val=vs; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=20: break

    model.load_state_dict(best_state)
    return model, best_val


# ══════════════════════════════════════════════════════════════════════════
# SPECIALIZATION
# ══════════════════════════════════════════════════════════════════════════

DESCRIPTOR_FNS = {
    "MW":Descriptors.ExactMolWt, "LogP":Descriptors.MolLogP,
    "HBA":rdMolDescriptors.CalcNumHBA, "HBD":rdMolDescriptors.CalcNumHBD,
    "TPSA":Descriptors.TPSA, "ArRings":rdMolDescriptors.CalcNumAromaticRings,
}


def eta_sq(groups):
    av=np.concatenate(groups); gm=np.mean(av)
    ssb=sum(len(g)*(np.mean(g)-gm)**2 for g in groups)
    sst=np.sum((av-gm)**2)
    return float(ssb/sst) if sst>0 else 0.0


def compute_spec(dominant, smiles_list):
    eids=np.unique(dominant)
    if len(eids)<2: return 0.0, len(eids)
    e2=[]
    for fn in DESCRIPTOR_FNS.values():
        vals=[]
        for smi in smiles_list[:len(dominant)]:
            try:
                mol=Chem.MolFromSmiles(str(smi))
                vals.append(float(fn(mol)) if mol else np.nan)
            except: vals.append(np.nan)
        vals=np.array(vals); nn=~np.isnan(vals)
        dv=dominant[:len(vals)][nn]; vv=vals[nn]
        gs=[vv[dv==e] for e in eids if (dv==e).sum()>=3]
        if len(gs)>=2:
            try: e2.append(eta_sq(gs))
            except: pass
    return (float(np.mean(e2)) if e2 else 0.0), len(eids)


# ══════════════════════════════════════════════════════════════════════════
# DATASET CONFIG
# ══════════════════════════════════════════════════════════════════════════

BEST_PARAMS = {
    "solubility_aqsoldb":{"hidden":256,"num_layers":4,"dropout":0.153,"num_experts":8,"top_k":3,"lr":0.000896,"weight_decay":1.99e-05,"tdc_name":"Solubility_AqSolDB","task":"regression","metric":"mae"},
    "caco2_wang":{"hidden":256,"num_layers":3,"dropout":0.031,"num_experts":16,"top_k":4,"lr":0.000599,"weight_decay":2.01e-05,"tdc_name":"Caco2_Wang","task":"regression","metric":"mae"},
    "lipophilicity_astrazeneca":{"hidden":256,"num_layers":4,"dropout":0.038,"num_experts":4,"top_k":1,"lr":0.000972,"weight_decay":7.03e-05,"tdc_name":"Lipophilicity_AstraZeneca","task":"regression","metric":"mae"},
    "ppbr_az":{"hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,"top_k":2,"lr":0.000200,"weight_decay":2.89e-05,"tdc_name":"PPBR_AZ","task":"regression","metric":"mae"},
    "ld50_zhu":{"hidden":256,"num_layers":4,"dropout":0.099,"num_experts":16,"top_k":2,"lr":0.000470,"weight_decay":4.92e-05,"tdc_name":"LD50_Zhu","task":"regression","metric":"mae"},
    "vdss_lombardo":{"hidden":256,"num_layers":3,"dropout":0.222,"num_experts":8,"top_k":2,"lr":0.000905,"weight_decay":2.82e-05,"tdc_name":"VDss_Lombardo","task":"regression","metric":"spearman"},
    "half_life_obach":{"hidden":256,"num_layers":3,"dropout":0.220,"num_experts":16,"top_k":4,"lr":0.000539,"weight_decay":1.83e-05,"tdc_name":"Half_Life_Obach","task":"regression","metric":"spearman"},
    "clearance_microsome_az":{"hidden":256,"num_layers":3,"dropout":0.003,"num_experts":16,"top_k":2,"lr":0.000195,"weight_decay":2.91e-05,"tdc_name":"Clearance_Microsome_AZ","task":"regression","metric":"spearman"},
    "clearance_hepatocyte_az":{"hidden":256,"num_layers":3,"dropout":0.221,"num_experts":16,"top_k":2,"lr":0.000973,"weight_decay":1.05e-06,"tdc_name":"Clearance_Hepatocyte_AZ","task":"regression","metric":"spearman"},
    "hia_hou":{"hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,"top_k":4,"lr":0.000914,"weight_decay":2.15e-05,"tdc_name":"HIA_Hou","task":"classification","metric":"auroc"},
    "bbb_martins":{"hidden":256,"num_layers":4,"dropout":0.202,"num_experts":16,"top_k":4,"lr":0.000527,"weight_decay":1.27e-05,"tdc_name":"BBB_Martins","task":"classification","metric":"auroc"},
    "herg":{"hidden":256,"num_layers":2,"dropout":0.222,"num_experts":8,"top_k":4,"lr":0.000156,"weight_decay":2.01e-05,"tdc_name":"hERG","task":"classification","metric":"auroc"},
    "ames":{"hidden":256,"num_layers":4,"dropout":0.191,"num_experts":16,"top_k":2,"lr":0.000316,"weight_decay":3.76e-05,"tdc_name":"AMES","task":"classification","metric":"auroc"},
    "dili":{"hidden":128,"num_layers":2,"dropout":0.031,"num_experts":16,"top_k":3,"lr":0.000835,"weight_decay":1.01e-05,"tdc_name":"DILI","task":"classification","metric":"auroc"},
}

DISPLAY = {
    "solubility_aqsoldb":"Solubility","caco2_wang":"Caco-2",
    "lipophilicity_astrazeneca":"Lipophilicity","ppbr_az":"PPBR",
    "ld50_zhu":"LD50","vdss_lombardo":"VDss",
    "half_life_obach":"Half-Life","clearance_microsome_az":"CL-Microsome",
    "clearance_hepatocyte_az":"CL-Hepatocyte","hia_hou":"HIA",
    "bbb_martins":"BBB","herg":"hERG","ames":"AMES","dili":"DILI",
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(BEST_PARAMS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--entropy_reg", type=float, default=ENTROPY_REG)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    results = {}

    print("="*70)
    print("  Adaptive Pharmacophore-Guided MoE-GCN v3")
    print(f"  Entropy regularization weight: {args.entropy_reg}")
    print("="*70)

    for dataset in args.datasets:
        if dataset not in BEST_PARAMS: continue
        params  = BEST_PARAMS[dataset]
        metric  = params["metric"]
        display = DISPLAY.get(dataset, dataset)

        print(f"\n{'─'*70}")
        print(f"  [{display}]")

        split = load_tdc(params["tdc_name"])
        if split is None: print("  TDC failed"); continue

        pm, ps = build_pharma_stats(list(split["train"]["Drug"]))
        tr_dl,tr_sm,tr_pa = build_dataset(split["train"]["Drug"],split["train"]["Y"],pm,ps)
        va_dl,va_sm,va_pa = build_dataset(split["valid"]["Drug"],split["valid"]["Y"],pm,ps)
        te_dl,te_sm,te_pa = build_dataset(split["test"]["Drug"],split["test"]["Y"],pm,ps)
        all_dl = tr_dl+va_dl+te_dl
        all_sm = tr_sm+va_sm+te_sm
        all_pa = np.vstack([tr_pa,va_pa,te_pa])
        print(f"  n={len(all_dl)}")

        a_scores,s_scores=[],[]
        a_spec,s_spec=[],[]
        ab_list,as_list,ma_list=[],[],[]

        for seed in args.seeds:
            print(f"\n  Seed {seed}:")

            # Adaptive v3
            print(f"    Adaptive-v3...", end="", flush=True)
            m,_ = train_adaptive_v3(
                params, tr_dl, tr_pa, va_dl, va_pa, seed,
                entropy_reg=args.entropy_reg)
            sc = eval_pharma(m, te_dl, te_pa, metric)
            a_scores.append(sc)

            m.eval()
            dom_a, alp_a = [], []
            with torch.no_grad():
                for b,p in PharmaLoader(all_dl,all_pa,256,shuffle=False):
                    _,_,dom,alpha,_ = m.forward_with_routing(b,p)
                    dom_a.extend(dom.cpu().numpy().tolist())
                    alp_a.extend(alpha.cpu().numpy().tolist())
            sp_a, ne_a = compute_spec(np.array(dom_a), all_sm)
            a_spec.append(sp_a)
            aps = m.get_alpha_stats()
            ab_list.append(aps["alpha_base"])
            as_list.append(aps["alpha_scale"])
            ma_list.append(float(np.mean(alp_a)))
            print(f" {metric}={sc:.4f} eta²={sp_a:.4f} "
                  f"n_exp={ne_a} α_b={aps['alpha_base']:.3f} "
                  f"α_s={aps['alpha_scale']:.3f} "
                  f"mean_α={float(np.mean(alp_a)):.3f}")

            # Standard
            print(f"    Standard-MoE...", end="", flush=True)
            sm,_ = train_standard(params, tr_dl, va_dl, seed)
            ssc  = eval_std(sm, te_dl, metric)
            s_scores.append(ssc)
            sm.eval()
            dom_s=[]
            with torch.no_grad():
                for b in DataLoader(all_dl,batch_size=256):
                    b=b.to(DEVICE); _,_,dom=sm.forward_with_routing(b)
                    dom_s.extend(dom.cpu().numpy().tolist())
            sp_s, ne_s = compute_spec(np.array(dom_s), all_sm)
            s_spec.append(sp_s)
            print(f" {metric}={ssc:.4f} eta²={sp_s:.4f} n_exp={ne_s}")

        hib  = higher_is_better(metric)
        ma   = float(np.mean(a_scores))
        ms   = float(np.mean(s_scores))
        msa  = float(np.mean(a_spec))
        mss  = float(np.mean(s_spec))
        gain = ((ma-ms)/abs(ms)*100) if hib else ((ms-ma)/abs(ms)*100)
        simpr= (msa-mss)/(mss+1e-6)*100

        verdict = ("✓ WINS both" if gain>0 and simpr>0
                  else "~ Spec only" if simpr>0
                  else "~ Perf only" if gain>0
                  else "✗ No advantage")

        print(f"\n  ── [{display}] ──")
        print(f"  Adaptive-v3:  {metric}={ma:.4f} eta²={msa:.4f} "
              f"α_base={np.mean(ab_list):.3f} α_scale={np.mean(as_list):.3f}")
        print(f"  Standard-MoE: {metric}={ms:.4f} eta²={mss:.4f}")
        print(f"  Perf: {gain:+.1f}%  Spec: {simpr:+.1f}%  {verdict}")

        results[dataset] = {
            "display":display,"metric":metric,
            "adaptive_score":round(ma,4),"standard_score":round(ms,4),
            "perf_gain_pct":round(gain,2),
            "adaptive_eta2":round(msa,4),"standard_eta2":round(mss,4),
            "spec_impr_pct":round(simpr,2),
            "mean_alpha":round(float(np.mean(ma_list)),3),
            "alpha_base":round(float(np.mean(ab_list)),3),
            "alpha_scale":round(float(np.mean(as_list)),3),
        }

    with open("pharma_moe_v3_results.json","w") as f:
        json.dump(results,f,indent=2)
    print(f"\n  [SAVED] pharma_moe_v3_results.json")

    print(f"\n{'='*70}")
    print("  SUMMARY — Adaptive Pharmacophore-Guided MoE-GCN v3")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<20} {'Perf':>8} {'Spec':>8} "
          f"{'α_base':>8} {'α_scale':>9} Verdict")
    print("  "+"-"*65)

    wins=0
    for d,v in results.items():
        vd=("✓" if v["perf_gain_pct"]>0 and v["spec_impr_pct"]>0
           else "~" if v["spec_impr_pct"]>0 or v["perf_gain_pct"]>0
           else "✗")
        if vd=="✓": wins+=1
        print(f"  {v['display']:<20} {v['perf_gain_pct']:>+7.1f}% "
              f"{v['spec_impr_pct']:>+7.1f}% "
              f"{v['alpha_base']:>8.3f} {v['alpha_scale']:>9.3f} {vd}")

    gs=[v["perf_gain_pct"] for v in results.values()]
    ss=[v["spec_impr_pct"] for v in results.values()]
    print(f"\n  Mean performance gain:           {np.mean(gs):+.1f}%")
    print(f"  Mean specialization improvement: {np.mean(ss):+.1f}%")
    print(f"  Wins (both metrics):             {wins}/{len(results)}")
    print(f"{'='*70}")


if __name__=="__main__":
    main()
