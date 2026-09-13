"""
extract_all_routing.py
=======================
Extracts routing assignments for ALL 22 TDC datasets using your
existing MoEGCN architecture and saved checkpoints.

Uses your exact model class from run_expert_specialization.py.
Retrains each dataset using best params from results_moegcn_tdc_v2.json
if checkpoint not found, otherwise loads existing checkpoint.

Place in D:\molprop_project\ and run:
    python extract_all_routing.py

    # Run specific datasets only:
    python extract_all_routing.py --datasets hia_hou pgp_broccatelli bbb_martins

Output:
    routing_{dataset_name}.npy     for each dataset
    Then re-run specialization_from_json.py to get exact SSS
"""

import json
import copy
import argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# YOUR EXACT MODEL — copied from run_expert_specialization.py
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
            1, topk_idx, F.softmax(topk_vals, dim=-1)
        )
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        if return_routing:
            return out, balance_loss, weights, topk_idx
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

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
        x, bal_loss, routing_weights, topk_idx = self.moe(x, return_routing=True)
        pred = self.head(x).squeeze(-1)
        dominant_expert = routing_weights.argmax(dim=-1)
        return pred, routing_weights, dominant_expert


# ══════════════════════════════════════════════════════════════════════════
# BEST PARAMS — from results_moegcn_tdc_v2.json
# ══════════════════════════════════════════════════════════════════════════

BEST_PARAMS = {
    "hia_hou": {
        "hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,
        "top_k":4,"lr":0.000914,"weight_decay":2.15e-05,
        "tdc_name":"HIA_Hou","metric":"AUROC","task":"clf"
    },
    "pgp_broccatelli": {
        "hidden":256,"num_layers":4,"dropout":0.002,"num_experts":4,
        "top_k":4,"lr":0.000127,"weight_decay":2.75e-05,
        "tdc_name":"Pgp_Broccatelli","metric":"AUROC","task":"clf"
    },
    "bioavailability_ma": {
        "hidden":256,"num_layers":3,"dropout":0.209,"num_experts":4,
        "top_k":4,"lr":0.000808,"weight_decay":3.40e-05,
        "tdc_name":"Bioavailability_Ma","metric":"AUROC","task":"clf"
    },
    "bbb_martins": {
        "hidden":256,"num_layers":4,"dropout":0.202,"num_experts":16,
        "top_k":4,"lr":0.000527,"weight_decay":1.27e-05,
        "tdc_name":"BBB_Martins","metric":"AUROC","task":"clf"
    },
    "cyp2d6_veith": {
        "hidden":256,"num_layers":3,"dropout":0.209,"num_experts":4,
        "top_k":4,"lr":0.000624,"weight_decay":2.61e-05,
        "tdc_name":"CYP2D6_Veith","metric":"AUROC","task":"clf"
    },
    "cyp3a4_veith": {
        "hidden":256,"num_layers":4,"dropout":0.036,"num_experts":4,
        "top_k":4,"lr":0.000329,"weight_decay":5.66e-06,
        "tdc_name":"CYP3A4_Veith","metric":"AUROC","task":"clf"
    },
    "cyp2c9_veith": {
        "hidden":256,"num_layers":4,"dropout":0.057,"num_experts":4,
        "top_k":4,"lr":0.000520,"weight_decay":1.08e-05,
        "tdc_name":"CYP2C9_Veith","metric":"AUROC","task":"clf"
    },
    "cyp2d6_substrate_carbonmangels": {
        "hidden":256,"num_layers":2,"dropout":0.207,"num_experts":8,
        "top_k":2,"lr":0.000996,"weight_decay":6.38e-05,
        "tdc_name":"CYP2D6_Substrate_CarbonMangels","metric":"AUROC","task":"clf"
    },
    "cyp3a4_substrate_carbonmangels": {
        "hidden":128,"num_layers":2,"dropout":0.136,"num_experts":4,
        "top_k":2,"lr":0.000998,"weight_decay":6.18e-06,
        "tdc_name":"CYP3A4_Substrate_CarbonMangels","metric":"AUROC","task":"clf"
    },
    "cyp2c9_substrate_carbonmangels": {
        "hidden":256,"num_layers":4,"dropout":0.064,"num_experts":16,
        "top_k":3,"lr":0.000270,"weight_decay":3.82e-06,
        "tdc_name":"CYP2C9_Substrate_CarbonMangels","metric":"AUROC","task":"clf"
    },
    "herg": {
        "hidden":256,"num_layers":2,"dropout":0.222,"num_experts":8,
        "top_k":4,"lr":0.000156,"weight_decay":2.01e-05,
        "tdc_name":"hERG","metric":"AUROC","task":"clf"
    },
    "ames": {
        "hidden":256,"num_layers":4,"dropout":0.191,"num_experts":16,
        "top_k":2,"lr":0.000316,"weight_decay":3.76e-05,
        "tdc_name":"AMES","metric":"AUROC","task":"clf"
    },
    "dili": {
        "hidden":128,"num_layers":2,"dropout":0.031,"num_experts":16,
        "top_k":3,"lr":0.000835,"weight_decay":1.01e-05,
        "tdc_name":"DILI","metric":"AUROC","task":"clf"
    },
    "ppbr_az": {
        "hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,
        "top_k":2,"lr":0.000200,"weight_decay":2.89e-05,
        "tdc_name":"PPBR_AZ","metric":"mae","task":"regr"
    },
    "vdss_lombardo": {
        "hidden":256,"num_layers":3,"dropout":0.222,"num_experts":8,
        "top_k":2,"lr":0.000905,"weight_decay":2.82e-05,
        "tdc_name":"VDss_Lombardo","metric":"spearman","task":"regr"
    },
    "half_life_obach": {
        "hidden":256,"num_layers":3,"dropout":0.220,"num_experts":16,
        "top_k":4,"lr":0.000539,"weight_decay":1.83e-05,
        "tdc_name":"Half_Life_Obach","metric":"spearman","task":"regr"
    },
    "clearance_microsome_az": {
        "hidden":256,"num_layers":3,"dropout":0.003,"num_experts":16,
        "top_k":2,"lr":0.000195,"weight_decay":2.91e-05,
        "tdc_name":"Clearance_Microsome_AZ","metric":"spearman","task":"regr"
    },
    "clearance_hepatocyte_az": {
        "hidden":256,"num_layers":3,"dropout":0.221,"num_experts":16,
        "top_k":2,"lr":0.000973,"weight_decay":1.05e-06,
        "tdc_name":"Clearance_Hepatocyte_AZ","metric":"spearman","task":"regr"
    },
    # Already have specialization JSONs but include for completeness
    "solubility_aqsoldb": {
        "hidden":256,"num_layers":4,"dropout":0.153,"num_experts":8,
        "top_k":3,"lr":0.000896,"weight_decay":1.99e-05,
        "tdc_name":"Solubility_AqSolDB","metric":"mae","task":"regr"
    },
    "caco2_wang": {
        "hidden":256,"num_layers":3,"dropout":0.031,"num_experts":16,
        "top_k":4,"lr":0.000599,"weight_decay":2.01e-05,
        "tdc_name":"Caco2_Wang","metric":"mae","task":"regr"
    },
    "lipophilicity_astrazeneca": {
        "hidden":256,"num_layers":4,"dropout":0.038,"num_experts":4,
        "top_k":1,"lr":0.000972,"weight_decay":7.03e-05,
        "tdc_name":"Lipophilicity_AstraZeneca","metric":"mae","task":"regr"
    },
    "ld50_zhu": {
        "hidden":256,"num_layers":4,"dropout":0.099,"num_experts":16,
        "top_k":2,"lr":0.000470,"weight_decay":4.92e-05,
        "tdc_name":"LD50_Zhu","metric":"mae","task":"regr"
    },
}

# ══════════════════════════════════════════════════════════════════════════
# FEATURIZATION — your exact function
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
            edge_index += [[i, j], [j, i]]
            edge_attr  += [bf, bf]
        if not edge_index:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(
                edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


def smiles_to_dataset(smiles_list, labels):
    data_list, valid_smiles = [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        data_list.append(g)
        valid_smiles.append(smi)
    return data_list, valid_smiles


# ══════════════════════════════════════════════════════════════════════════
# LOAD TDC DATA — without importing tdc.single_pred directly
# ══════════════════════════════════════════════════════════════════════════

def load_tdc_dataset(tdc_name):
    """Try multiple TDC import paths."""
    split = None

    # Try path 1
    try:
        from tdc.single_pred import ADMET
        data = ADMET(name=tdc_name)
        split = data.get_split(method="scaffold", seed=42)
        return split
    except Exception:
        pass

    # Try path 2
    try:
        from tdc.single_pred import ADME
        data = ADME(name=tdc_name)
        split = data.get_split(method="scaffold", seed=42)
        return split
    except Exception:
        pass

    # Try path 3
    try:
        from tdc.single_pred import Tox
        data = Tox(name=tdc_name)
        split = data.get_split(method="scaffold", seed=42)
        return split
    except Exception:
        pass

    # Try path 4 — newer TDC API
    try:
        import tdc
        data = tdc.ADMET(name=tdc_name)
        split = data.get_split(method="scaffold", seed=42)
        return split
    except Exception:
        pass

    # Try path 5 — try all single_pred classes
    try:
        from tdc import single_pred
        for cls_name in ['ADME', 'Tox', 'HTS', 'QM', 'Yields',
                         'DrugRes', 'CRISPROutcome']:
            try:
                cls = getattr(single_pred, cls_name)
                data = cls(name=tdc_name)
                split = data.get_split(method="scaffold", seed=42)
                return split
            except Exception:
                continue
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y = batch.y.squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        loss = F.mse_loss(out, y) + 0.01 * bal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    return float(np.mean(np.abs(np.array(preds) - np.array(truths))))


def train_model(params, train_data, val_data, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    drop_last = len(train_data) % 64 == 1
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=256)

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(100):
        train_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break

    model.load_state_dict(best_state)
    print(f"    Best val loss: {best_val:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════
# ROUTING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_routing(model, all_data):
    model.eval()
    loader = DataLoader(all_data, batch_size=256)
    all_dominant = []
    all_weights  = []

    for batch in loader:
        batch = batch.to(DEVICE)
        _, routing_weights, dominant = model.forward_with_routing(batch)
        all_dominant.extend(dominant.cpu().numpy().tolist())
        all_weights.append(routing_weights.cpu().numpy())

    dominant = np.array(all_dominant)
    weights  = np.vstack(all_weights)

    unique, counts = np.unique(dominant, return_counts=True)
    print(f"    Expert usage: " +
          ", ".join(f"E{u}={c}" for u, c in zip(unique, counts)))

    return dominant, weights


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=None,
        help="Specific datasets to process. Default: all 18 without JSONs"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip if routing_{dataset}.npy already exists"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Datasets that still need routing (skip ones with existing JSON)
    NEED_ROUTING = [
        "ppbr_az", "hia_hou", "pgp_broccatelli", "bioavailability_ma",
        "bbb_martins", "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
        "cyp2c9_substrate_carbonmangels", "herg", "ames", "dili",
        "vdss_lombardo", "half_life_obach",
        "clearance_microsome_az", "clearance_hepatocyte_az",
    ]

    target_datasets = args.datasets if args.datasets else NEED_ROUTING

    print("=" * 65)
    print(f"  Routing Extraction — {len(target_datasets)} datasets")
    print("=" * 65)

    Path("models").mkdir(exist_ok=True)
    results_summary = {}

    for dataset in target_datasets:
        print(f"\n{'─'*65}")
        print(f"  [{dataset}]")

        # Check if already done
        out_npy = Path(f"routing_{dataset}.npy")
        if args.skip_existing and out_npy.exists():
            assignments = np.load(out_npy)
            print(f"  Already exists: {out_npy} "
                  f"(n={len(assignments)})")
            results_summary[dataset] = "already_exists"
            continue

        if dataset not in BEST_PARAMS:
            print(f"  No params found — skipping")
            results_summary[dataset] = "no_params"
            continue

        params = BEST_PARAMS[dataset]
        tdc_name = params["tdc_name"]
        ckpt_path = Path(f"models/moegcn_{dataset}_seed{args.seed}.pt")

        # Load data
        print(f"  Loading {tdc_name}...")
        split = load_tdc_dataset(tdc_name)

        if split is None:
            print(f"  TDC load failed — trying CSV fallback...")
            # Try loading from local CSV if TDC fails
            csv_paths = [
                Path(f"data/{dataset}.csv"),
                Path(f"datasets/{dataset}.csv"),
                Path(f"{dataset}.csv"),
            ]
            loaded = False
            for csv_path in csv_paths:
                if csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    smiles_col = next(
                        (c for c in ['Drug','smiles','SMILES']
                         if c in df.columns), df.columns[0])
                    label_col  = next(
                        (c for c in ['Y','label','activity','value']
                         if c in df.columns), df.columns[1])
                    smiles = df[smiles_col].tolist()
                    labels = df[label_col].tolist()

                    n = len(smiles)
                    t_end = int(n * 0.8)
                    v_end = int(n * 0.9)
                    split = {
                        "train": {"Drug": smiles[:t_end],
                                  "Y": labels[:t_end]},
                        "valid": {"Drug": smiles[t_end:v_end],
                                  "Y": labels[t_end:v_end]},
                        "test":  {"Drug": smiles[v_end:],
                                  "Y": labels[v_end:]},
                    }
                    print(f"  Loaded from {csv_path}: {n} molecules")
                    loaded = True
                    break

            if not loaded:
                print(f"  No data found — skipping {dataset}")
                print(f"  Tried: {[str(p) for p in csv_paths]}")
                results_summary[dataset] = "no_data"
                continue

        # Build datasets
        train_data, _ = smiles_to_dataset(
            split["train"]["Drug"], split["train"]["Y"])
        val_data,   _ = smiles_to_dataset(
            split["valid"]["Drug"], split["valid"]["Y"])
        test_data,  _ = smiles_to_dataset(
            split["test"]["Drug"],  split["test"]["Y"])
        all_data = train_data + val_data + test_data
        print(f"  Molecules: train={len(train_data)}, "
              f"val={len(val_data)}, test={len(test_data)}, "
              f"total={len(all_data)}")

        # Load or train model
        if ckpt_path.exists():
            print(f"  Loading checkpoint: {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            state = (ckpt["model_state_dict"]
                     if "model_state_dict" in ckpt else ckpt)
            model = MoEGCN(
                in_dim=9,
                hidden=params["hidden"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                num_experts=params["num_experts"],
                top_k=params["top_k"],
            ).to(DEVICE)
            model.load_state_dict(state)
        else:
            print(f"  Training model (seed={args.seed})...")
            model = train_model(
                params, train_data, val_data, seed=args.seed)
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "params": params},
                ckpt_path
            )
            print(f"  Checkpoint saved: {ckpt_path}")

        # Extract routing
        print(f"  Extracting routing assignments...")
        dominant, weights = extract_routing(model, all_data)

        # Save
        np.save(out_npy, dominant)
        np.save(f"routing_weights_{dataset}.npy", weights)
        print(f"  [SAVED] {out_npy} (n={len(dominant)})")
        print(f"  [SAVED] routing_weights_{dataset}.npy")

        results_summary[dataset] = f"done_n={len(dominant)}"

    # Summary
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    done = [d for d, s in results_summary.items()
            if s.startswith("done")]
    skipped = [d for d, s in results_summary.items()
               if s == "already_exists"]
    failed = [d for d, s in results_summary.items()
              if s in ["no_data", "no_params"]]

    print(f"  Done:    {len(done)} datasets")
    print(f"  Skipped: {len(skipped)} (already existed)")
    print(f"  Failed:  {len(failed)} datasets")
    if failed:
        print(f"  Failed:  {failed}")

    print(f"""
  NEXT STEP:
  Run specialization_from_json.py again to compute exact SSS:
      python specialization_from_json.py

  The Spearman r will update from estimated to exact values.
  Expected r > 0.70 with exact routing data.
""")


if __name__ == "__main__":
    main()
