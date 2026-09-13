"""
plain_gcn_kmeans_control.py
============================
P0-4 fix: the existing kmeans_control_all.json clusters the pooled
representation extracted from the TRAINED MoE-GCN checkpoint
(models/moegcn_{ds}_seed0_v2.pt) -- a representation shaped end-to-end
by the router. That control is circular: it cannot show the chemical
organization existed independently of routing.

This script trains a genuinely independent plain GCN (no MoE layer,
no router, no routing loss) on the same 8 TDC datasets, with the SAME
backbone capacity (hidden, num_layers) as each dataset's MoE model --
taken from retrain_and_extract.py's BEST_PARAMS -- so capacity cannot
be the confound. It then extracts the pooled representation and runs
the identical k-means protocol as kmeans_control.py:
  - same 4 descriptors (MW, LogP, TPSA, ArRings)
  - same n_clusters = num_experts (per dataset)
  - same KMeans(random_state=42, n_init=10)
  - same eta_squared formula

Output: kmeans_control_plain_gcn.json -- directly comparable, dataset
by dataset and descriptor by descriptor, to kmeans_control_all.json.

If plain-GCN eta2 is comparable to MoE-representation eta2, the paper's
"router inherits, does not create" claim is supported. If it is not,
the claim must be rewritten to "routing co-shapes the representation."

Run: python plain_gcn_kmeans_control.py
Estimated time: a few hours on GTX 1660 Ti (8 datasets, up to 9980 mols each)
"""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EPOCHS   = 150
PATIENCE = 20
BATCH    = 64
SAVE_PATH = "kmeans_control_plain_gcn.json"

# -- Same 8 datasets, same n_experts (=k for clustering) as kmeans_control_all.json --
DATASETS = {
    "caco2_wang":               ("ADME", "Caco2_Wang", 4),
    "solubility_aqsoldb":       ("ADME", "Solubility_AqSolDB", 8),
    "lipophilicity_astrazeneca":("ADME", "Lipophilicity_AstraZeneca", 4),
    "ld50_zhu":                 ("Tox",  "LD50_Zhu", 16),
    "bbb_martins":              ("ADME", "BBB_Martins", 16),
    "herg":                     ("Tox",  "hERG", 8),
    "ames":                     ("Tox",  "AMES", 16),
    "dili":                     ("Tox",  "DILI", 16),
}

# -- Backbone capacity per dataset, taken verbatim from retrain_and_extract.py's
#    BEST_PARAMS so the plain GCN has the SAME hidden/num_layers as its MoE
#    counterpart. Only hidden/num_layers/dropout/lr/wd/task/metric are used;
#    num_experts/top_k are irrelevant here (no MoE layer).
BEST_PARAMS = {
    "caco2_wang": {
        "hidden": 256, "num_layers": 3, "dropout": 0.031,
        "lr": 0.000599, "weight_decay": 2.01e-05,
        "task": "regression", "metric": "mae",
    },
    "solubility_aqsoldb": {
        "hidden": 256, "num_layers": 4, "dropout": 0.153,
        "lr": 0.000896, "weight_decay": 1.99e-05,
        "task": "regression", "metric": "mae",
    },
    "lipophilicity_astrazeneca": {
        "hidden": 256, "num_layers": 4, "dropout": 0.038,
        "lr": 0.000972, "weight_decay": 7.03e-05,
        "task": "regression", "metric": "mae",
    },
    "ld50_zhu": {
        "hidden": 256, "num_layers": 4, "dropout": 0.099,
        "lr": 0.000470, "weight_decay": 4.92e-05,
        "task": "regression", "metric": "mae",
    },
    "bbb_martins": {
        "hidden": 256, "num_layers": 4, "dropout": 0.202,
        "lr": 0.000527, "weight_decay": 1.27e-05,
        "task": "classification", "metric": "auroc",
    },
    "herg": {
        "hidden": 256, "num_layers": 2, "dropout": 0.222,
        "lr": 0.000156, "weight_decay": 2.01e-05,
        "task": "classification", "metric": "auroc",
    },
    "ames": {
        "hidden": 256, "num_layers": 4, "dropout": 0.191,
        "lr": 0.000316, "weight_decay": 3.76e-05,
        "task": "classification", "metric": "auroc",
    },
    "dili": {
        "hidden": 128, "num_layers": 2, "dropout": 0.031,
        "lr": 0.000835, "weight_decay": 1.01e-05,
        "task": "classification", "metric": "auroc",
    },
}

DESC = {
    "MW":      Descriptors.ExactMolWt,
    "LogP":    Descriptors.MolLogP,
    "TPSA":    Descriptors.TPSA,
    "ArRings": rdMolDescriptors.CalcNumAromaticRings,
}


# -- Plain GCN: identical backbone to MoEGCN in retrain_and_extract.py, no MoE layer --
class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_outputs=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, num_outputs)

    def encode(self, data):
        """Pooled representation -- same extraction point as kmeans_control.py's
        MoEGCNProbe.encode() (post-backbone, pre-task-head)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, data):
        x = self.encode(data)
        return self.head(x).squeeze(-1)


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
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        data_list.append(g)
        valid_smiles.append(smi)
    return data_list, valid_smiles


def load_tdc(tdc_name, cls_name):
    from tdc.single_pred import ADME, Tox
    Cls = ADME if cls_name == "ADME" else Tox
    data = Cls(name=tdc_name)
    return data.get_split(method="scaffold", seed=42)


def train_epoch(model, loader, optimizer, task):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.squeeze()
        if y.dim() == 0: y = y.unsqueeze(0)
        if out.dim() == 0: out = out.unsqueeze(0)
        loss = (F.mse_loss(out, y) if task == "regression"
                else F.binary_cross_entropy_with_logits(out, y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, task, metric):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        if out.dim() == 0: out = out.unsqueeze(0)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    preds, truths = np.array(preds), np.array(truths)
    if metric == "mae":
        return float(np.mean(np.abs(preds - truths)))
    else:  # auroc
        from sklearn.metrics import roc_auc_score
        from scipy.special import expit
        try:
            return float(roc_auc_score(truths, expit(preds)))
        except Exception:
            return 0.5


def higher_is_better(metric):
    return metric == "spearman" or metric == "auroc"


def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0


def main():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            results = json.load(f)
        print(f"Resuming -- {len(results)} datasets done")
    else:
        results = []
        done = set()

    done = {r["dataset"] for r in results} if isinstance(results, list) else set()
    if isinstance(results, dict):
        results = list(results.values())

    for ds_key, (cls_name, tdc_name, n_experts) in DATASETS.items():
        if ds_key in done:
            print(f"  Skipping {ds_key} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"  Plain GCN (capacity-matched) | {ds_key}")
        print(f"{'='*60}")
        t0 = time.time()

        params = BEST_PARAMS[ds_key]
        split = load_tdc(tdc_name, cls_name)

        train_data, _ = build_dataset(split["train"]["Drug"], split["train"]["Y"])
        val_data, _   = build_dataset(split["valid"]["Drug"], split["valid"]["Y"])
        test_data, _  = build_dataset(split["test"]["Drug"], split["test"]["Y"])
        all_data = train_data + val_data + test_data
        all_smi  = (list(split["train"]["Drug"]) + list(split["valid"]["Drug"])
                    + list(split["test"]["Drug"]))
        # build_dataset drops invalid SMILES; re-filter all_smi to match all_data length
        # by re-deriving valid smiles directly:
        _, train_smi = build_dataset(split["train"]["Drug"], split["train"]["Y"])
        _, val_smi   = build_dataset(split["valid"]["Drug"], split["valid"]["Y"])
        _, test_smi  = build_dataset(split["test"]["Drug"], split["test"]["Y"])
        all_smi = train_smi + val_smi + test_smi

        print(f"  n={len(all_data)} molecules")

        torch.manual_seed(0)
        np.random.seed(0)

        model = PlainGCN(in_dim=9, hidden=params["hidden"],
                          num_layers=params["num_layers"],
                          dropout=params["dropout"]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
            mode="max" if higher_is_better(params["metric"]) else "min")

        train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=256)

        best_val = -float("inf") if higher_is_better(params["metric"]) else float("inf")
        best_state, patience_count = None, 0

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, params["task"])
            val_score = evaluate(model, val_loader, params["task"], params["metric"])
            scheduler.step(val_score)
            improved = (val_score > best_val if higher_is_better(params["metric"])
                        else val_score < best_val)
            if improved:
                best_val = val_score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        test_score = evaluate(model, DataLoader(test_data, batch_size=256),
                               params["task"], params["metric"])
        print(f"  Test {params['metric']}: {test_score:.4f}")

        # -- Extract pooled representation over ALL molecules, same as kmeans_control.py --
        model.eval()
        full_loader = DataLoader(all_data, batch_size=256)
        embeds = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(DEVICE)
                embeds.append(model.encode(batch).cpu().numpy())
        embeds = np.vstack(embeds)

        # -- Identical k-means protocol to kmeans_control.py --
        km = KMeans(n_clusters=n_experts, random_state=42, n_init=10).fit(embeds)
        labels = km.labels_

        eta2_kmeans = {}
        for name, fn in DESC.items():
            vals = []
            for smi in all_smi:
                try:
                    vals.append(fn(Chem.MolFromSmiles(smi)))
                except Exception:
                    vals.append(np.nan)
            vals = np.array(vals)
            mask = ~np.isnan(vals)
            groups = [vals[mask][labels[mask] == k] for k in np.unique(labels)
                      if (labels[mask] == k).sum() >= 3]
            eta2_kmeans[name] = eta_squared(groups) if len(groups) >= 2 else None

        elapsed = (time.time() - t0) / 60
        result = {
            "dataset": ds_key,
            "n_molecules": len(all_smi),
            "n_clusters": n_experts,
            "test_score": round(test_score, 4),
            "metric": params["metric"],
            "kmeans_eta2": eta2_kmeans,
            "time_min": round(elapsed, 1),
        }
        results.append(result)
        print(f"  eta2: {eta2_kmeans}")
        print(f"  ({elapsed:.1f} min)")

        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results -> {SAVE_PATH}")

    # -- Quick side-by-side vs the MoE-representation control, if available --
    if os.path.exists("kmeans_control_all.json"):
        with open("kmeans_control_all.json") as f:
            moe_control = {r["dataset"]: r["kmeans_eta2"] for r in json.load(f)}
        print(f"\n{'='*70}")
        print("  PLAIN-GCN vs MoE-REPRESENTATION K-MEANS ETA2 (P0-4 comparison)")
        print(f"{'='*70}")
        for r in results:
            ds = r["dataset"]
            if ds not in moe_control:
                continue
            print(f"\n  {ds}:")
            for desc in DESC:
                plain_v = r["kmeans_eta2"].get(desc)
                moe_v = moe_control[ds].get(desc)
                if plain_v is None or moe_v is None:
                    continue
                print(f"    {desc:8} plain_GCN={plain_v:.4f}  MoE_repr={moe_v:.4f}  "
                      f"diff={plain_v - moe_v:+.4f}")


if __name__ == "__main__":
    main()
