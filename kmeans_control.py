import torch, json, numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader as GeoLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoEGCNProbe(nn.Module):
    def __init__(self, in_dim, hidden, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)

def infer_arch(state_dict):
    conv_keys = sorted([k for k in state_dict if k.startswith("convs.") and "lin.weight" in k])
    num_layers = len(conv_keys)
    hidden = state_dict[conv_keys[0]].shape[0]
    in_dim = state_dict[conv_keys[0]].shape[1]
    return in_dim, hidden, num_layers

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    feat = []
    for atom in mol.GetAtoms():
        feat.append([atom.GetAtomicNum(), int(atom.GetChiralTag()), atom.GetDegree(),
                     atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                     int(atom.GetHybridization()), int(atom.GetIsAromatic()), int(atom.IsInRing())])
    x = torch.tensor(feat, dtype=torch.float)
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=ei)

def eta_squared(groups):
    all_v = np.concatenate(groups)
    grand = np.mean(all_v)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_v - grand) ** 2)
    return float(ssb / sst) if sst > 0 else 0.0

DESC = {
    "MW": Descriptors.ExactMolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "ArRings": rdMolDescriptors.CalcNumAromaticRings,
}

def run_control(dataset_name, ckpt_path, smiles_list, num_experts):
    sd = torch.load(ckpt_path, map_location=DEVICE)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    in_dim, hidden, num_layers = infer_arch(sd)
    model = MoEGCNProbe(in_dim, hidden, num_layers).to(DEVICE)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()}, strict=False)
    model.eval()

    graphs, valid_smi = [], []
    for smi in smiles_list:
        g = mol_to_graph(smi)
        if g:
            graphs.append(g)
            valid_smi.append(smi)

    loader = GeoLoader(graphs, batch_size=256)
    embeds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            embeds.append(model.encode(batch).cpu().numpy())
    embeds = np.vstack(embeds)

    km = KMeans(n_clusters=num_experts, random_state=42, n_init=10).fit(embeds)
    labels = km.labels_

    eta2_kmeans = {}
    for name, fn in DESC.items():
        vals = []
        for smi in valid_smi:
            try:
                vals.append(fn(Chem.MolFromSmiles(smi)))
            except Exception:
                vals.append(np.nan)
        vals = np.array(vals)
        mask = ~np.isnan(vals)
        groups = [vals[mask][labels[mask] == k] for k in np.unique(labels) if (labels[mask] == k).sum() >= 3]
        eta2_kmeans[name] = eta_squared(groups) if len(groups) >= 2 else None

    return {
        "dataset": dataset_name,
        "n_molecules": len(valid_smi),
        "n_clusters": num_experts,
        "kmeans_eta2": eta2_kmeans,
    }