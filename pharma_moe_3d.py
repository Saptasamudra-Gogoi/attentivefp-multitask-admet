"""
3D Conformer Pharmacophore-Guided MoE-GCN
Fix: use AllChem.EmbedMolecule instead of ETKDGv3
"""
import json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader, Data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from scipy.stats import ttest_rel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(name):
    from tdc import single_pred
    cfg = {
        'solubility_aqsoldb': ('Solubility_AqSolDB', 'mae',   'ADME'),
        'caco2_wang':          ('Caco2_Wang',          'mae',   'ADME'),
        'bbb_martins':         ('BBB_Martins',          'auroc', 'ADME'),
        'herg':                ('hERG',                 'auroc', 'Tox'),
        'ld50_zhu':            ('LD50_Zhu',             'mae',   'Tox'),
    }
    tdc_name, metric, cls = cfg[name]
    data = getattr(single_pred, cls)(name=tdc_name)
    return data.get_split(method='scaffold'), metric

def extract_pharma_2d(smiles_list):
    feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(np.zeros(7, dtype=np.float32)); continue
        feats.append(np.array([
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 100.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Descriptors.MolWt(mol) / 500.0,
        ], dtype=np.float32))
    arr = np.stack(feats)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8), 7

def get_3d_pharmacophore(mol):
    try:
        mol_h = Chem.AddHs(mol)
        # Use basic EmbedMolecule — works on this system
        r = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=10)
        if r == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        conf = mol_h.GetConformer()

        positions = {}
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() == 1: continue
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            positions[idx] = np.array([pos.x, pos.y, pos.z])

        hbd, hba, hydro, arom = [], [], [], []
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() == 1: continue
            idx = atom.GetIdx()
            if idx not in positions: continue
            an = atom.GetAtomicNum()
            if an in [7,8] and atom.GetTotalNumHs() > 0:
                hbd.append(positions[idx])
            if an in [7,8]:
                hba.append(positions[idx])
            if an == 6 and not atom.GetIsAromatic():
                hydro.append(positions[idx])

        ri = mol_h.GetRingInfo()
        for ring in ri.AtomRings():
            pts = [positions[a] for a in ring
                   if a in positions and mol_h.GetAtomWithIdx(a).GetIsAromatic()]
            if pts: arom.append(np.mean(pts, axis=0))

        feat = np.zeros(24, dtype=np.float32)
        for i, c in enumerate(hbd[:3]):   feat[i*3:(i+1)*3] = c
        for i, c in enumerate(hba[:3]):   feat[9+i*3:9+(i+1)*3] = c
        if hydro: feat[18:21] = np.mean(hydro, axis=0)
        if arom:  feat[21:24] = np.mean(arom,  axis=0)
        return feat
    except:
        return None

def extract_pharma_3d(smiles_list):
    feats = []; failed = 0
    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f'    3D: {i}/{len(smiles_list)} ({failed} failed)')
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(np.zeros(24, dtype=np.float32)); failed += 1; continue
        f = get_3d_pharmacophore(mol)
        feats.append(f if f is not None else np.zeros(24, dtype=np.float32))
        if f is None: failed += 1
    arr = np.stack(feats)
    print(f'    3D done: {len(smiles_list)} mols, {failed} failed ({100*failed/len(smiles_list):.1f}%)')
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8), 24

def smiles_to_pyg(smiles_list, labels):
    pts = []
    for smi, y in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        atoms = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic()),
                  int(a.IsInRing()), a.GetFormalCharge(),
                  a.GetTotalNumHs(), a.GetNumRadicalElectrons()]
                 for a in mol.GetAtoms()]
        if not atoms: continue
        x = torch.tensor(atoms, dtype=torch.float)
        ei = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei += [[i,j],[j,i]]
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() \
                     if ei else torch.zeros((2,0), dtype=torch.long)
        pts.append(Data(x=x, edge_index=edge_index,
                        y=torch.tensor([float(y)], dtype=torch.float)))
    return pts

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            self.convs.append(GCNConv(d, hidden))
            self.bns.append(torch.nn.BatchNorm1d(hidden))
    def forward(self, x, ei, batch):
        for c, bn in zip(self.convs, self.bns):
            x = F.relu(bn(c(x, ei)))
        return global_mean_pool(x, batch)

class PharmaModel(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers, n_exp, top_k, pharma_dim, use_pharma):
        super().__init__()
        self.use_pharma = use_pharma
        self.n_exp = n_exp; self.top_k = top_k
        self.encoder = GCNEncoder(in_dim, hidden, layers)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, hidden))
            for _ in range(n_exp)])
        if use_pharma:
            self.g_gate = torch.nn.Linear(hidden, n_exp)
            self.p_enc  = torch.nn.Sequential(
                torch.nn.Linear(pharma_dim, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, n_exp))
            self.alpha  = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = torch.nn.Linear(hidden, n_exp)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, data, pharma=None):
        h = self.encoder(data.x.float(), data.edge_index, data.batch)
        if self.use_pharma:
            a = torch.sigmoid(self.alpha)
            g = a * self.g_gate(h) + (1-a) * self.p_enc(pharma)
        else:
            g = self.gate(h)
        tv, ti = torch.topk(g, self.top_k, dim=-1)
        w = torch.zeros_like(g).scatter_(1, ti, F.softmax(tv, dim=-1))
        bal = self.n_exp * (w.mean(0)**2).sum()
        out = (w.unsqueeze(-1) * torch.stack([e(h) for e in self.experts],1)).sum(1)
        return self.head(out).squeeze(-1), bal

def run(tr, va, te, ph_tr, ph_va, ph_te, pharma_dim, use_pharma, metric, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PharmaModel(tr[0].x.shape[1], 256, 3, 8, 2,
                        pharma_dim, use_pharma).to(DEVICE)
    opt = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    best_val, best_test, wait = float('inf'), 0.0, 0

    def score_set(ds, ph):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        preds, trues = [], []
        idx = 0
        for b in loader:
            b = b.to(DEVICE); bs = b.num_graphs
            p = torch.tensor(ph[idx:idx+bs], dtype=torch.float).to(DEVICE) \
                if use_pharma else None
            idx += bs
            o, _ = model(b, p)
            preds.extend(o.cpu().numpy())
            trues.extend(b.y.squeeze().cpu().numpy())
        p, t = np.array(preds), np.array(trues)
        if metric == 'mae': return float(np.mean(np.abs(p-t)))
        from sklearn.metrics import roc_auc_score
        try: return float(roc_auc_score(t, p))
        except: return 0.5

    for epoch in range(150):
        model.train()
        idx = 0
        for batch in tr_loader:
            batch = batch.to(DEVICE); bs = batch.num_graphs
            ph = torch.tensor(ph_tr[idx:idx+bs], dtype=torch.float).to(DEVICE) \
                 if use_pharma else None
            idx += bs
            out, bal = model(batch, ph)
            y = batch.y.squeeze()
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            loss = F.mse_loss(out[mask], y[mask]) + 0.01*bal
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vs = score_set(va, ph_va)
            improved = vs < best_val if metric=='mae' else vs > (1-best_val)
            if improved:
                best_val = vs if metric=='mae' else 1-vs
                best_test = score_set(te, ph_te)
                wait = 0
            else:
                wait += 1
                if wait >= 20: break

    return best_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',   nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--datasets',nargs='+',
        default=['solubility_aqsoldb','bbb_martins','herg','caco2_wang','ld50_zhu'])
    args = parser.parse_args()

    print('Device:', DEVICE)
    print('='*70)
    print('  3D Pharmacophore-Guided MoE-GCN: Standard vs 2D vs 3D')
    print('='*70)

    all_results = {}
    for ds in args.datasets:
        print(f'\n{"─"*70}\n  [{ds}]')
        split, metric = load_dataset(ds)
        tr_df, va_df, te_df = split['train'], split['valid'], split['test']
        tr = smiles_to_pyg(tr_df['Drug'].tolist(), tr_df['Y'].tolist())
        va = smiles_to_pyg(va_df['Drug'].tolist(), va_df['Y'].tolist())
        te = smiles_to_pyg(te_df['Drug'].tolist(), te_df['Y'].tolist())

        print('  2D features...')
        ph2_tr, d2 = extract_pharma_2d(tr_df['Drug'].tolist())
        ph2_va, _  = extract_pharma_2d(va_df['Drug'].tolist())
        ph2_te, _  = extract_pharma_2d(te_df['Drug'].tolist())

        print('  3D features...')
        ph3_tr, d3 = extract_pharma_3d(tr_df['Drug'].tolist())
        ph3_va, _  = extract_pharma_3d(va_df['Drug'].tolist())
        ph3_te, _  = extract_pharma_3d(te_df['Drug'].tolist())

        ss, s2, s3 = [], [], []
        for seed in args.seeds:
            sv  = run(tr,va,te, None,   None,   None,   7,  False, metric, seed)
            s2v = run(tr,va,te, ph2_tr, ph2_va, ph2_te, d2, True,  metric, seed)
            s3v = run(tr,va,te, ph3_tr, ph3_va, ph3_te, d3, True,  metric, seed)
            ss.append(sv); s2.append(s2v); s3.append(s3v)
            g2 = (sv-s2v)/sv*100 if metric=='mae' else (s2v-sv)/sv*100
            g3 = (sv-s3v)/sv*100 if metric=='mae' else (s3v-sv)/sv*100
            print(f'  Seed {seed}: Std={sv:.4f}  2D={s2v:.4f}({g2:+.1f}%)  3D={s3v:.4f}({g3:+.1f}%)')

        sm,p2m,p3m = np.mean(ss),np.mean(s2),np.mean(s3)
        g2 = (sm-p2m)/sm*100 if metric=='mae' else (p2m-sm)/sm*100
        g3 = (sm-p3m)/sm*100 if metric=='mae' else (p3m-sm)/sm*100
        g32= (p2m-p3m)/p2m*100 if metric=='mae' else (p3m-p2m)/p2m*100

        comp = lambda a,b: ([-x for x in b] if metric=='mae' else b)
        _,pv2 = ttest_rel(s2, comp(s2,ss))
        _,pv3 = ttest_rel(s3, comp(s3,ss))
        _,pv32= ttest_rel(s3, comp(s3,s2))
        sig = lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

        print(f'\n  RESULT [{ds}]')
        print(f'  Standard:  {sm:.4f}')
        print(f'  2D Pharma: {p2m:.4f}  vs std={g2:+.1f}%  p={pv2:.4f} {sig(pv2)}')
        print(f'  3D Pharma: {p3m:.4f}  vs std={g3:+.1f}%  p={pv3:.4f} {sig(pv3)}')
        print(f'  3D vs 2D:  {g32:+.1f}%  p={pv32:.4f} {sig(pv32)}')

        all_results[ds] = {
            'standard': float(sm),
            '2d_gain': float(g2), '2d_p': float(pv2),
            '3d_gain': float(g3), '3d_p': float(pv3),
            '3d_vs_2d': float(g32), '3d_vs_2d_p': float(pv32),
            'metric': metric
        }

    print('\n'+'='*70)
    print('  FINAL SUMMARY')
    print('='*70)
    print(f'  {"Dataset":<25} {"2D Gain":>9} {"3D Gain":>9} {"3D>2D":>7}')
    print('  '+'-'*53)
    for ds, r in all_results.items():
        better = '✓' if r['3d_gain'] > r['2d_gain'] else '✗'
        print(f'  {ds:<25} {r["2d_gain"]:>+8.1f}% {r["3d_gain"]:>+8.1f}% {better:>7}')

    with open('pharma_3d_results.json','w') as f:
        json.dump(all_results, f, indent=2)
    print('\n[SAVED] pharma_3d_results.json')

if __name__ == '__main__':
    main()
