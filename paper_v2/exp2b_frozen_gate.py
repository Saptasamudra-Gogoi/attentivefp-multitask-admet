"""
exp2_topk_dynamics.py
=====================
Does the load-over-training "phase transition" (old Fig. 2) and the expert specialization survive
when the router actually receives gradient (K>=2)?  Same model/params, only top_k varies.

For each (dataset, K, seed) it trains for a FIXED number of epochs (no early stop, so the full
trajectory is visible) and logs per epoch:
   expert load (fraction of molecules whose argmax expert is e, measured on the train set, eval mode),
   gate-weight norm, mean routing entropy, train loss, val metric.
At the best-validation epoch it stores dominant-expert labels for ALL molecules (train+valid+test,
TDC scaffold order for TDC sets; scaffold_split order for Tox21) and computes eta^2 for LogP and
ArRings (+ all 8 descriptors) so specialization at K>=2 can be compared with K=1.

Datasets:
   tox21                      (MoleculeNet via PyG, 12 tasks, E=4, original params; K in 1 2 4)
   lipophilicity_astrazeneca  (TDC, E=4, repo BEST_PARAMS; K in 1 2 4)
Run (repo root, env moe_admet):
   python -u exp2_topk_dynamics.py --datasets lipophilicity_astrazeneca --ks 1 2 4 --seeds 0 1 2
   python -u exp2_topk_dynamics.py --datasets tox21 --ks 1 2 4 --seeds 0 1 2
Resumes from exp2_topk_dynamics.json.  ~80 epochs x 9 runs per dataset; Tox21 is the slow one.
"""
import argparse, copy, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.data import DataLoader
from scipy.stats import f_oneway
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
import retrain_and_extract as R

EPOCHS = 80
DESC = {"MW": Descriptors.ExactMolWt, "LogP": Descriptors.MolLogP, "HBA": rdMolDescriptors.CalcNumHBA,
        "HBD": rdMolDescriptors.CalcNumHBD, "TPSA": Descriptors.TPSA, "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "RingCount": rdMolDescriptors.CalcNumRings, "ArRings": rdMolDescriptors.CalcNumAromaticRings}
TOX21_PARAMS = dict(hidden=256, num_layers=4, dropout=0.242, lr=5.07e-4, weight_decay=1.59e-5, num_experts=4, top_k=1)


def eta2(labels, v, min_group=10):
    ids = [e for e in np.unique(labels) if (labels == e).sum() >= min_group]
    if len(ids) < 2: return None
    g = [v[labels == e] for e in ids]; a = np.concatenate(g); gm = a.mean()
    sst = ((a - gm) ** 2).sum(); ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in g)
    return float(ssb / sst) if sst > 0 else None


def tox21_scaffold_split(ds):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    sc = defaultdict(list)
    for i in range(len(ds)):
        m = Chem.MolFromSmiles(ds.smiles[i])
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else str(i)
        sc[key].append(i)
    sets = sorted(sc.values(), key=len, reverse=True); n = len(ds)
    tr, va, te = [], [], []
    for s in sets:
        if len(tr) < int(n * .8): tr += s
        elif len(va) < int(n * .9) - int(n * .8): va += s
        else: te += s
    sub = lambda idx: torch.utils.data.Subset(ds, idx)
    return sub(tr), sub(va), sub(te), [ds.smiles[i] for i in tr + va + te]


def prepare(name):
    if name == "tox21":
        from torch_geometric.datasets import MoleculeNet
        ds = MoleculeNet(root="./data", name="Tox21", transform=lambda d: (setattr(d, "x", d.x.float()), d)[1])
        tr, va, te, smi = tox21_scaffold_split(ds)
        order = list(tr) + list(va) + list(te)
        return dict(train=tr, val=va, all=order, smiles=smi, P=TOX21_PARAMS, tasks=12, classif=True)
    P = R.BEST_PARAMS[name]; sp = R.load_tdc(P["tdc_name"])
    sets = {k: R.build_dataset(list(sp[k]["Drug"]), list(sp[k]["Y"])) for k in ("train", "valid", "test")}
    smi = sets["train"][1] + sets["valid"][1] + sets["test"][1]
    allg = list(sets["train"][0]) + list(sets["valid"][0]) + list(sets["test"][0])
    return dict(train=sets["train"][0], val=sets["valid"][0], all=allg, smiles=smi, P=P, tasks=1, classif=False)


def loss_fn(out, y, tasks, classif):
    if classif:
        y = y.float().view(-1, tasks); m = ~torch.isnan(y)
        return F.binary_cross_entropy_with_logits(out[m], y[m]) if m.any() else None
    y = y.float().view(-1); return F.mse_loss(out.view(-1), y)


@torch.no_grad()
def routing_pass(model, data):
    model.eval(); W = []
    for b in DataLoader(data, batch_size=256):
        b = b.to(R.DEVICE); _, rw, _ = model.forward_with_routing(b); W.append(rw.cpu().numpy())
    return np.vstack(W)


@torch.no_grad()
def val_score(model, data, tasks, classif):
    model.eval(); P, Y = [], []
    for b in DataLoader(data, batch_size=256):
        b = b.to(R.DEVICE); o, _ = model(b); P.append(o.view(-1, tasks).cpu()); Y.append(b.y.view(-1, tasks).cpu())
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    if not classif: return float(np.mean(np.abs(P.ravel() - Y.ravel())))        # MAE, lower better
    aucs = []
    for t in range(tasks):
        m = ~np.isnan(Y[:, t])
        if m.sum() > 1 and len(np.unique(Y[m, t])) == 2: aucs.append(roc_auc_score(Y[m, t], P[m, t]))
    return float(np.mean(aucs))                                                 # AUROC, higher better


def run(D, K, seed, frozen=False):
    P = dict(D["P"]); P["top_k"] = K; tasks, classif = D["tasks"], D["classif"]
    torch.manual_seed(seed); np.random.seed(seed)
    m = R.MoEGCN(9, P["hidden"], P["num_layers"], P["dropout"], P["num_experts"], K, num_outputs=tasks).to(R.DEVICE)
    if frozen:
        for p_ in m.moe.gate.parameters(): p_.requires_grad = False
    opt = torch.optim.Adam([p_ for p_ in m.parameters() if p_.requires_grad], lr=P["lr"], weight_decay=P["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, mode="max" if classif else "min")
    dl = DataLoader(D["train"], batch_size=64, shuffle=True, drop_last=(len(D["train"]) % 64 == 1))
    E = P["num_experts"]; log = dict(load=[], gate_norm=[], entropy=[], train_loss=[], val=[])
    best, best_ep, best_lab, best_state = (-1e9 if classif else 1e9), -1, None, None
    for ep in range(EPOCHS):
        m.train(); tl = []
        for b in dl:
            b = b.to(R.DEVICE); opt.zero_grad(); out, bal = m(b)
            l = loss_fn(out, b.y, tasks, classif)
            if l is None: continue
            l = l + 0.01 * bal; l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl.append(l.item())
        W = routing_pass(m, D["train"]); dom = W.argmax(1)
        load = np.bincount(dom, minlength=E) / len(dom)
        ent = float(-(W * np.log(W + 1e-12)).sum(1).mean())
        v = val_score(m, D["val"], tasks, classif); sched.step(v)
        log["load"].append(load.tolist()); log["gate_norm"].append(float(m.moe.gate.weight.norm())); log["entropy"].append(ent)
        log["train_loss"].append(float(np.mean(tl))); log["val"].append(v)
        if (v > best) if classif else (v < best):
            best, best_ep = v, ep; best_state = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state); lab = routing_pass(m, D["all"]).argmax(1)
    ok = []; DV = {k: [] for k in DESC}
    for s in D["smiles"]:
        mol = Chem.MolFromSmiles(str(s)); ok.append(mol is not None)
        for k, f in DESC.items(): DV[k].append(float(f(mol)) if mol is not None else np.nan)
    ok = np.array(ok); et = {}
    for k in DESC:
        v = np.array(DV[k]); mk = ok & ~np.isnan(v); et[k] = eta2(lab[mk], v[mk])
    return dict(frozen=frozen, K=K, seed=seed, E=E, best_epoch=best_ep, best_val=best, log=log, eta2=et,
                active_experts=int(sum((lab == e).sum() >= 10 for e in range(E))), labels=lab.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lipophilicity_astrazeneca", "tox21"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--frozen_gate", action="store_true", help="control: random gate never updated"); ap.add_argument("--out", default="exp2_topk_dynamics.json"); a = ap.parse_args()
    res = json.load(open(a.out)) if Path(a.out).exists() else {}
    for name in a.datasets:
        D = prepare(name)
        for K in a.ks:
            for s in a.seeds:
                key = f"{name}|K{K}|seed{s}" + ("|frozen" if a.frozen_gate else "")
                if key in res: continue
                r = run(D, K, s, a.frozen_gate); res[key] = r
                Path(a.out).write_text(json.dumps(res))
                print(f"{key:45s} best_val {r['best_val']:.4f} @ep{r['best_epoch']}  gate|W| {r['log']['gate_norm'][0]:.3f}->{r['log']['gate_norm'][-1]:.3f}"
                      f"  active {r['active_experts']}/{r['E']}  eta2 LogP {r['eta2']['LogP']} ArRings {r['eta2']['ArRings']}")
    print("done ->", a.out)


if __name__ == "__main__":
    main()
