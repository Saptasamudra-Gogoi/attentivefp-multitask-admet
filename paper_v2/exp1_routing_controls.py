"""
exp1_routing_controls.py
========================
Causal controls for the MoE-ADMET paper. Reuses the repo's own model, data loading and
training protocol (retrain_and_extract.py) so results are directly comparable.

STEP 0  Gate-gradient check (prints first). With top_k=1 the repo's MoELayer computes
        weights = softmax(topk_vals) over ONE value == 1.0 constant, so the gate receives
        zero gradient from the task loss AND from the balance loss. Verify on your machine.

STEP 1  For each regression dataset x seed, train 4 arms with IDENTICAL backbone/expert
        architecture, hyper-parameters, split and training protocol:
          learned       - the paper's router (as in retrain_and_extract.py)
          frozen_gate   - same, but gate parameters never updated (random fixed gate)
          fixed_kmeans  - hard top-1 routing by k-means on 8 RDKit descriptors (non-learned,
                          chemistry-informed partition; centroids fit on TRAIN molecules only)
          fixed_hash    - hard top-1 routing by a fixed random assignment (non-chemical partition)
        Test metric = MAE, same scaffold split (seed=42) as the paper.

STEP 2  Forced-swap test on the trained `learned` and `frozen_gate` models: for every test
        molecule force routing to each expert j and record error. M[g][j] = mean abs error of
        molecules naturally in group g when forced to expert j. Diagonal-dominant => experts are
        functionally distinct; flat => expert labels are interchangeable.

Run (repo root, conda env moe_admet):
    python exp1_routing_controls.py                       # 4 datasets x 5 seeds x 4 arms
    python exp1_routing_controls.py --datasets caco2_wang --seeds 0 1 2   # quick pass
Outputs: exp1_routing_controls.json  (+ printed summary)
Expected runtime: small-data sets are minutes per run on a GTX 1660 Ti; AqSolDB / LD50 are the slow ones.
"""
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import retrain_and_extract as R   # BEST_PARAMS, load_tdc, build_dataset, MoEGCN, DESCRIPTOR_FNS, DEVICE

DEFAULT_DS = ["solubility_aqsoldb", "caco2_wang", "ld50_zhu", "lipophilicity_astrazeneca"]
ARMS = ["learned", "frozen_gate", "fixed_kmeans", "fixed_hash"]


# ───────────────────────── STEP 0 ─────────────────────────
def check_gate_gradient():
    print("STEP 0 | gate gradient under top_k (dummy loss + balance loss, repo MoELayer)")
    for k in (1, 2, 4):
        torch.manual_seed(0)
        layer = R.MoELayer(16, 16, 8, k)
        x = torch.randn(64, 16)
        out, bal = layer(x)
        loss = out.pow(2).mean() + 0.01 * bal
        loss.backward()
        g = layer.gate.weight.grad
        print(f"   top_k={k}: ||grad(gate.weight)|| = {0.0 if g is None else g.norm().item():.3e}")
    print("   -> if top_k=1 prints 0, the K=1 router is never trained by the task or balance loss.\n")


# ───────────────────────── model with routing modes ─────────────────────────
class RoutedMoEGCN(R.MoEGCN):
    def __init__(self, *a, mode="learned", **k):
        super().__init__(*a, **k)
        self.mode, self.force, self.last_w = mode, None, None
        if mode == "frozen_gate":
            for p in self.moe.gate.parameters():
                p.requires_grad = False

    def _encode(self, data):
        x = data.x
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return R.global_mean_pool(x, data.batch)

    def forward(self, data):
        x = self._encode(data)
        E = self.moe.num_experts
        if self.force is not None:
            w = F.one_hot(torch.full((x.size(0),), int(self.force), device=x.device), E).float()
        elif self.mode in ("fixed_kmeans", "fixed_hash"):
            w = F.one_hot(data.route.view(-1).long(), E).float()
        else:
            logits = self.moe.gate(x)
            tv, ti = torch.topk(logits, self.moe.top_k, dim=-1)
            w = torch.zeros_like(logits).scatter_(1, ti, F.softmax(tv, dim=-1))
        self.last_w = w.detach()
        bal = E * (w.mean(0) ** 2).sum()
        eo = torch.stack([e(x) for e in self.moe.experts], dim=1)
        out = (w.unsqueeze(-1) * eo).sum(dim=1)
        return self.head(out).squeeze(-1), bal


def train_arm(params, mode, train, val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RoutedMoEGCN(9, params["hidden"], params["num_layers"], params["dropout"],
                     params["num_experts"], params["top_k"], mode=mode).to(R.DEVICE)
    gate0 = m.moe.gate.weight.detach().norm().item()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, mode="min")
    dl = DataLoader(train, batch_size=64, shuffle=True, drop_last=(len(train) % 64 == 1))
    vl = DataLoader(val, batch_size=256)
    best, state, pat = 1e9, None, 0
    for ep in range(150):
        R.train_one_epoch(m, dl, opt, "regression")
        v = R.evaluate(m, vl, "regression", "mae"); sched.step(v)
        if v < best:
            best, state, pat = v, copy.deepcopy(m.state_dict()), 0
        else:
            pat += 1
            if pat >= 20:
                break
    m.load_state_dict(state)
    return m, best, gate0, m.moe.gate.weight.detach().norm().item()


# ───────────────────────── helpers ─────────────────────────
def desc_matrix(smiles):
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append([float(f(mol)) if mol is not None else np.nan for f in R.DESCRIPTOR_FNS.values()])
    X = np.array(rows)
    col_mean = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    return X


def attach_routes(datasets, routes):
    for ds, r in zip(datasets, routes):
        for g, ri in zip(ds, r):
            g.route = torch.tensor([int(ri)])


@torch.no_grad()
def per_mol_error(model, data, force=None):
    model.eval(); model.force = force
    errs, nat = [], []
    for b in DataLoader(data, batch_size=256):
        b = b.to(R.DEVICE)
        out, _ = model(b)
        errs.append((out - b.y.view(-1)).abs().cpu().numpy())
        nat.append(model.last_w.argmax(1).cpu().numpy() if force is None else None)
    model.force = None
    return np.concatenate(errs), (np.concatenate(nat) if force is None else None)


def swap_matrix(model, test, E, min_n=10):
    e_nat, grp = per_mol_error(model, test, None)
    M = np.full((E, E), np.nan)
    for j in range(E):
        e_j, _ = per_mol_error(model, test, j)
        for g in range(E):
            if (grp == g).sum() >= min_n:
                M[g, j] = e_j[grp == g].mean()
    rows = [g for g in range(E) if (grp == g).sum() >= min_n]
    diag = np.array([M[g, g] for g in rows]); off = np.array([np.nanmean(np.delete(M[g], g)) for g in rows])
    return dict(matrix=np.where(np.isnan(M), None, M).tolist(), groups=rows, n_per_group=[int((grp == g).sum()) for g in rows],
                natural_mae=float(e_nat.mean()),
                mean_diag=float(diag.mean()) if len(rows) else None, mean_offdiag=float(off.mean()) if len(rows) else None,
                offdiag_over_diag=float(off.mean() / diag.mean()) if len(rows) else None,
                rows_with_diag_min=int(sum(np.nanargmin(M[g]) == g for g in rows)), n_rows=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--out", default="exp1_routing_controls.json")
    a = ap.parse_args()
    check_gate_gradient()
    res = json.load(open(a.out)) if Path(a.out).exists() else {}
    for ds in a.datasets:
        P = R.BEST_PARAMS[ds]
        assert P["task"] == "regression", "exp1 is regression-only (MAE)"
        split = R.load_tdc(P["tdc_name"])
        sets = {k: R.build_dataset(list(split[k]["Drug"]), list(split[k]["Y"])) for k in ("train", "valid", "test")}
        data = {k: v[0] for k, v in sets.items()}; smi = {k: v[1] for k, v in sets.items()}
        E = P["num_experts"]
        X = {k: desc_matrix(smi[k]) for k in data}
        for seed in a.seeds:
            sc = StandardScaler().fit(X["train"])
            km = KMeans(E, n_init=10, random_state=seed).fit(sc.transform(X["train"]))
            rng = np.random.default_rng(1000 + seed)
            for mode in a.arms:
                key = f"{ds}|seed{seed}|{mode}"
                if key in res:
                    continue
                if mode == "fixed_kmeans":
                    attach_routes([data[k] for k in data], [km.predict(sc.transform(X[k])) for k in data])
                elif mode == "fixed_hash":
                    attach_routes([data[k] for k in data], [rng.integers(0, E, len(data[k])) for k in data])
                m, val, g0, g1 = train_arm(P, mode, data["train"], data["valid"], seed)
                test_mae = R.evaluate(m, DataLoader(data["test"], batch_size=256), "regression", "mae")
                rec = dict(val_mae=val, test_mae=test_mae, gate_norm_init=g0, gate_norm_final=g1, E=E, top_k=P["top_k"])
                if mode in ("learned", "frozen_gate"):
                    rec["swap"] = swap_matrix(m, data["test"], E)
                res[key] = rec
                print(f"{key:55s} test MAE {test_mae:.4f}  gate|W| {g0:.3f}->{g1:.3f}"
                      + (f"  swap off/diag {rec['swap']['offdiag_over_diag']:.3f}" if "swap" in rec and rec['swap']['offdiag_over_diag'] else ""))
                Path(a.out).write_text(json.dumps(res, indent=1))
    # summary
    print("\n=== SUMMARY: test MAE mean +/- sd over seeds ===")
    for ds in a.datasets:
        line = f"{ds:28s}"
        for mode in a.arms:
            v = [res[k]["test_mae"] for k in res if k.startswith(ds + "|") and k.endswith("|" + mode)]
            if v: line += f" | {mode} {np.mean(v):.3f}+/-{np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
        print(line)
    print("\nInterpretation guide (decide BEFORE looking):")
    print(" learned ~ frozen_gate  -> training the router adds nothing at this capacity.")
    print(" learned < fixed_hash   -> chemistry-aligned partition helps; compare with fixed_kmeans to see if learning beats a simple chemical partition.")
    print(" swap off/diag >> 1     -> experts functionally distinct. ~1 -> labels interchangeable; drop 'specialization' as a functional claim.")


if __name__ == "__main__":
    main()
