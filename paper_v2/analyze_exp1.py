import json, numpy as np, collections
from scipy import stats
D=json.load(open("/mnt/user-data/attachments/12140505-d239-5d62-8472-c43de77b96c2/42f3edf7-exp1_routing_controls.json"))
runs=collections.defaultdict(dict)   # ds -> arm -> {seed: rec}
for k,v in D.items():
    ds,s,arm=k.split("|"); runs[ds].setdefault(arm,{})[int(s[4:])]=v
print("counts:",{ds:{a:len(x) for a,x in arms.items()} for ds,arms in runs.items()})
ARMS=["learned","frozen_gate","fixed_kmeans","fixed_hash"]
out={}
print("\n== TEST MAE mean±sd (n seeds), K, E")
for ds,arms in runs.items():
    K=next(iter(arms["learned"].values()))["top_k"]; E=next(iter(arms["learned"].values()))["E"]
    row=[]
    for a in ARMS:
        v=np.array([arms[a][s]["test_mae"] for s in sorted(arms[a])]); row.append(f"{a} {v.mean():.3f}±{v.std(ddof=1):.3f}")
    print(f"{ds:28s} K={K} E={E:2d} n={len(arms['learned'])} | "+" | ".join(row))
print("\n== PAIRED differences (A−B test MAE; negative = A better), seeds paired")
pairs=[("learned","frozen_gate"),("learned","fixed_kmeans"),("learned","fixed_hash"),("frozen_gate","fixed_kmeans"),("frozen_gate","fixed_hash"),("fixed_kmeans","fixed_hash")]
for ds,arms in runs.items():
    seeds=sorted(set.intersection(*[set(arms[a]) for a in ARMS]))
    for A,B in pairs:
        d=np.array([arms[A][s]["test_mae"]-arms[B][s]["test_mae"] for s in seeds])
        p=stats.ttest_1samp(d,0).pvalue if len(d)>2 else float('nan')
        print(f"{ds:28s} {A:12s} − {B:12s}: mean {d.mean():+.4f}  A better in {int((d<0).sum())}/{len(d)} seeds  paired-t p={p:.3f}")
print("\n== SWAP off/diag: all columns (as logged) vs ACTIVE columns only (groups with >=10 test molecules)")
def active_ratio(sw):
    M=np.array([[np.nan if x is None else x for x in r] for r in sw["matrix"]],float)
    g=sw["groups"]
    if len(g)<2: return np.nan,np.nan,0
    sub=M[np.ix_(g,g)]
    diag=np.diag(sub); off=np.array([np.mean(np.delete(sub[i],i)) for i in range(len(g))])
    dm=sum(int(np.argmin(sub[i])==i) for i in range(len(g)))
    return off.mean()/diag.mean(), np.mean(off/diag), dm/len(g)
for ds,arms in runs.items():
    for a in ("learned","frozen_gate"):
        allr=[];act=[];rowd=[];dead=[]
        for s,r in sorted(arms[a].items()):
            sw=r["swap"]; allr.append(sw["offdiag_over_diag"] if sw["offdiag_over_diag"] else np.nan)
            ar,_,fd=active_ratio(sw); act.append(ar); rowd.append(fd)
            M=np.array([[np.nan if x is None else x for x in rr] for rr in sw["matrix"]],float)
            col_err=np.nanmean(M,axis=0); dead.append(int(np.sum(col_err>2.0)))
        print(f"{ds:28s} {a:12s} all-col {np.nanmean(allr):.2f} | active-only {np.nanmean(act):.2f} (per seed {[round(x,2) for x in act]}) | frac rows diag-min {np.mean(rowd):.2f} | dead cols(mean err>2) per seed {dead}")
print("\n== gate norm init->final (learned)")
for ds,arms in runs.items():
    print(ds,[ (round(r['gate_norm_init'],2),round(r['gate_norm_final'],2)) for s,r in sorted(arms['learned'].items())])
