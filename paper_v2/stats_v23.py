import json,numpy as np,collections
from scipy import stats
D=json.load(open("exp1_routing_controls.json"));R=collections.defaultdict(dict)
for k,v in D.items():
    d,s,a=k.split("|");R[d].setdefault(a,{})[int(s[4:])]=v
out={}
def act(sw):
    M=np.array([[np.nan if x is None else x for x in r] for r in sw["matrix"]],float);g=sw["groups"]
    if len(g)<2:return np.nan
    sub=M[np.ix_(g,g)];return np.mean([np.mean(np.delete(sub[i],i)) for i in range(len(g))])/np.mean(np.diag(sub))
for d in R:
    o={};L=R[d]
    for a in L:
        v=[L[a][s]["test_mae"] for s in sorted(L[a])];o[a]=dict(n=len(v),mean=np.mean(v),sd=np.std(v,ddof=1),seeds=sorted(L[a]))
    for b in ("frozen_gate","fixed_kmeans","fixed_hash"):
        ss=sorted(set(L["learned"])&set(L[b]));x=np.array([L["learned"][s]["test_mae"]-L[b][s]["test_mae"] for s in ss])
        t=stats.ttest_1samp(x,0) if len(x)>1 else None
        o["diff_learned_minus_"+b]=dict(n=len(ss),mean=x.mean(),sd=x.std(ddof=1),p=float(t.pvalue),wins=int((x<0).sum()))
    for a in ("learned","frozen_gate"):
        r=[act(L[a][s]["swap"]) for s in sorted(L[a])];ng=[len(L[a][s]["swap"]["groups"]) for s in sorted(L[a])]
        o["swap_"+a]=dict(ratio_mean=float(np.nanmean(r)),ratio_sd=float(np.nanstd(r,ddof=1)),active=ng,E=L[a][0]["E"] if 0 in L[a] else None)
    o["K"]=L["learned"][min(L["learned"])]["top_k"];o["E"]=L["learned"][min(L["learned"])]["E"]
    out[d]=o
json.dump(out,open("stats_v23.json","w"),indent=1,default=float)
for d,o in out.items():
    print(d,"K",o["K"],"E",o["E"])
    for a in("learned","frozen_gate","fixed_kmeans","fixed_hash"):print("  ",a,o[a]["n"],f'{o[a]["mean"]:.3f}±{o[a]["sd"]:.3f}')
    for k,v in o.items():
        if k.startswith("diff"):print("  ",k,f'n={v["n"]} Δ={v["mean"]:+.3f}±{v["sd"]:.3f} p={v["p"]:.3f} wins={v["wins"]}')
        if k.startswith("swap"):print("  ",k,f'{v["ratio_mean"]:.2f}±{v["ratio_sd"]:.2f} active={v["active"]}')
