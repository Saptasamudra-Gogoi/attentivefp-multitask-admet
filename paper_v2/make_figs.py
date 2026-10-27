import json, numpy as np, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#c3c2b7","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","text.color":"#0b0b0b"})
INK="#0b0b0b"; SEC="#52514e"; GRID="#e1e0d9"
# ---------- Fig 3A ----------
E=json.load(open("eta2_across_seeds.json"))
ds=[("solubility_aqsoldb","Solubility\n(AqSolDB)"),("caco2_wang","Caco-2\n(Wang)"),("ld50_zhu","LD50\n(Zhu)"),("lipophilicity_astrazeneca","Lipophilicity\n(AstraZeneca)")]
rows=[("LogP","LogP"),("ArRings","Aromatic rings"),("MW","MW"),("TPSA","Polar surface area"),("HBA","H-bond acceptors"),("HBD","H-bond donors"),("RotBonds","Rotatable bonds"),("RingCount","Ring count")]
M=np.array([[E[d]["eta2"][r]["mean"] for d,_ in ds] for r,_ in rows]); S=np.array([[E[d]["eta2"][r]["sd"] for d,_ in ds] for r,_ in rows])
cmap=LinearSegmentedColormap.from_list("seq",["#f0f6fd","#cde2fb","#86b6ef","#3987e5","#256abf","#184f95","#0d366b"])
fig,ax=plt.subplots(figsize=(5.6,5.0),dpi=300)
im=ax.imshow(M,cmap=cmap,vmin=0,vmax=0.45,aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        c="#ffffff" if M[i,j]>0.24 else INK
        ax.text(j,i-0.12,f"{M[i,j]:.2f}",ha="center",va="center",color=c,fontsize=9.5,fontweight="bold")
        ax.text(j,i+0.22,f"± {S[i,j]:.2f}",ha="center",va="center",color=c,fontsize=7.5)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in ds]); ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _,l in rows])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Fig. 3A | Expert specialization by descriptor\nη² (mean ± s.d. over 5 seeds; routing re-extracted per seed)",fontsize=9.5,loc="left")
cb=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.03); cb.set_label("η² (mean over seeds)"); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig("Fig3A_eta2_mean_sd.png"); plt.close(fig)
# ---------- Fig 5 ----------
D=json.load(open("exp1_routing_controls.json")); R=collections.defaultdict(dict)
for k,v in D.items():
    d,s,a=k.split("|"); R[d].setdefault(a,{})[int(s[4:])]=v
ARMS=[("learned","Learned","#2a78d6","o"),("frozen_gate","Frozen","#eb6834","s"),("fixed_kmeans","k-means","#1baf7a","^"),("fixed_hash","Random","#eda100","D")]
order=[("caco2_wang","Caco-2"),("solubility_aqsoldb","Solubility"),("ld50_zhu","LD50"),("lipophilicity_astrazeneca","Lipophilicity")]
fig=plt.figure(figsize=(9.4,6.4),dpi=300); gs=fig.add_gridspec(2,4,height_ratios=[1.35,1],hspace=0.55,wspace=0.42)
rng=np.random.default_rng(0)
for j,(d,name) in enumerate(order):
    ax=fig.add_subplot(gs[0,j]); K=R[d]["learned"][min(R[d]["learned"])]["top_k"]; Ee=R[d]["learned"][min(R[d]["learned"])]["E"]; n=len(R[d]["learned"])
    for i,(a,lab,col,mk) in enumerate(ARMS):
        v=np.array([R[d][a][s]["test_mae"] for s in sorted(R[d][a])]); x=i+rng.uniform(-0.13,0.13,len(v))
        ax.scatter(x,v,s=22,color=col,marker=mk,edgecolor="#fcfcfb",linewidth=0.8,zorder=3)
        ax.hlines(v.mean(),i-0.3,i+0.3,color=INK,linewidth=1.6,zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l,_,_ in ARMS],fontsize=7.5,rotation=35,ha='right'); ax.set_xlim(-0.6,3.6)
    ax.grid(axis="y",color=GRID,linewidth=0.6,zorder=0); ax.set_axisbelow(True)
    ax.set_title(f"{name}\nK={K}, E={Ee}, {n} seeds",fontsize=8.5,loc="left"); 
    if j==0: ax.set_ylabel("Test MAE (lower is better)")
ax=fig.add_subplot(gs[1,:])
def active(sw):
    Mx=np.array([[np.nan if x is None else x for x in r] for r in sw["matrix"]],float); g=sw["groups"]
    if len(g)<2: return np.nan
    sub=Mx[np.ix_(g,g)]; dg=np.diag(sub); off=np.array([np.mean(np.delete(sub[i],i)) for i in range(len(g))]); return off.mean()/dg.mean()
w=0.32
for jj,(d,name) in enumerate(order):
    for kk,(a,lab,col,mk) in enumerate(ARMS[:2]):
        v=np.array([active(R[d][a][s]["swap"]) for s in sorted(R[d][a])]); x=jj+(kk-0.5)*w*1.3+rng.uniform(-0.05,0.05,len(v))
        ax.scatter(x,v,s=22,color=col,marker=mk,edgecolor="#fcfcfb",linewidth=0.8,zorder=3)
        ax.hlines(v.mean(),jj+(kk-0.5)*w*1.3-0.14,jj+(kk-0.5)*w*1.3+0.14,color=INK,linewidth=1.6,zorder=4)
ax.axhline(1.0,color=SEC,linewidth=0.9,linestyle=(0,(4,3)),zorder=1); ax.text(3.62,1.0,"1.0 = experts\ninterchangeable",fontsize=7,color=SEC,va="center")
ax.set_xticks(range(4)); ax.set_xticklabels([n for _,n in order]); ax.set_xlim(-0.6,4.25); ax.set_ylabel("Forced-swap error ratio\n(other active experts / own expert)")
ax.grid(axis="y",color=GRID,linewidth=0.6,zorder=0); ax.set_axisbelow(True)
ax.scatter([],[],color=ARMS[0][2],marker="o",label="Learned router"); ax.scatter([],[],color=ARMS[1][2],marker="s",label="Frozen random gate"); ax.legend(frameon=False,loc="upper left",fontsize=8,ncol=2)
fig.suptitle("Fig. 5 | Routing controls with a shared backbone and training protocol (dots = seeds, bars = means)",x=0.01,ha="left",fontsize=10)
fig.text(0.01,0.935,"Top: test MAE per arm. Frozen = random gate never updated; k-means = fixed partition on 8 descriptors; Random = fixed random assignment.\nBottom: forced-swap ratio restricted to active experts (≥10 test molecules).",fontsize=7.5,color=SEC,ha="left",va="center")
fig.subplots_adjust(top=0.84,bottom=0.09,left=0.07,right=0.97); fig.savefig("Fig5_routing_controls.png"); plt.close(fig)
print("ok")
