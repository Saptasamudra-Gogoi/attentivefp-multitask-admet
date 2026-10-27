import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({"font.family":"DejaVu Sans"})
INK="#0b0b0b";SEC="#52514e";LINE="#c3c2b7";BL="#2a78d6";OR="#eb6834";GR="#1baf7a"
fig,ax=plt.subplots(figsize=(10,4.6),dpi=300);ax.set_xlim(0,100);ax.set_ylim(0,50);ax.axis("off")
def box(x,y,w,h,txt,fc="#f6f5f1",ec=LINE,fs=8.5,bold=False,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=1.2",fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal",linespacing=1.35)
def arrow(a,b,c=SEC,ls="-",rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=1.2,color=c,linestyle=ls,connectionstyle=f"arc3,rad={rad}"))
# molecule graph
pts=[(5,24),(9,28),(14,26),(14,20),(9,18),(5,20)]
for i in range(6):
    a,b=pts[i],pts[(i+1)%6];ax.plot([a[0],b[0]],[a[1],b[1]],color=SEC,lw=1.2)
ax.plot([14,19],[26,29],color=SEC,lw=1.2);ax.plot([14,19],[20,17],color=SEC,lw=1.2)
for p in pts+[(19,29),(19,17)]:ax.add_patch(Circle(p,1.1,fc="#ffffff",ec=INK,lw=1))
ax.text(12,11.5,"Molecular graph\n(atoms 9-d features)",ha="center",fontsize=8,color=SEC)
box(25,19,16,10,"GCN backbone\nL layers + BatchNorm\n+ mean pooling",fs=8)
arrow((20,23.5),(25,24)); ax.text(43.5,27.3,"h",ha="center",fontsize=9,color=SEC,style="italic")
# router
box(48,31,13,8,"Router\nlinear gate\ntop-K logits",fc="#e9f1fc",ec=BL,fs=8,tc=INK)
arrow((41,24.5),(48,34.5),rad=-0.1)
# experts
for i,y in enumerate([26,21,16,11]):
    box(66,y,11,4,"" if False else f"Expert {i+1}" if i<3 else "Expert E",fs=7.5)
ax.text(71.5,6.2,"each expert: Linear + ReLU\nE ∈ {4, 8, 16}",ha="center",fontsize=7.5,color=SEC)
arrow((41,22.5),(66,28),c=SEC); arrow((41,22.5),(66,23),c=SEC); arrow((41,22.5),(66,18),c=SEC); arrow((41,22.5),(66,13),c=SEC)
arrow((61,34),(71.5,30.3),c=BL)
ax.text(63,35.2,"top-K weights\n(softmax over selected)",fontsize=7.5,color=BL,ha="left",va="bottom")
# combine + head
box(82,19,8,10,"Σ  w·FFN",fs=8.5,fc="#f6f5f1")
for y in [28,23,18,13]:arrow((77,y),(82,24.2),c=SEC)
box(92,20,7,8,"Head\n→ ŷ",fs=8.5)
arrow((90,24),(92,24))
# no supervision callout
ax.add_patch(FancyBboxPatch((36,41.3),62,4.2,boxstyle="round,pad=0.2,rounding_size=1",fc="#fff4ec",ec=OR,lw=1))
ax.text(67,43.4,"Router input = pooled vector only: no descriptors, scaffold labels or expert supervision",ha="center",va="center",fontsize=7.6,color=INK)
# balance loss and K=1 note
ax.text(50,3.4,"Auxiliary load-balance loss:  L = L_task + 0.01 · E Σ (mean w_i)²",fontsize=8,color=SEC,ha="left")
ax.text(50,0.4,"K = 1: selected weight ≡ 1, so the gate receives no gradient from either loss.",fontsize=8,color=OR,ha="left")
ax.text(1,48.3,"Fig. 1 | Sparse-MoE plug-in between pooling and the prediction head",fontsize=10,color=INK,ha="left",va="center")
fig.savefig("Fig1_architecture.png",bbox_inches="tight",facecolor="white");print("ok")
