# Sparse mixture-of-experts routing recovers, rather than creates, physicochemical organization of chemical space for molecular property prediction

**Authors**

Saptasamudra Gogoi¹, Yuquan Li¹·²·*

¹College of Life Sciences, Guizhou University, Guiyang 550025, China
²State Key Laboratory of Green Pesticide / College of Computer Science and Technology, Guizhou University, Guiyang 550025, China

\* Correspondence: yvquan.li@gzu.edu.cn

---

## Abstract

Structure–property relationships in molecular ADMET behavior are heterogeneous across chemical space, yet graph neural networks apply one shared set of parameters to every molecule. Sparse mixture-of-experts (MoE) routing offers an alternative by directing molecules to specialized subnetworks, and expert assignment has been reported to align with chemical structure, but whether this reflects a property unique to routing has not been tested against a simple clustering baseline. Here we integrate an architecture-agnostic sparse top-K MoE module into graph neural network backbones and evaluate it across ten MoleculeNet datasets and the twenty-two-dataset Therapeutics Data Commons ADMET benchmark under matched scaffold splits. The MoE plug-in improves regression accuracy on ten of twelve datasets, significant when pooled, but a parameter-matched ablation shows no significant difference from equal-capacity dense baselines, indicating routing does not confer a measurable performance advantage. Expert assignment aligns with physicochemical descriptors not explicitly provided to the router, most notably aromatic ring count and lipophilicity, replicating across three datasets and much weaker on a fourth. Two controls temper this: k-means clustering of the same trained representation recovers this organization at least as strongly on 15 of 16 comparisons, and across all 22 TDC datasets, specialization strength does not predict the size of MoE performance gain (Spearman r = 0.019, P = 0.934). Neither the accuracy gain nor the chemical organization is therefore unique to routing: the representation already encodes physicochemical structure, and the router inherits rather than discovers it. Sparse MoE routing recovers, rather than creates, this organization — its value lies not in discovering novel chemical structure, but in providing an endogenous, prediction-coupled partition of a representation that is already chemically organized.

------

## Introduction

Unfavorable absorption, distribution, metabolism, excretion and toxicity (ADMET) properties remain a leading cause of late-stage attrition in drug development, where a single failure can represent a substantial fraction of the reported US$2.6 billion average cost of bringing a drug to market [1]. Reliable computational prediction of ADMET endpoints from molecular structure therefore allows unpromising candidates to be filtered before costly synthesis and assay, and has become a routine component of early-stage virtual screening, with public benchmarks such as the Therapeutics Data Commons (TDC) ADMET suite [2] and MoleculeNet [3] now serving as standard testbeds for such models.

Graph neural networks—graph convolutional networks [4], graph isomorphism networks [5], directed message-passing networks [6], graph-attention models [7], and large pretrained variants [8]—are the dominant approach for learning molecular properties directly from structure. These models share a structural assumption that is rarely examined: a single shared set of parameters maps every input molecule to its predicted property, regardless of where that molecule sits in chemical space. This forces, for instance, a small polar sugar and a large lipophilic steroid through the same weights, even though their solubility is governed by largely disjoint structural features—diluting the signal available to each.

Sparse mixture-of-experts (MoE) architectures, originally developed to scale language models, address exactly this kind of heterogeneity by replacing a single computation path with a collection of expert subnetworks and a lightweight router that activates only a few experts per input [9–11]. Applied to molecules, the appeal is intuitive: chemical space has natural cluster structure—drug-like, lipophilic, polar, fragment-like—and a router could in principle assign each cluster to a dedicated expert learning its specific structure–property mechanism. Prior molecular MoE work [12–15] has begun to explore this idea—via collaborative multi-task experts, BRICS-guided substructure routing, hybrid refinement layers, and multi-view fusion across string and graph encodings—but shares two limitations: chemical knowledge is either imposed on the router by construction rather than discovered [13], or expert behavior is reported qualitatively without a null hypothesis, a parameter-matched control, or cross-dataset replication [12,14,15]. The closest structural analogue is TopExpert [20], where experts likewise align with a molecular axis; there, however, the axis is topology and experts are trained via explicit clustering, whereas our experts arise from end-to-end sparse routing without explicit physicochemical or expert-assignment supervision, and align with physicochemical rather than topological structure.

This leaves two questions unresolved. First, does end-to-end trained MoE routing discover chemically meaningful molecular partitions without hard-coded chemical rules, or does it merely reflect a convenient load-balancing artefact? Second, are any observed predictive gains caused by specialized routing logic, or simply by the additional parameter capacity the expert bank provides? A router that appears to separate "lipophilic" from "polar" molecules on one dataset may be capturing a dataset-specific artefact rather than a transferable regularity, and a performance gain attributed to routing may simply reflect a larger effective model. Without a statistical test that routing tracks physicochemical structure, a controlled comparison against equal-capacity non-routed baselines, and replication across independent datasets, the recurring claim that "experts specialize chemically" cannot be distinguished from coincidence.

Despite growing interest in molecular mixture-of-experts models, the central question is not simply whether sparse routing can improve benchmark performance, but what the routing mechanism actually learns. If expert assignment reflects meaningful chemical organization, sparse routing offers more than an architectural modification—it provides a way to decompose heterogeneous chemical space into distinct predictive regimes. If routing instead reflects optimization dynamics or parameter allocation alone, observed improvements carry limited scientific significance beyond the benchmark itself. Distinguishing between these possibilities requires more than reporting accuracy: it requires testing whether specialization corresponds to physicochemical structure the router never saw, whether the effect survives a parameter-matched control, and whether it reproduces across chemically distinct datasets. We integrate a sparse top-K MoE module as an architecture-agnostic plug-in within standard graph neural network backbones and benchmark it across ten MoleculeNet datasets and the full twenty-two-dataset Therapeutics Data Commons (TDC) ADMET suite under matched scaffold splits, addressing each of these three requirements directly. We report the predictive performance of the plug-in honestly, including the tasks and baselines on which it does not help, and argue that quantified, reproducible expert specialization—decoupled from any performance claim—is what makes this architecture worth studying.

---

## Results

### Sparse routing provides a testable expectation of chemically specialized computation

Sparse mixture-of-experts architectures assume that different molecules benefit from different computational pathways, but whether such specialization actually emerges during molecular property prediction cannot be inferred from the architecture alone. The model treats prediction as a routing problem (Fig. 1). A standard graph neural network compresses a molecule into a single representation vector; that vector is passed to a small bank of expert subnetworks together with a learned router, and the router forwards each molecule to the few experts best suited to it. Only those experts compute, and their outputs are combined into the final prediction.

Crucially, the router receives no explicit physicochemical descriptors, scaffold labels, or expert-assignment supervision; routing is learned end-to-end from the molecular representation and the prediction objective alone, with a mild pressure, through a load-balancing term, not to ignore any expert. Any chemical structure in the resulting assignment must therefore emerge through this end-to-end optimization rather than from direct chemical supervision. If experts acquire chemically distinct roles during training, the resulting partition should be detectable independently of predictive performance—the basis on which we examine routing below, before turning to benchmark accuracy.

Because the module attaches to the pooled representation and leaves the underlying network untouched, the same component can be dropped into different backbones. We use a graph convolutional network as the primary backbone and report its behavior throughout, returning to the other backbones only to test transferability.

### Expert routing improves regression accuracy but not classification

If sparse routing captures meaningful heterogeneity within chemical space, its benefits should appear preferentially on tasks whose underlying structure–property relationships are themselves heterogeneous; regression benchmarks, where chemically distinct molecular classes often contribute through different mechanisms to the same continuous endpoint, provide a direct test of that expectation. On regression tasks, adding the expert plug-in to the graph convolutional backbone improved accuracy consistently (Table 1). Across the three MoleculeNet regression sets and nine TDC regression sets, the MoE model reduced error on ten of twelve datasets, with the largest gains on the chemically diverse solubility and solvation benchmarks: a 19.4% reduction in root-mean-square error on ESOL (1.067 versus 1.324), 27.1% on FreeSolv (3.591 versus 4.927) and 20.5% in mean absolute error on Caco-2 permeability (0.365 versus 0.461; per-dataset-tuned three-seed values, Table 7 — distinct from the fixed-architecture parameter-matched ablation reported in Table 6). Pooled at the dataset level across all twelve regression datasets, MoE-GCN won on ten and lost on two, which by an exact one-sided sign test is statistically significant (*P* = 0.019); a Wilcoxon signed-rank test on the same signed differences gives a consistent result (*P* = 0.017, rank-biserial correlation 0.69) and is reported as a sensitivity analysis, since it ranks the magnitude of RMSE, MAE, and Spearman differences together despite their different scales, whereas the sign test avoids that issue by testing only win/loss counts (Table 2). Both losses occurred on the two smallest regression datasets (half-life and hepatocyte clearance, each under 500 training molecules), consistent with, but insufficient to establish, a small-data limitation (see Limitations).

This regression advantage did not extend to classification, and the boundary between the two is a practically useful takeaway rather than an incidental finding. On the seven MoleculeNet classification datasets the plug-in matched but did not reliably beat the plain backbone, and pooling the classification results (a single metric, AUROC, so directly comparable) yielded no significant difference (*P* = 0.94; full per-dataset TDC results in Table 7). We do not report a single pooled statistic across all 22 TDC datasets, since that would mix classification and regression metrics on the same incommensurate scales the sign test was designed to avoid for the regression-only pool; the per-dataset breakdown in Table 7 is the appropriate way to read the full TDC picture. We report these null results directly rather than selectively: in our benchmarks, the observed gains were concentrated in regression tasks, whereas classification showed no consistent benefit.

Two further comparisons bound the practical value of the gain. Against a stronger attention-based backbone (AttentiveFP), the expert model was outperformed on all three MoleculeNet regression datasets, with a large margin on FreeSolv (3.59 versus 2.57 RMSE; Table 1); attention applied during message passing and routing applied after pooling address complementary parts of the architecture, and our post-pooling plug-in does not subsume the former. Against a model pretrained on ten million molecules (GROVER), the gap was larger still on several tasks.

EC-MPNN — an edge-conditioned message-passing network (NNConv with a GRUCell update); we verified against the source that it does not implement the reverse-bond message exclusion that defines Yang et al.'s original D-MPNN, so we do not claim D-MPNN fidelity — showed the same directional benefit as GCN on two of three datasets: MoE routing improved RMSE on ESOL (1.105 versus 1.163) and FreeSolv (2.865 versus 3.089) but not on Lipophilicity (0.712 versus 0.701). This suggests the plug-in's benefit, while not universal, transfers more consistently across backbones than a single-dataset failure would imply, an open question we return to in the Discussion. Cross-architecture transferability to GIN has not yet been tested and we make no claim about it here.

### The routing mechanism does not itself confer an accuracy advantage

Because a larger model can outperform a smaller one for reasons unrelated to routing, we compared MoE-GCN against two non-routed baselines matched to within 1% of its total parameter count (Table 6): Dense-uniform, in which the same expert subnetworks are retained but averaged with equal, non-learned weights, and Dense-wide, a single wider feed-forward layer of equivalent total width. Across five regression datasets and five seeds each, MoE-GCN showed no significant difference from Dense-uniform (paired *t*-test *P* = 0.58; Wilcoxon *P* = 0.71) or from Dense-wide (paired *t*-test *P* = 0.99; Wilcoxon *P* = 0.79); no configuration dominated across all five datasets.

Sparse routing thus achieves accuracy indistinguishable from equal-capacity dense alternatives. The distinction is not that only routing produces chemically organized structure — the clustering control below shows the dense representation already contains that structure — but that routing provides an endogenous, prediction-coupled assignment: each molecule's expert label is generated by the model itself as part of computing its prediction, whereas a dense baseline has no analogous internal mechanism from which a comparable per-molecule label could be read off without a separate, post-hoc clustering step.

### Expert routing self-organizes through an exploration-to-specialization transition

Expert specialization is not present at initialization; it emerges abruptly during training (Fig. 2). Tracking the fraction of molecules routed to each expert on Tox21 over the course of optimization revealed three regimes (Table 3). Early in training the load was near-uniform across experts, reflecting the random initialization of the router.

Between roughly epochs 15 and 25 this symmetry broke: one expert acquired a growing share of the routing through positive feedback—an expert that handles more molecules receives more gradient signal and improves faster, attracting still more molecules—while another expert's share collapsed toward a few percent. By epoch 30 the assignment had stabilized, with a dominant expert handling about 60% of molecules and the remainder distributed among supporting experts.

This exploration-to-specialization transition mirrors the routing dynamics reported for sparse experts in language models [10] and is consistent with the partition being actively learned rather than fixed by the architecture.

### Expert assignments reflect medicinal-chemistry-relevant physicochemical structure, with a boundary set by chemical diversity

If different experts encode different regions of chemical space, molecules assigned to separate experts should differ systematically in physicochemical properties not explicitly provided to the router during optimization; failure to observe such differences would indicate arbitrary optimization dynamics rather than genuine physicochemical structure. We quantified the relationship between dominant expert assignment and eight independent physicochemical descriptors across four ADMET datasets (Fig. 3; Table 5). On the aqueous solubility dataset (*n* = 9,980 molecules, five active experts), all eight physicochemical descriptors varied significantly across experts by both analysis of variance and the non-parametric Kruskal–Wallis test (*P* < 0.001 in every case). The effect was strongest for lipophilicity: expert membership explained one-third of the total variance in calculated LogP (η² = 0.33), with the next largest effects for aromatic-ring count, polar surface area and molecular weight (η² = 0.10, 0.09 and 0.10). Inspecting the experts directly makes the partition concrete (Table 4): one expert specialized in highly lipophilic, low-polarity, aromatic molecules (mean LogP 4.5, polar surface area 45), a second in hydrophilic, highly polar molecules (mean LogP −0.7, polar surface area 96), a third in small, moderately polar molecules, a fourth in very small fragments (mean molecular weight 89) and a fifth in drug-like molecules of intermediate polarity.

Repeating the analysis on two further independent datasets confirmed generalization. On Caco-2 permeability (*n* = 910), a smaller dataset measuring membrane transport rather than solubility, aromatic-ring count was the dominant axis (η² = 0.325) alongside lipophilicity (η² = 0.15). On acute toxicity (LD50, *n* = 7,385), aromatic-ring count reached η² = 0.445, the strongest single effect observed in this study. Across these three datasets, LogP and aromatic-ring count consistently emerged as the dominant routing axes; averaged across all four datasets in this study, aromatic-ring count showed the larger overall effect (mean η² = 0.24 versus 0.17 for LogP, averaged across all four datasets; full values in Table 5)—medicinal-chemistry-relevant physicochemical axes, particularly lipophilicity and aromaticity, that chemists use to reason about drug-likeness; LogP is one of the four canonical Lipinski Rule-of-Five parameters [16], while aromatic-ring count is a related but distinct heuristic not itself part of that rule.

On a fourth dataset, lipophilicity itself (AstraZeneca, *n* = 4,200), specialization was small in magnitude, though still statistically detectable at this sample size: every descriptor's effect size fell at or below η² = 0.09. The source literature reports this dataset as lower in chemical diversity; we tested that claim directly by measuring two standard diversity metrics (Table S2): Bemis–Murcko scaffold-to-molecule ratio and mean pairwise ECFP4 Tanimoto distance, computed identically across all four datasets. The two metrics disagree for this dataset. By mean pairwise Tanimoto distance, AstraZeneca Lipophilicity is marginally the least diverse of the four (0.881, versus 0.882–0.922 for the others), consistent with the source literature's characterization. By scaffold-to-molecule ratio, however, it is the *most* diverse of the four (0.582, versus 0.206–0.536 for the others)—the opposite of what the low-diversity explanation predicts. Given this disagreement, we do not consider the low-diversity account established, and we report it as an open question rather than a settled mechanism: something about this dataset suppresses specialization, but our diversity metrics do not cleanly explain what. The finding therefore replicates robustly across three datasets and is much weaker on a fourth, for reasons this study does not fully resolve.

---

## Discussion

Two takeaways separate the predictive and interpretive contributions of this study, and both turn out to be qualified rather than unqualified. First, predictive performance receives only qualified support: a significant but modest regression gain, no classification benefit, and no advantage over parameter-matched dense baselines. Second, the interpretive claim is real but narrower than it first appears: expert assignment does align with medicinal-chemistry-relevant physicochemical axes, particularly lipophilicity and aromaticity, statistically significant against eight descriptors and replicating across three datasets measuring unrelated ADMET endpoints, much weaker on a fourth for reasons this study does not fully resolve. Two further controls temper this claim rather than support it further: k-means clustering of the same trained representation recovers this organization at least as strongly, so the partition reflects the underlying representation rather than a property unique to routing, and, separately, the strength of specialization does not predict the size of MoE's performance gain across the full TDC benchmark, showing the interpretive and predictive results are empirically independent rather than two expressions of the same effect. We discuss each in turn below.

### Predictive performance of sparse MoE molecular plug-ins

Sparse mixture-of-experts models are often motivated by the widely hypothesized advantage that heterogeneous chemical space should be represented by heterogeneous computation, but until now the evidence supporting this advantage has remained largely qualitative. Our benchmarking shows the predictive case is real but bounded: MoE improves regression accuracy on ten of twelve datasets, significant when pooled at the dataset level (exact sign test *P* = 0.019), but is outperformed by an attention-based backbone on regression, by a pretrained model on most tasks, and—critically—by two parameter-matched dense baselines with no significant difference in either direction. This result shows the improvement over the smaller plain GCN cannot be uniquely attributed to routing: comparable parameter capacity without a learned routing mechanism is sufficient to match MoE-GCN's accuracy, so additional capacity remains a plausible explanation for the gain over the plain backbone, and reframes what the architecture is actually for. This qualified picture contrasts with Viganò et al. [21], who report MoE outperforming its baselines in a cardiotoxicity multitask setting. Their gain arises from a multitask architecture that shares experts across related endpoints, whereas our ablation isolates routing within a single fixed representation and a single task per model — a materially different comparison, not a contradiction. We do not compare against the Chemprop or TDC leaderboards directly, since our comparison of interest is an ablation within a fixed backbone rather than a cross-method performance ranking.

### Chemical organization is inherited by the router, not created by it

We tested whether the routing partition reflects something unique to the learned routing mechanism using two independent controls, targeting two different questions.

The first control asks whether a simpler method applied to the same trained representation recovers the same organization. We took the pooled representation from each trained MoE-GCN model — the GCN backbone's output immediately before the router, unstandardized — for eight datasets (the four used in the main specialization analysis, plus BBB-Martins, hERG, AMES, and DILI), and clustered it with k-means (`scikit-learn`, `random_state=42`, `n_init=10`), setting the cluster count to the number of experts configured for that dataset. We then recomputed η² for the two descriptors carrying the largest routing effects in the main analysis, LogP and aromatic-ring count, on the resulting k-means clusters. Across these 8 datasets × 2 descriptors (16 comparisons), k-means matched or exceeded the router's η² on 15 of 16. This result points to the representation, not the router: the physicochemical organization we report is recoverable by a generic clustering method applied to the same pooled vector, and is not a distinguishing property of the learned routing mechanism specifically.

The second control asks a different question: does the *degree* of specialization predict the *size* of MoE's predictive benefit? For each of the 22 TDC datasets, we computed a specialization strength score (the largest single-descriptor η² from the main analysis, discounted by the dataset's normalized routing entropy, so that near-uniform routing scores lower) and correlated it against MoE-GCN's percentage performance gain over plain GCN using Spearman's rank correlation. This test was null (r = 0.019, *P* = 0.934): how strongly a dataset's experts specialize chemically does not predict how much accuracy MoE gains on that dataset. This is a separate finding from the k-means control above, not a restatement of it, and it further separates the paper's two claims: the interpretive result (specialization is real and representation-driven) and the predictive result (accuracy gains are modest and not explained by specialization strength) are empirically independent of one another.

Both controls point the same direction from different angles: the physicochemical organization we report is not a distinguishing property of the learned router, and it is not what drives whatever predictive benefit MoE-GCN provides.

This reframes what the earlier sections establish. The pooled representation produced by the graph neural network backbone already encodes physicochemical structure—LogP and aromatic-ring count separate molecules in that space whether or not a router is present—and the router inherits this structure rather than discovering it through its own optimization. This is consistent with the parameter-matched result above: if routing conferred no unique organizational capability and no unique accuracy advantage, that is exactly the pattern a representation-driven rather than routing-driven explanation predicts.

This does not make the routing partition uninteresting, but it changes what claim it supports. Unlike a post-hoc k-means clustering, the router's partition is endogenous to the trained model and directly coupled to the prediction task: each expert's output enters the final prediction, and the partition can be inspected, expert by expert, without a separate clustering step run after the fact. What routing does not provide is evidence that the partition itself is a product of the routing mechanism, or that it captures chemical structure a simpler method would miss. The distinction matters for how this architecture should be positioned: not as a method that discovers chemical organization, but as one that exposes organization already present in the representation, packaged in a form that is directly tied to what the model predicts.

This addresses a gap in prior molecular MoE work, which either builds chemical knowledge into the router by construction, so interpretability is assumed rather than demonstrated [13], or reports expert behavior qualitatively without a null hypothesis, a parameter-matched control, replication, or a clustering baseline [12,14,15]. By testing the routing partition against exactly that baseline, we show that the recurring intuition that experts "specialize chemically" is real at the level of the representation, but is not evidence that routing itself is doing something a simpler unsupervised method could not. Taken together, these results reframe what a molecular MoE model is for: not a route to discovering new chemical organization, but a route to an interpretable, task-coupled view of organization the backbone representation already contains.

### Limitations of the present study

Several limitations bound these conclusions. First, the plug-in does not deliver state-of-the-art accuracy, reflecting where it operates—after pooling rather than inside message passing—and that it trains from scratch rather than leveraging pretraining; inserting the same routing into an attention-based or pretrained backbone is a natural extension. Second, the accuracy benefit is confined to regression; both observed regression losses occurred on the two smallest datasets in our regression set (half-life and hepatocyte clearance, each under 500 training molecules), which is suggestive but not sufficient evidence to establish a general small-data boundary from only two instances, and we observed one expert collapsing to a few percent of the load on a small classification set. Third, cross-architecture transferability is itself architecture-dependent: EC-MPNN gained on ESOL and FreeSolv but lost on Lipophilicity, a partial rather than consistent match to the gains observed with GCN. GIN was excluded from all interpretability analysis in this study and its transferability remains untested. GIN's strictly greater graph-isomorphism discriminative power may alter routing partitioning patterns relative to GCN, so this exclusion should not be read as an implicit negative result; we make no claim about GIN either way. Fourth, and most importantly, the parameter-matched ablation shows the improvement over plain GCN cannot be uniquely attributed to routing — comparable capacity without learned routing is sufficient to match it — but our evidence that routing is chemically meaningful remains correlational: expert assignment tracks physicochemical descriptors, but we have not intervened directly on the routing decision itself. Three concrete follow-ups would strengthen the causal case: (i) forced routing ablation, in which identical molecules are explicitly assigned to alternative experts and the resulting shift in predicted ADMET values and latent representations is quantified; (ii) router feature occlusion, zeroing physicochemically relevant latent dimensions to test whether aromatic-ring or LogP partitioning collapses; and (iii) controlled synthetic compound libraries, curating matched molecule pairs differing only in aromatic-ring count or LogP to isolate single physicochemical axes. None of these was performed here, and we flag them explicitly as the most important remaining work rather than as completed analysis. Finally, our replication spans four datasets chosen to span a range of chemical diversity by design, though our direct measurement of that diversity (Table S2) did not cleanly separate the fourth dataset from the other three; extending this to more datasets with independently characterized diversity would better map the boundary conditions under which the recovered axes emerge.

### Recommendations for future molecular MoE benchmarking

On crowded molecular benchmarks, MoE plug-ins yield modest and uneven accuracy gains that are easily matched by stronger backbones, pretraining, or parameter-matched dense alternatives, and a contribution resting on leaderboard position is fragile. The more durable contribution is the structure of the learned routing itself, but establishing that durability requires two distinct kinds of evidence, not one: whether the plug-in's predictive gain is routing-specific is a question for a parameter-matched dense control, while whether the resulting chemical organization is routing-specific is a separate question for a clustering or random-partition null — a dense model has no routing assignment to test in the first place, so it cannot serve as the control for the interpretability claim. Both controls should be run and reported, and neither substitutes for the other. We recommend four concrete directions: integrating MoE routing inside the graph neural network message-passing layers rather than only after pooling; combining the plug-in with self-supervised molecular pretraining in the style of GROVER; testing routing generalization on external, proprietary compound libraries to validate virtual-screening enrichment; and pursuing the causal intervention experiments outlined above to confirm that physicochemical partitioning is not an optimization artefact. More broadly, we propose a community evaluation workflow for molecular MoE architectures with five components: (1) a parameter-matched dense ablation to isolate routing-specific predictive effects from raw capacity; (2) a clustering or random-partition null (e.g. k-means on the same representation) to isolate routing-specific organizational effects from structure already present in the backbone; (3) cross-dataset replication spanning high- and low-chemical-diversity libraries; (4) η² quantification of expert alignment against standard physicochemical descriptors, reported in full rather than selectively; and (5) transparent reporting of null results on classification and small-dataset regimes rather than selective reporting of favorable benchmarks alone.

---

## Conclusions

Three findings summarize this work. First, on predictive performance: a sparse mixture-of-experts module improves regression accuracy on chemically diverse ADMET benchmarks, significantly so when pooled across datasets, but offers no classification benefit, does not surpass stronger backbones or pretrained models, and does not outperform parameter-matched dense baselines. Second, on mechanism: expert assignment aligns with physicochemical structure—principally aromatic-ring count and LogP—reproducibly across three of four ADMET datasets and much weaker on the fourth for reasons not fully explained by directly measured chemical diversity. Two controls temper this finding: a k-means clustering of the same trained representation recovers this organization at least as strongly, indicating the router inherits rather than creates this structure, and separately, the strength of specialization does not predict the size of MoE's performance gain across the full TDC benchmark, indicating the interpretive and predictive results are empirically independent. Third, on impact: because routing confers no unique accuracy advantage over parameter-matched dense alternatives and no unique organizational advantage over simple clustering of the same representation, we argue that the value of MoE architectures for molecular property prediction lies not in discovering novel chemical organization, but in exposing organization the backbone representation already contains through a partition that is endogenous to the model and directly coupled to its predictions.

---

## Methods

### Molecular representation

Each molecule is represented as an undirected graph in which nodes are atoms and edges are bonds. Atoms carry a nine-dimensional feature vector (atomic number, chirality, degree, formal charge, hydrogen count, radical electron count, hybridization, aromaticity and ring membership) and bonds a three-dimensional feature vector (bond type, stereochemistry and conjugation). Features were generated with RDKit [19] and cast to single precision at load time. The same featurization was used for every model and dataset.

### Backbone architectures

GCN [4] is the primary backbone, on which all interpretability analyses and the parameter-matched ablation were performed: each layer applies the symmetric-normalized propagation **H**⁽ˡ⁺¹⁾ = σ(**D̂**⁻¹ᐟ²**Â** **D̂**⁻¹ᐟ²**H**⁽ˡ⁾**W**⁽ˡ⁾) with **Â** = **A** + **I**, followed by batch normalization, ReLU and dropout, with a global mean pooling read-out producing the molecular representation. As a secondary cross-architecture transferability test, we additionally used an edge-conditioned message-passing network (EC-MPNN) built from an NNConv operator with a GRUCell update rather than a reduction to standard convolution [6]; we inspected the implementation directly and confirmed it does not carry the reverse-bond message exclusion that defines Yang et al.'s original D-MPNN, so we do not claim fidelity to that architecture and refer to it as EC-MPNN throughout rather than D-MPNN or DMPNN. It was benchmarked against the plug-in on three MoleculeNet regression datasets. A graph isomorphism network (GIN) [5] with sum aggregation was implemented and is available in the accompanying repository, but was not included in any reported benchmark comparison or transferability test in this study; we make no claim about GIN transferability in either direction. All primary results and mechanistic conclusions in this study originate from the MoE-GCN configuration; the EC-MPNN results are presented only as a secondary robustness check. GCN, EC-MPNN, and GIN were all implemented in PyTorch Geometric.

### Sparse mixture-of-experts module

The MoE module is inserted between the pooled molecular representation **h** ∈ ℝᴰ and the task prediction head, leaving the backbone unchanged. A linear gating network scores the *E* experts, and only the top-*K* are activated:

**g**(**h**) = TopK(**W**g**h** + **b**g, *K*),  **W**g ∈ ℝ^{*E*×*D*}

w_i = softmax(g_i(**h**)) · 𝟙[i ∈ TopK],  Σ_i w_i = 1

y = Σ_{i ∈ TopK} w_i · FFNᵢ(**h**)

where each expert FFNᵢ is a two-layer perceptron with ReLU activation mapping ℝᴰ → ℝᴰ. Non-selected logits are set to −∞ before the softmax so that exactly *K* experts contribute to each forward pass. To prevent the router from collapsing onto a single expert, we added a load-balancing auxiliary loss

L_bal = *E* · Σ_{i=1}^{*E*} ( mean_{x ∈ B} w_i(x) )²

and optimized the total objective L = L_task + λ L_bal with λ = 0.01, where L_task is masked binary cross-entropy (classification, with missing labels excluded) or mean squared error (regression). The number of experts *E* ∈ {4, 8, 16} and the sparsity *K* ∈ [1, min(4, *E*)] were treated as hyperparameters.

All datasets were split by Bemis–Murcko scaffold [17]: molecules were first clustered by Bemis–Murcko core scaffold, and full scaffold clusters—not individual molecules—were then randomly assigned to train/validation/test in an 80/10/10 ratio, eliminating scaffold leakage across splits. A stratified scaffold split was used for the two smallest, most imbalanced classification sets (BBBP and BACE) to preserve class ratios across splits and avoid single-class validation folds. Hyperparameters were tuned per dataset with the Optuna tree-structured Parzen estimator [18] over 30 trials with median pruning. Models were trained with Adam and weight decay, a learning-rate schedule that halved the rate on validation plateaus, early stopping with patience 15, a maximum of 100 epochs and batch size 64. MoE-GCN is reported as the mean and standard deviation over five random seeds on MoleculeNet and three seeds on TDC. GCN's reporting differs by benchmark: on MoleculeNet, GCN is a single scaffold-split run without a repeated-seed estimate (see Table 1); on TDC, GCN is averaged over three seeds, matching the source result files (see Table 7). We use this convention as-is throughout, including in the pooled statistical test (see Statistical evaluation), rather than fabricating a repeated-seed GCN estimate on MoleculeNet that was not run. All experiments ran on a single NVIDIA GTX 1660 Ti (6 GB).

### Parameter-matched ablation

To determine whether performance gains arise from the routing mechanism itself rather than from the additional parameters the expert bank introduces, we compared MoE-GCN against two equal-parameter, non-routed baselines matched to within 1% of MoE-GCN's total parameter count: Dense-uniform, in which the same expert subnetworks are retained but averaged with equal, non-learned weights (removing the routing signal while preserving expert capacity), and Dense-wide, a single wider feed-forward layer of equivalent total width (removing the multi-expert structure entirely). All three configurations shared identical GCN backbones, training protocol and hyperparameter search budget. We evaluated all three on five regression datasets (ESOL, FreeSolv, Lipophilicity, Caco-2, AqSolDB-Solubility) over five seeds each. For each dataset, performance was averaged across the five seeds, yielding five dataset-level paired observations for each comparison; paired *t*-tests and Wilcoxon signed-rank tests were then applied to these dataset-level differences.

### Quantifying expert chemical specialization

To test whether routing is chemically meaningful, we extracted the dominant expert (the top-1 routed expert) for every molecule in a dataset by a forward pass with routing weights exposed, and related this categorical expert label to eight RDKit physicochemical descriptors: molecular weight, calculated LogP, hydrogen-bond acceptor and donor counts, topological polar surface area, rotatable-bond count, ring count and aromatic-ring count. For each descriptor we computed (i) the mutual information between expert label and the descriptor (discretized into bins, reported in Supplementary Table S1); (ii) a one-way analysis of variance (ANOVA) across expert groups; (iii) a non-parametric Kruskal–Wallis test; and (iv) the η² effect size, the fraction of descriptor variance explained by expert membership — (ii)–(iv) reported in Table 5. This analysis was performed independently on four datasets spanning distinct ADMET endpoints: aqueous solubility (AqSolDB, *n* = 9,980), Caco-2 permeability (Wang, *n* = 910), acute toxicity (LD50, Zhu, *n* = 7,385) and lipophilicity (AstraZeneca, *n* = 4,200). One of these four (AstraZeneca Lipophilicity) was selected as a putative low-diversity comparator based on the source literature; we measured chemical diversity directly with two standard metrics (Bemis–Murcko scaffold-to-molecule ratio and mean pairwise ECFP4 Tanimoto distance, Table S2), which disagree on whether this dataset is in fact the least diverse of the four. This design allows us to assess replication and to report, honestly, the limits of what our diversity measurements can explain about where specialization does and does not emerge.

### Representation-level controls: k-means clustering and specialization-versus-gain correlation

We ran two independent controls to test whether the physicochemical organization above is a distinguishing property of the learned router, rather than of the representation it operates on.

**Clustering control.** For eight datasets (the four used in the main specialization analysis above, plus BBB-Martins, hERG, AMES, and DILI, chosen to extend coverage beyond the primary four without retraining new models), we extracted the pooled molecular representation from each dataset's trained MoE-GCN checkpoint — the GCN backbone's output immediately before the router, i.e. the same vector the router itself receives — via a forward pass over all molecules in the dataset. These representations were clustered with k-means (`scikit-learn`, `n_init=10`, `random_state=42`), using raw, unstandardized embedding coordinates, with the number of clusters set equal to the number of experts configured for that dataset. We then recomputed η² for LogP and aromatic-ring count — the two descriptors with the largest routing effects in the main analysis — on the resulting cluster labels, using the same one-way ANOVA formula as above. This gives 8 datasets × 2 descriptors = 16 paired comparisons between the router's η² and the matched k-means run's η² on the identical representation.

**Specialization-versus-gain correlation.** Separately, across all 22 TDC datasets, we computed a specialization strength score per dataset as the largest single-descriptor η² from that dataset's specialization analysis, discounted by the dataset's routing entropy normalized to its maximum possible value (so that near-uniform routing across experts, indicating weak or collapsed specialization, scores lower even when a nominal η² is present); datasets where routing collapsed to a single active expert were scored zero. We correlated this score against MoE-GCN's percentage performance gain over plain GCN (RMSE or MAE reduction for regression datasets, AUROC increase for classification datasets) using Spearman's rank correlation across the 22 datasets. This tests a distinct hypothesis from the clustering control above — not whether the router's organization is unique, but whether stronger specialization is associated with larger predictive benefit.

### Statistical evaluation

All significance testing in this study follows a single consolidated protocol, reported here rather than scattered across Results. The primary statistical unit for the pooled regression test is the dataset (*n* = 12: 3 MoleculeNet + 9 TDC), not the individual seed: for each dataset we used the mean MoE-GCN value across seeds and the corresponding GCN value as reported in its source table — a three-seed mean for the nine TDC datasets, and a single scaffold-split run for the three MoleculeNet datasets, which do not have a repeated-seed GCN estimate (see Backbone architectures, above) — giving one paired observation per dataset. Because TDC regression datasets mix metrics with opposite improvement directions (MAE, ↓ better; Spearman correlation, ↑ better), we defined a signed improvement score per dataset — (GCN − MoE-GCN) for MAE-scored datasets and (MoE-GCN − GCN) for Spearman-scored datasets — so that a positive value always indicates MoE-GCN improved over GCN. Because these signed scores mix RMSE, MAE, and Spearman differences on incommensurate numerical scales, we do not treat the magnitude of a signed-rank test on the raw values as fully reliable, and instead take an exact one-sided sign (binomial) test on win/loss counts as the primary pooled test, since it depends only on which model won each dataset and not on the relative size of wins measured in different units. We report the Wilcoxon signed-rank test on the same signed scores as a sensitivity analysis, one-sided (alternative: MoE > baseline) with the rank-biserial correlation as effect size. For the parameter-matched ablation, where all five datasets share the same RMSE metric and scale, we report both a paired *t*-test and a Wilcoxon signed-rank test at the dataset level (*n* = 5); the Wilcoxon test is the primary judge of significance there, as the paired *t*-test assumes approximately normal RMSE differences, an assumption we do not verify and which is unlikely to hold with only five datasets. Effect sizes (rank-biserial correlation, η²) are reported alongside every significance test. Wording implying significance is used only where *P* ≤ 0.05.

### Datasets

MoleculeNet [3] supplied seven classification datasets (BBBP, BACE, Tox21, ToxCast, SIDER, ClinTox, HIV) and three regression datasets (ESOL, FreeSolv, Lipophilicity). The TDC ADMET benchmark group [2] supplied 22 datasets spanning all five ADMET categories (13 classification, evaluated by AUROC; 9 regression, evaluated by mean absolute error or Spearman correlation). All datasets were accessed through their standard public releases.

---

## Declarations

**Availability of data and materials.** All datasets are open-access and publicly available: MoleculeNet (https://moleculenet.org) [3] and the Therapeutics Data Commons ADMET benchmark group (https://tdcommons.ai) [2]. Code, hyperparameter Optuna search logs, dataset split files, routing-analysis scripts, and plotting scripts for every table and figure in this manuscript are available at https://github.com/Saptasamudra-Gogoi/attentivefp-multitask-admet (archived at Zenodo, DOI: 10.5281/zenodo.22162186) under an MIT licence; this repository reproduces every result reported here.

**Competing interests.** The authors declare no competing interests.

**Funding.** This research was supported by the National Natural Science Foundation of China (No. 32560689, 32125033, 62162008), the National Key R&D Program of China (2024YFD2001100, 2024YFE0214300), the Central Government Guides Local Science and Technology Development Fund Projects (Qiankehezhongyindi [2023] 001), the Program of Introducing Talents of Discipline to Universities of China (111 Program, D20023), the Natural Science Special Fund of Guizhou University (No. 202409), Guizhou Provincial Science and Technology Projects ([2024]002, CXTD[2023]027), the Guizhou Province Youth Science and Technology Talent Project ([2024]317), and the Guiyang Guian Science and Technology Talent Training Project ([2024]2-15). We thank the Public Big Data Supercomputing Center and State Key Laboratory of Green Pesticide of Guizhou University for providing high-performance computing resources.

**Authors' contributions (CRediT taxonomy).** Conceptualization, Methodology, Software, Investigation, Formal analysis, Visualization, Writing – original draft: S.G. Supervision, Writing – review & editing, Project administration: Y.L. All authors discussed the results and approved the manuscript.

**Acknowledgements.** The authors thank the developers of RDKit, PyTorch Geometric, and the Therapeutics Data Commons for making their tools and datasets publicly available.

---

## Tables

**Table 1 | The expert plug-in improves regression over a plain graph convolutional backbone but trails a stronger attention backbone.** RMSE (↓, lower error is better) for regression and AUROC (↑, higher predictive power is better) for classification on MoleculeNet. MoE-GCN is mean ± s.d. over five Bemis–Murcko scaffold-split seeds; GCN is a single scaffold-split run without a repeated-seed estimate and is reported as a point value. AttentiveFP values are our own five-seed scaffold-split runs, except BBBP and BACE, taken from Xiong et al. [7] under their original random-split protocol. GCN, plain graph convolutional backbone; MoE-GCN, the same backbone with the expert plug-in; AttentiveFP, attention-based backbone; GROVER, model pretrained on 10⁷ molecules (reference upper bound), values taken from the GROVER<sub>large</sub> configuration reported by Rong et al. [8] under their original random-split protocol, not independently reproduced here; HIV was not reported in the original GROVER study and is omitted (—). None of the literature-sourced cells (AttentiveFP-BBBP/BACE, all GROVER cells) carry a standard deviation because none is reported in the source; cross-protocol comparison against our own scaffold-split results should be read with that caveat. Δ is the relative percentage RMSE reduction of MoE-GCN relative to GCN (regression rows) or the relative percentage AUROC change (classification rows); e.g. BBBP 0.749→0.762 is +1.7% relative, not +1.30 percentage points.

| Dataset | Metric | GCN | MoE-GCN | AttentiveFP | GROVER | Δ vs GCN |
|---|---|---|---|---|---|---|
| ESOL | RMSE ↓ | 1.324 | 1.067 ± 0.046 | **1.061** | 0.831 | +19.4% |
| FreeSolv | RMSE ↓ | 4.927 | 3.591 ± 0.527 | **2.573** | 1.544 | +27.1% |
| Lipophilicity | RMSE ↓ | 0.812 | 0.722 ± 0.008 | **0.685** | 0.561 | +11.1% |
| BBBP | AUROC ↑ | 0.749 | 0.762 ± 0.031 | **0.908** | 0.940 | +1.7% |
| BACE | AUROC ↑ | **0.955** | 0.893 ± 0.051 | 0.852 | 0.894 | −6.5% |
| Tox21 | AUROC ↑ | 0.727 | 0.745 ± 0.005 | **0.746** | 0.831 | +2.5% |
| ToxCast | AUROC ↑ | 0.633 | 0.647 ± 0.007 | **0.675** | 0.737 | +2.2% |
| SIDER | AUROC ↑ | 0.580 | 0.569 ± 0.005 | **0.594** | 0.658 | −1.9% |
| ClinTox | AUROC ↑ | **0.835** | 0.815 ± 0.018 | 0.729 | 0.944 | −2.4% |
| HIV | AUROC ↑ | **0.740** | 0.736 ± 0.007 | 0.715 | — | −0.5% |

*Bold = best non-pretrained model on that dataset. GROVER shown for reference only and excluded from "best" marking.*

**Table 2 | Pooled statistical test of the regression improvement.** Primary test: exact one-sided sign (binomial) test on win/loss counts across 12 dataset-level paired observations (one signed improvement score per dataset, mean-of-seeds; MAE-scored datasets signed as GCN−MoE-GCN, Spearman-scored datasets as MoE-GCN−GCN, so a positive value always favors MoE-GCN — see Methods), MoleculeNet and TDC regression datasets combined. The sign test asks only which model won on each dataset, avoiding the scale-heterogeneity problem of ranking raw RMSE, MAE, and Spearman differences together; the Wilcoxon signed-rank test is reported as a sensitivity analysis using the same signed differences.

| Quantity | Value |
|---|---|
| Regression datasets pooled | 12 (3 MoleculeNet + 9 TDC) |
| Paired observations (dataset-level) | 12 |
| Wins / losses / ties | 10 / 2 / 0 |
| **Exact sign test, *P* (one-sided, MoE > GCN)** | **0.019** |
| Wilcoxon statistic (sensitivity analysis) | 12.0 |
| Wilcoxon *P* (one-sided, MoE > GCN) | 0.017 |
| Wilcoxon *P* (two-sided) | 0.034 |
| Wilcoxon effect size (rank-biserial) | 0.69 |
| Classification-only *P* (non-significant, AUROC only, 7 MoleculeNet datasets) | 0.94 |

**Table 3 | Routing undergoes an exploration-to-specialization transition during training.** Fraction of molecules routed to each expert on Tox21 (four experts, top-1 routing) at four training checkpoints, from a single representative training run.

| Epoch | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Regime |
|---|---|---|---|---|---|
| 10 | 26.8% | 36.2% | 23.9% | 13.2% | Exploration (near-uniform) |
| 20 | 10.0% | 18.5% | 25.6% | 45.9% | Transition |
| 30 | 3.7% | 15.8% | 20.1% | 60.3% | Specialization |
| 60 | 4.2% | 17.3% | 19.9% | 58.5% | Stable |

**Table 4 | Experts acquire distinct physicochemical profiles without supervision.** Mean descriptor values, computed over molecules assigned to each dominant (top-1 routed) expert—not averaged across all experts a molecule may activate under top-K routing—on the aqueous solubility dataset (*n* = 9,980; five active experts). MW, molecular weight; TPSA, topological polar surface area; HBD, hydrogen-bond donors; ArRings, aromatic rings.

| Expert | n | MW | LogP | TPSA | HBD | ArRings | Chemical character |
|---|---|---|---|---|---|---|---|
| E5 | 3,433 | 313 | 4.47 | 44.9 | 0.72 | 1.59 | Lipophilic / aromatic |
| E1 | 1,904 | 315 | −0.68 | 96.2 | 1.38 | 0.98 | Hydrophilic / polar |
| E7 | 3,500 | 200 | 0.88 | 64.8 | 1.46 | 0.67 | Small, moderately polar |
| E6 | 195 | 89 | 0.30 | 30.8 | 1.12 | 0.04 | Small fragments |
| E3 | 948 | 282 | 2.74 | 56.3 | 0.69 | 1.00 | Drug-like, intermediate |

**Table 5 | Effect of expert membership on all eight physicochemical descriptors across four ADMET datasets.** η² effect size (fraction of descriptor variance explained by expert assignment; higher = stronger specialization), from one-way ANOVA; all entries *P* < 0.001 by both ANOVA and Kruskal–Wallis except Lipophilicity-HBD (†, *P* = 0.33, not significant). Solubility, AqSolDB (*n* = 9,980); Caco-2, Wang (*n* = 910); LD50, Zhu (*n* = 7,385); Lipophilicity, AstraZeneca (*n* = 4,200). Three of the four datasets show effect sizes of 0.03–0.445; the Lipophilicity dataset's effects are statistically significant at this sample size but small in magnitude (no descriptor above η² = 0.09) — no substantial specialization by effect size, not the absence of a statistically detectable one. Directly measured diversity metrics for this dataset do not cleanly explain this pattern (Table S2; see Results and Discussion). Mean column is the row average across all four datasets.

| Descriptor | Solubility η² | Caco-2 η² | LD50 η² | Lipophilicity η² | Mean |
|---|---|---|---|---|---|
| LogP (lipophilicity) | 0.325 | 0.151 | 0.142 | 0.051 | 0.167 |
| Aromatic rings | 0.102 | 0.325 | 0.445 | 0.093 | 0.241 |
| Molecular weight | 0.101 | 0.103 | 0.110 | 0.006 | 0.080 |
| Polar surface area | 0.087 | 0.204 | 0.105 | 0.023 | 0.105 |
| H-bond acceptors | 0.049 | 0.151 | 0.152 | 0.016 | 0.092 |
| H-bond donors | 0.057 | 0.140 | 0.029 | 0.001† | 0.057 |
| Rotatable bonds | 0.042 | 0.121 | 0.169 | 0.019 | 0.088 |
| Ring count | 0.077 | 0.075 | 0.291 | 0.017 | 0.115 |

**Table 6 | Parameter-matched ablation: MoE-GCN versus equal-parameter, non-routed dense baselines.** Root-mean-square error (RMSE, ↓ = lower error is better), mean ± s.d. over five seeds; five regression datasets. The three configurations shown here share one fixed GCN architecture, identical training budget and identical hyperparameter search space *with each other*, and are matched to within 1% of total parameter count, isolating the effect of routing from architecture or capacity differences. This fixed-architecture MoE-GCN is a separate experiment from the per-dataset hyperparameter-tuned MoE-GCN reported in Table 1 and is not directly comparable to it: Table 1 reports the best achievable configuration for each dataset individually, whereas this table holds one architecture constant across all three arms specifically to enable a controlled, parameter-matched comparison. Neither dense baseline is significantly different from MoE-GCN by paired *t*-test or Wilcoxon signed-rank test; the Wilcoxon test is the primary judge given non-normal per-dataset RMSE differences.

| Dataset | MoE-GCN | Dense-uniform | Dense-wide |
|---|---|---|---|
| ESOL | 1.118 ± 0.007 | 1.123 ± 0.033 | 1.108 ± 0.026 |
| FreeSolv | 3.067 ± 0.073 | 3.043 ± 0.091 | 3.087 ± 0.169 |
| Lipophilicity | 0.783 ± 0.007 | 0.764 ± 0.009 | 0.809 ± 0.018 |
| Caco-2 | 0.540 ± 0.008 | 0.547 ± 0.007 | 0.525 ± 0.022 |
| Solubility (AqSolDB) | 1.136 ± 0.019 | 1.137 ± 0.011 | 1.112 ± 0.011 |
| **Paired test vs. MoE-GCN** | — | *t*: *P*=0.58; Wilcoxon: *P*=0.71 | *t*: *P*=0.99; Wilcoxon: *P*=0.79 |

**Table 7 | Per-dataset results across all 22 TDC ADMET datasets, MoE-GCN versus plain GCN.** The dataset-level numbers underlying the pooled statistics in Table 2. Both models reported at matched *n*=3 seeds (MoE-GCN's first three of five available seeds, for direct comparability with GCN). Metric direction: AUROC (↑, higher is better) for classification; MAE (↓, lower is better) or Spearman correlation (↑, higher is better) for regression, as indicated.

| Dataset | Metric | GCN (n=3) | MoE-GCN (n=3) |
|---|---|---|---|
| AMES | AUROC ↑ | 0.841 ± 0.005 | 0.838 ± 0.005 |
| BBB (Martins) | AUROC ↑ | 0.859 ± 0.002 | 0.857 ± 0.016 |
| Bioavailability (Ma) | AUROC ↑ | 0.597 ± 0.008 | 0.628 ± 0.037 |
| Caco-2 (Wang) | MAE ↓ | 0.461 ± 0.078 | 0.365 ± 0.016 |
| Clearance, hepatocyte (AZ) | Spearman ↑ | 0.382 ± 0.001 | 0.335 ± 0.033 |
| Clearance, microsome (AZ) | Spearman ↑ | 0.474 ± 0.042 | 0.548 ± 0.022 |
| CYP2C9 substrate (Carbon-Mangels) | AUROC ↑ | 0.626 ± 0.026 | 0.618 ± 0.007 |
| CYP2C9 (Veith) | AUROC ↑ | 0.875 ± 0.002 | 0.875 ± 0.005 |
| CYP2D6 substrate (Carbon-Mangels) | AUROC ↑ | 0.802 ± 0.010 | 0.777 ± 0.056 |
| CYP2D6 (Veith) | AUROC ↑ | 0.853 ± 0.002 | 0.833 ± 0.010 |
| CYP3A4 substrate (Carbon-Mangels) | AUROC ↑ | 0.590 ± 0.015 | 0.563 ± 0.024 |
| CYP3A4 (Veith) | AUROC ↑ | 0.874 ± 0.004 | 0.889 ± 0.004 |
| DILI | AUROC ↑ | 0.949 ± 0.007 | 0.908 ± 0.006 |
| Half-life (Obach) | Spearman ↑ | 0.312 ± 0.024 | 0.181 ± 0.036 |
| hERG | AUROC ↑ | 0.736 ± 0.019 | 0.670 ± 0.016 |
| HIA (Hou) | AUROC ↑ | 0.935 ± 0.020 | 0.949 ± 0.014 |
| LD50 (Zhu) | MAE ↓ | 0.697 ± 0.008 | 0.653 ± 0.030 |
| Lipophilicity (AstraZeneca) | MAE ↓ | 0.594 ± 0.007 | 0.547 ± 0.002 |
| Pgp inhibition (Broccatelli) | AUROC ↑ | 0.888 ± 0.009 | 0.887 ± 0.008 |
| PPBR (AZ) | MAE ↓ | 9.740 ± 0.340 | 9.459 ± 0.188 |
| Solubility (AqSolDB) | MAE ↓ | 1.048 ± 0.010 | 0.910 ± 0.017 |
| VDss (Lombardo) | Spearman ↑ | 0.384 ± 0.021 | 0.423 ± 0.072 |

---

## References

[1] DiMasi JA, Grabowski HG, Hansen RW. Innovation in the pharmaceutical industry: new estimates of R&D costs. *J Health Econ.* 2016;47:20–33. DOI: 10.1016/j.jhealeco.2016.01.012.

[2] Huang K, Fu T, Gao W, et al. Therapeutics Data Commons: machine learning datasets and tasks for drug discovery and development. *NeurIPS Datasets and Benchmarks.* 2021. arXiv:2102.09548.

[3] Wu Z, Ramsundar B, Feinberg EN, et al. MoleculeNet: a benchmark for molecular machine learning. *Chem Sci.* 2018;9(2):513–530. DOI: 10.1039/C7SC02664A.

[4] Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. *ICLR.* 2017. arXiv:1609.02907.

[5] Xu K, Hu W, Leskovec J, Jegelka S. How powerful are graph neural networks? *ICLR.* 2019. arXiv:1810.00826.

[6] Yang K, Swanson K, Jin W, et al. Analyzing learned molecular representations for property prediction. *J Chem Inf Model.* 2019;59(8):3370–3388. DOI: 10.1021/acs.jcim.9b00237.

[7] Xiong Z, Wang D, Liu X, et al. Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. *J Med Chem.* 2020;63(16):8749–8760. DOI: 10.1021/acs.jmedchem.9b00959.

[8] Rong Y, Bian Y, Xu T, et al. Self-supervised graph transformer on large-scale molecular data. *NeurIPS.* 2020. arXiv:2007.02835.

[9] Shazeer N, Mirhoseini A, Maziarz K, et al. Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. *ICLR.* 2017. arXiv:1701.06538.

[10] Fedus W, Zoph B, Shazeer N. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. *J Mach Learn Res.* 2022;23(120):1–39. arXiv:2101.03961.

[11] Zhou Y, Lei T, Liu H, et al. Mixture-of-experts with expert choice routing. *NeurIPS.* 2022. arXiv:2202.09368.

[12] Yao X, Liang S, Han S, Huang H. Enhancing molecular property prediction via mixture of collaborative experts. *arXiv.* 2023. arXiv:2312.03292.

[13] Jiang T, Wang Z, Yu S, Xuan Q. Adaptive substructure-aware expert model for molecular property prediction (ASE-Mol). *arXiv.* 2025. arXiv:2504.05844.

[14] Sun Y, Lu Y, Li YY, Jing Z, Leung CK, Hu P. MolGraph-xLSTM: a graph-based dual-level xLSTM framework for enhanced molecular representation and interpretability. *Commun Chem.* 2025. DOI: 10.1038/s42004-025-01683-z.

[15] Soares E, Shirasuna V, Brazil EV, Priyadarsini I, Takeda S. Multi-view mixture-of-experts for predicting molecular properties using SMILES, SELFIES, and graph-based representations (MoL-MoE). *Mach Learn Sci Technol.* 2025;6(2):025070. DOI: 10.1088/2632-2153/ade4ef.

[16] Lipinski CA, Lombardo F, Dominy BW, Feeney PJ. Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Adv Drug Deliv Rev.* 2001;46(1–3):3–26. DOI: 10.1016/S0169-409X(00)00129-0.

[17] Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. *J Med Chem.* 1996;39(15):2887–2893. DOI: 10.1021/jm9602928.

[18] Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. *KDD.* 2019:2623–2631. DOI: 10.1145/3292500.3330701.

[19] Landrum G. RDKit: open-source cheminformatics. https://www.rdkit.org.

[20] Kim S, Lee D, Kang S, Lee S, Yu H. Learning topology-specific experts for molecular property prediction. *AAAI.* 2023. arXiv:2302.13693.

[21] Viganò EL, Iwan M, Colombo E, Ballabio D, Roncaglioni A. Mixture of experts for multitask learning in cardiotoxicity assessment. *J Cheminform.* 2025. DOI: 10.1186/s13321-025-01072-7.

[22] Wu J, Wang J, Wu Z, Zhang S, Deng Y, Kang Y, Cao D, Hsieh C-Y, Hou T. ALipSol: an attention-driven mixture-of-experts model for lipophilicity and solubility prediction. *J Chem Inf Model.* 2022;62(23):5975–5987. DOI: 10.1021/acs.jcim.2c01290.

---

## Supplementary material

**Table S1 | Mutual information between dominant expert assignment and each physicochemical descriptor, all four datasets.** MI (bits, discretized descriptor bins) — companion to Table 5's η² values, from the same analysis (Methods). Values sourced directly from `expert_specialization_SUMMARY.json`, not recomputed for this table.

| Descriptor | Solubility MI | Caco-2 MI | LD50 MI | Lipophilicity MI |
|---|---|---|---|---|
| Molecular weight | 0.144 | 0.109 | 0.123 | 0.015 |
| LogP (lipophilicity) | 0.390 | 0.115 | 0.099 | 0.030 |
| H-bond acceptors | 0.071 | 0.124 | 0.135 | 0.013 |
| H-bond donors | 0.042 | 0.111 | 0.025 | 0.005 |
| Polar surface area | 0.089 | 0.200 | 0.106 | 0.020 |
| Rotatable bonds | 0.050 | 0.151 | 0.109 | 0.017 |
| Ring count | 0.129 | 0.113 | 0.234 | 0.016 |
| Aromatic rings | 0.110 | 0.240 | 0.315 | 0.060 |

---

**Table S2 | Measured chemical diversity, all four expert-specialization datasets.** Bemis–Murcko scaffold-to-molecule ratio (higher = more scaffold-diverse) and mean pairwise ECFP4 Tanimoto distance (higher = more structurally diverse; sampled up to 1,000 molecules per dataset), computed identically across datasets. See Results/Discussion for the disagreement between these two metrics on the Lipophilicity dataset.

| Dataset | *n* molecules | Unique scaffolds | Scaffold-to-molecule ratio | Mean pairwise Tanimoto distance |
|---|---|---|---|---|
| Solubility (AqSolDB) | 9,982 | 2,056 | 0.206 | 0.922 |
| Caco-2 (Wang) | 910 | 488 | 0.536 | 0.892 |
| LD50 (Zhu) | 7,385 | 1,677 | 0.227 | 0.919 |
| Lipophilicity (AstraZeneca) | 4,200 | 2,443 | 0.582 | 0.881 |

---

## Figure legends

- **Fig. 1 | Schematic of the architecture-agnostic expert plug-in.** Molecular graph → backbone GNN → pooled vector → learned router → top-K experts → combined prediction. The router receives no explicit physicochemical descriptors, scaffold labels, or expert-assignment supervision; routing is learned end-to-end from the molecular representation and the prediction objective, so any chemical structure in its output arises without direct physicochemical supervision rather than being directly supplied.
- **Fig. 2 | Exploration-to-specialization transition in expert routing.** Fraction of molecules routed to each expert versus epoch, from a single representative training run (Tox21, four experts, top-1 routing; numerical values in Table 3), with exploration, transition, and specialization regimes indicated.
- **Fig. 3 | Quantified expert specialization across four ADMET datasets.** (A) η² heatmap—the proportion of descriptor variance explained by expert assignment—across eight descriptors and four datasets (data: Table 5). (B) Per-expert descriptor profiles on the solubility dataset (data: Table 4).
- **Fig. 4 | EC-MPNN cross-architecture transferability: a partial result.** RMSE for the edge-conditioned MPNN backbone (plain) versus EC-MPNN + MoE on ESOL, FreeSolv, and Lipophilicity; MoE improves ESOL and FreeSolv but not Lipophilicity.
- **Fig. 5 | Regression performance gain over plain GCN.** Per-dataset percentage RMSE improvement, MoE-GCN versus GCN, shown for the three MoleculeNet regression datasets (ESOL, FreeSolv, Lipophilicity); the annotated pooled significance test (exact sign test, *P* = 0.019) covers all 12 regression datasets (MoleculeNet + TDC), not only the three bars shown here (data: Tables 1–2, 7).
- **Fig. 6 | Parameter-matched ablation.** RMSE for MoE-GCN, Dense-uniform, and Dense-wide across five regression datasets, with pooled significance tests annotated (data: Table 6).
