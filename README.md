# Sparse Mixture-of-Experts Routing for Molecular Property Prediction

Code and results for **"Sparse mixture-of-experts routing recovers, rather than creates, physicochemical organization of chemical space for molecular property prediction."**

## Summary

We integrate a sparse top-K mixture-of-experts (MoE) plug-in into graph neural network backbones and evaluate it across 10 MoleculeNet datasets and the 22-dataset TDC ADMET benchmark under matched scaffold splits. Three findings:

1. **Modest, real performance gain on regression, no gain on classification.** MoE-GCN beats plain GCN on 10 of 12 regression datasets (pooled exact sign test, *P* = 0.019), with corrected per-dataset gains of 4.7% (ESOL), 1.7% (FreeSolv, not significant at the per-dataset level), and 3.1% (Lipophilicity) once GCN is given the same hyperparameter search and seed count as MoE-GCN. Classification shows no consistent benefit.

2. **The gain is not uniquely attributable to routing.** A parameter-matched ablation against Dense-uniform, Dense-wide, and a Plain-GCN backbone (all four sharing identical architecture, training protocol, and hyperparameters) shows no configuration significantly outperforms the others — Plain-GCN, with no expert or ensemble machinery at all, wins outright on 2 of 5 datasets.

3. **Expert routing recovers, rather than creates, chemical structure.** Expert assignment aligns with physicochemical descriptors (aromatic ring count, LogP) not explicitly given to the router, replicating across 3 of 4 datasets. Two controls, however, show this organization is not unique to the router: k-means clustering of the same trained representation matches or exceeds the router's effect size on 14 of 16 comparisons, and specialization strength does not predict the size of MoE's performance gain across all 22 TDC datasets (Spearman ρ = 0.019, *P* = 0.934).

## Manuscript

The full manuscript is [`Sapta_MoE-ADMET_manuscript_REVISED.md`](Sapta_MoE-ADMET_manuscript_REVISED.md), including all tables, statistical methods, and supplementary material (S1–S3).

## Repository structure

| Path | Contents |
|---|---|
| `retrain_and_extract.py` | Trains MoE-GCN across all 22 TDC datasets (Optuna HPO, 5 seeds), extracts routing assignments |
| `run_gcn_matched.py` | Matched-protocol GCN baseline (same HPO budget, same seeds as MoE-GCN) — corrects the original single-run comparison |
| `ablation_routing.py` / `ablation_routing_add_plain.py` | Parameter-matched ablation: MoE-GCN vs Dense-uniform vs Dense-wide vs Plain-GCN, fixed hyperparameters, 5 seeds |
| `kmeans_control.py` / `plain_gcn_kmeans_control.py` | Circularity controls: k-means on the trained MoE representation, and on an independently-trained Plain-GCN representation |
| `random_partition_null.py` | Random-partition null test for expert specialization η², all 22 datasets × 5 seeds |
| `lambda_sweep.py` | Load-balancing coefficient sweep (λ ∈ {0, 0.001, 0.01, 0.1, 1.0}) testing whether specialization survives varying regularization strength |
| `endpoint_correlation_vs_eta2.py` | Tests whether endpoint–descriptor correlation predicts suppressed specialization |
| `p1_1_p1_2_combined.py` | Benjamini–Hochberg correction, bootstrap 95% CIs, and ω² for all Table 5 statistics |
| `make_control_figures.py` | Generates the router-vs-k-means and specialization-vs-gain control figures |
| `*.json` (routing/, entropy_results/, checkpoints not included — regenerate via scripts) | Raw analysis outputs backing every table and figure in the manuscript |

## Reproducing the results

```bash
conda env create -f environment.yml  # or see requirements below
conda activate moe_admet

# Train MoE-GCN across all TDC datasets, extract routing
python retrain_and_extract.py

# Matched-protocol GCN baseline
python run_gcn_matched.py

# Parameter-matched ablation (MoE-GCN / Dense-uniform / Dense-wide / Plain-GCN)
python ablation_routing.py
python ablation_routing_add_plain.py

# Circularity controls
python plain_gcn_kmeans_control.py
python random_partition_null.py

# Load-balancing sweep
python lambda_sweep.py

# Statistical corrections and figures
python p1_1_p1_2_combined.py
python make_control_figures.py
```

Core dependencies: `torch`, `torch_geometric`, `rdkit`, `optuna`, `PyTDC`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`.

## Data

- [MoleculeNet](https://moleculenet.org) — ESOL, FreeSolv, Lipophilicity, BBBP, BACE, Tox21, ToxCast, SIDER, ClinTox, HIV
- [Therapeutics Data Commons (TDC) ADMET Benchmark Group](https://tdcommons.ai) — 22 datasets spanning absorption, distribution, metabolism, excretion, and toxicity

Both are fetched automatically by the scripts above; no manual download required.

## Citation

If you use this code or build on these findings, please cite the manuscript (citation details to be added on publication) and this repository (archived at Zenodo, DOI: `10.5281/zenodo.22162186` — see Releases for versioned snapshots).

## License

MIT. See [LICENSE](LICENSE).

## Contact

Saptasamudra Gogoi, College of Life Sciences, Guizhou University.
Correspondence: Yuquan Li (yvquan.li@gzu.edu.cn), State Key Laboratory of Green Pesticide / College of Computer Science and Technology, Guizhou University.
