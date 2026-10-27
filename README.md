# Sparse Mixture-of-Experts Routing for Molecular Property Prediction

Code and results for **"What does sparse expert routing contribute in molecular property prediction? Learned, frozen and fixed routers compared."**

## Summary

We add a sparse top-K mixture-of-experts (MoE) layer after GCN pooling and ask what the router actually contributes, using MoleculeNet and the 22-dataset TDC ADMET benchmark under matched scaffold splits. Controls: learned vs frozen random gate (the only K-matched contrast), fixed k-means and fixed random assignment (hard top-1, not K-matched), forced-swap expert tests, and a parameter-matched dense ablation (Dense-uniform, Dense-wide, Plain-GCN).

1. **Directional but fragile regression gain, none on classification.** MoE-GCN beats matched plain GCN on 10/12 regression datasets (exact sign test P = 0.019); 9/11 after removing duplicated Lipophilicity measurements (P = 0.033); 7/9 excluding collapsed-routing runs (P = 0.090). Matched GCN classification shows no consistent benefit (five matched datasets, sign test P = 0.81).
2. **The gain is not attributable to routing.** In the dense ablation (n = 5 datasets), relative-RMSE differences vs Dense-uniform (-0.3%, 95% CI [-2.0, +1.5]) and Dense-wide (-0.3%, [-3.3, +2.7]) are indistinguishable from zero; Plain-GCN is competitive.
3. **Learning the gate gives no demonstrable extra benefit.** At K = 1 the gate receives zero task gradient. At K = 2 on Tox21, aromatic-ring eta-squared is 0.372 (learned) vs 0.286 (frozen random): difference +0.086, 95% CI [-0.015, +0.186] (five seeds), i.e. inconclusive. Specialization strength does not predict gain (Spearman rho = -0.17, 95% CI [-0.60, +0.31]); endpoint-relevant specialization shows rho = 0.43 [0.21, 0.59]. Fixed routing has precedent in language models (Hash Layers; Fan et al.); the contribution here is the molecular setting and descriptor-level analysis.

## Manuscript

Current draft: [`paper_v2/manuscript_v2.30_DRAFT.md`](paper_v2/manuscript_v2.30_DRAFT.md). The earlier `Sapta_MoE-ADMET_manuscript_REVISED.md` is superseded and kept for history only; its headline claims (including "recovers, rather than creates" and rho = 0.019) are **no longer supported**.

## Repository structure

**`paper_v2/`** holds the scripts, saved results and figures for the current manuscript (v2.30); the table below describes the earlier pipeline it builds on.

| Path (`paper_v2/`) | Contents |
|---|---|
| `exp1_routing_controls.py`, `analyze_exp1.py` | Learned/frozen/fixed router controls, forced-swap |
| `exp2_topk_dynamics.py`, `exp2b_frozen_gate.py` | K-sweep gate dynamics, frozen-gate arm |
| `exp0*_*.py` | Eta-squared across seeds, expert profiles, Tables S1-S3 |
| `run_gcn_matched_classif.py` | Matched Optuna (30 trials) + 5 seeds GCN classification |
| `stats_v23.py` | Sign tests, bootstrap CIs, dense-ablation statistics |
| `make_fig*.py` | Figure scripts |
| `results/*.json`, `figures/` | Saved outputs behind the tables and figures |

**Earlier pipeline (repo root):**

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

If you use this code or build on these findings, please cite the manuscript (citation details to be added on publication) and this repository (archived at Zenodo, DOI: `10.5281/zenodo.22746900` — see Releases for versioned snapshots).

## License

MIT. See [LICENSE](LICENSE).

## Contact

Saptasamudra Gogoi, College of Life Sciences, Guizhou University.
Correspondence: Yuquan Li (yvquan.li@gzu.edu.cn), State Key Laboratory of Green Pesticide / College of Computer Science and Technology, Guizhou University.
