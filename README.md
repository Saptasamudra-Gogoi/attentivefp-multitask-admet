# MoE-ADMET: a sparse mixture-of-experts plug-in for molecular property prediction

Code, saved results and analysis scripts for a manuscript on a sparse top-K mixture-of-experts (MoE)
plug-in for graph convolutional networks (GCN), and on what its router does and does not contribute.
The manuscript is not yet peer reviewed. The numbers in the manuscript are authoritative; every number
traces to a script and a result file listed below.

- Manuscript draft: `paper_v2/manuscript_v2.33_DRAFT.md`
- Archive (Zenodo): https://doi.org/10.5281/zenodo.24023691
- Licence: MIT per the manuscript; confirm a `LICENSE` file is present before release

## What was found

- **Regression (MoleculeNet + TDC).** MoE-GCN had lower mean error than plain GCN on ten of twelve
  regression comparisons, but dense baselines of comparable size were not significantly different, so
  the gain is not shown to come from routing.
- **Classification (BBBP, BACE, ClinTox, SIDER, Tox21), ten randomized scaffold splits.** No consistent
  benefit: MoE-GCN was higher on three of five datasets (pooled sign test P = 0.50). The ClinTox gain
  reflects instability of the plain GCN on two splits, not MoE strength.

  | Dataset | GCN | MoE-GCN | Δ (MoE − GCN) | 95% CI |
  |---|---|---|---|---|
  | BACE | 0.860 | 0.802 | −0.058 | −0.106 to −0.009 |
  | BBBP | 0.878 | 0.858 | −0.020 | −0.051 to +0.010 |
  | ClinTox | 0.743 | 0.837 | +0.093 | +0.008 to +0.179 |
  | SIDER | 0.585 | 0.585 | +0.001 | −0.024 to +0.025 |
  | Tox21 | 0.791 | 0.797 | +0.006 | −0.001 to +0.013 |

- **Specialization.** Expert assignments are associated with physicochemical descriptors (aromatic-ring
  count most consistently: largest η² on 16 of 19 TDC datasets with a non-collapsed router), but a
  frozen random gate reproduces most of that organization, and no consistent accuracy benefit from
  training the router was detected.
- **Specialization versus gain.** Across 19 datasets, Spearman ρ between the largest descriptor η² and
  MoE gain was −0.17 (95% CI −0.57 to 0.27).

## Router-collapse rule (used throughout)

A router is *collapsed* on a dataset if no seed has more than two active experts (dominant expert for at
least 10 molecules): PPBR-AZ, hepatocyte clearance, microsome clearance. *Limited routing* means at
least one seed with fewer than three active experts without being collapsed (CYP2C9 substrate).
Primary analyses use the 19 non-collapsed datasets; 21- and 18-dataset versions are reported as
sensitivity analyses. The rule was written down after some outcome tests had been seen.

## Reproducing each table

Run scripts from the directory they live in. Large intermediate files are not all included.

| Item | Scripts | Results |
|---|---|---|
| Table 1 (MoleculeNet regression) | `moegcn_regr.py`, `run_gcn_matched.py`, `extend_seeds.py` | `results_moegcn_regr.json`, `results_gcn_matched.json` |
| Table 1b (classification) | `paper_v2/randomized_scaffold.py`, `paper_v2/run_classif_randsplit.py` | `paper_v2/results_randsplit/*.json`, splits in `paper_v2/splits/`, RF split audits `paper_v2/rf_audit_*.txt` |
| Table 2 (22 TDC datasets) | `run_gcn_tdc_baseline.py`, `run_moegcn_tdc_benchmark.py` | `results_moegcn_tdc_v2.json` |
| Table 3 (dense baselines) | `ablation_routing.py`, `ablation_routing_add_plain.py` | `ablation_routing_results.json` |
| Table 5 (top-K dynamics, frozen gate) | `paper_v2/exp2_topk_dynamics.py`, `paper_v2/exp2b_frozen_gate.py` | `paper_v2/results/exp2_topk_dynamics.json` |
| Table 6 (η² across seeds) | `paper_v2/exp0_eta2_across_seeds.py` | `paper_v2/results/eta2_across_seeds.json` |
| Table 7 (expert profiles) | `paper_v2/exp0b_expert_profiles.py` | `paper_v2/results/profiles_solubility_aqsoldb_seed0.json` |
| Table 8 (routing controls) | `paper_v2/exp1_routing_controls.py`, `paper_v2/stats_v23.py` | `paper_v2/results/exp1_routing_controls.json`, `paper_v2/results/stats_v23.json` |
| Tables S1, S3 | `paper_v2/exp0c_tableS1_S3_all_datasets.py`, `paper_v2/s3_recompute.py` | `paper_v2/results/tableS1_S3.json`, `paper_v2/results/tableS3.csv`, `paper_v2/s3_recompute_output.txt` |
| Specialization vs gain | `paper_v2/gain_corr_recompute.py` | `paper_v2/gain_corr_output.txt` |

`paper_v2/check_cited_files.py` checks that every file the manuscript cites is tracked in git.

## Protocol notes and known limits

- **Regression splits.** ESOL, FreeSolv and Lipophilicity use one deterministic Bemis–Murcko scaffold
  split (largest scaffold groups to training); seeds vary initialization and batch order only, so
  reported standard deviations exclude split variance. The five seeds were produced by running the
  original three-seed scripts and then `extend_seeds.py`, which adds seeds 3 and 4.
- **Classification splits.** Ten randomized scaffold splits (seeds 0–9), saved as JSON indices. Each
  model is tuned once on split 0, so splits 1–9 are not fully independent of tuning. Molecules that
  PyTorch Geometric cannot featurize are excluded before splitting (11 in BBBP, 4 in ClinTox, 8 in
  Tox21). ToxCast and HIV were not rerun.
- **Withdrawn split.** An earlier stratified scaffold split for BBBP and BACE selected scaffolds using
  class labels (Morgan-fingerprint random forest: 0.984 test AUROC on that BACE split versus
  0.881 ± 0.044 over the ten randomized splits). It is withdrawn; `run_gcn_matched_classif.py` and the
  older classification scripts that use it are superseded by `paper_v2/run_classif_randsplit.py`.
- **`fixed_hash` arm.** In the routing-controls code and result files this arm is the seeded random
  assignment described in the manuscript, not a hash of the molecule.
- Many `apply_*.py`, `fix_*.py`, `check_*.py` and similar scripts in the repository root are historical
  editing or debugging scripts and are not needed to reproduce results.

## Data

All datasets are public: MoleculeNet (https://moleculenet.org) and the Therapeutics Data Commons ADMET
benchmark group (https://tdcommons.ai). No proprietary data are used.

## Environment

`environment.yml` records the conda environment used (Python, PyTorch, PyTorch Geometric, RDKit, Optuna).
Experiments were run on a single NVIDIA GTX 1660 Ti (6 GB). The environment file was exported on Windows (win-64) and may need its pins loosened on other platforms.

```
conda env create -f environment.yml
```
