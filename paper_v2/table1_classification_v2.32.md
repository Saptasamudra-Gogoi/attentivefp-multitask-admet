**Table 1b | Classification: MoE-GCN versus plain GCN on ten randomized scaffold splits (replaces the classification rows of Table 1).**
AUROC (mean ± s.d. over ten randomized scaffold splits, split seeds 0–9; both models tuned once on split 0, paired on identical splits). Δ = MoE-GCN − GCN, paired, with *t*-based 95% CI over the ten splits (uncorrected). ToxCast and HIV were not rerun. AttentiveFP and GROVER classification values are omitted because they were obtained under different split protocols.

| Dataset | GCN | MoE-GCN | Δ (MoE − GCN) | 95% CI | Splits won |
|---|---|---|---|---|---|
| BACE | 0.860 ± 0.048 | 0.802 ± 0.086 | −0.058 | −0.106 to −0.009 | 0/10 |
| BBBP | 0.878 ± 0.046 | 0.858 ± 0.050 | −0.020 | −0.051 to +0.010 | 4/10 |
| ClinTox | 0.743 ± 0.110 | 0.837 ± 0.034 | +0.093 | +0.008 to +0.179 | 8/10 |
| SIDER | 0.585 ± 0.018 | 0.585 ± 0.030 | +0.001 | −0.024 to +0.025 | 6/10 |
| Tox21 | 0.791 ± 0.016 | 0.797 ± 0.016 | +0.006 | −0.001 to +0.013 | 7/10 |

Pooled over the five datasets: mean Δ = +0.004 (95% CI −0.065 to +0.073); MoE-GCN higher on 3/5; exact one-sided sign test *P* = 0.50; Wilcoxon *P* = 0.50.
Source: `paper_v2/results_randsplit/*.json`, summarized by `run_classif_randsplit.py --summarize`.
