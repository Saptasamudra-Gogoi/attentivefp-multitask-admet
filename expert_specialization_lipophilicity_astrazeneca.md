# Expert Specialization Statistics — lipophilicity_astrazeneca

## Summary
- Significant descriptors (p<0.05): MW, LogP, HBA, TPSA, RotBonds, Rings, ArRings
- High effect size (η²>0.05): LogP, ArRings

## Statistics Table

| Descriptor | MI | F-stat | p-ANOVA | p-KW | η² | Sig |
|------------|-----|--------|---------|------|----|-----|
| MW | 0.015 | 8.2 | 0.0000 | 0.0000 | 0.006 | *** |
| LogP | 0.030 | 74.6 | 0.0000 | 0.0000 | 0.051 | *** |
| HBA | 0.013 | 23.3 | 0.0000 | 0.0000 | 0.016 | *** |
| HBD | 0.005 | 1.1 | 0.3294 | 0.0721 | 0.001 | n.s. |
| TPSA | 0.020 | 33.0 | 0.0000 | 0.0000 | 0.023 | *** |
| RotBonds | 0.017 | 27.3 | 0.0000 | 0.0000 | 0.019 | *** |
| Rings | 0.016 | 24.0 | 0.0000 | 0.0000 | 0.017 | *** |
| ArRings | 0.060 | 143.4 | 0.0000 | 0.0000 | 0.093 | *** |

## LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Expert routing specialization statistics. F-statistics from one-way ANOVA across expert groups; $\eta^2$ = eta-squared effect size; MI = mutual information with discretized descriptor.}
\begin{tabular}{lrrrrrrrr}
\hline
Descriptor & MW & LogP & HBA & HBD & TPSA & RotBonds & Rings & ArRings \\
\hline
Expert 0 & $375.1\pm122.6$ & $2.5\pm1.3$ & $5.1\pm2.2$ & $1.6\pm1.5$ & $90.1\pm45.1$ & $4.8\pm3.1$ & $3.5\pm1.4$ & $1.9\pm1.2$ \\
Expert 1 & $379.5\pm46.7$ & $2.4\pm0.5$ & $8.0\pm1.4$ & $2.0\pm1.3$ & $112.5\pm9.4$ & $5.4\pm0.7$ & $3.5\pm0.7$ & $2.8\pm0.5$ \\
Expert 2 & $386.3\pm104.8$ & $3.4\pm1.3$ & $4.8\pm2.0$ & $1.6\pm1.1$ & $77.2\pm29.3$ & $5.4\pm2.9$ & $3.4\pm1.1$ & $2.7\pm0.9$ \\
Expert 3 & $357.2\pm99.5$ & $3.4\pm1.5$ & $4.4\pm2.1$ & $1.6\pm1.0$ & $75.9\pm30.8$ & $4.0\pm2.5$ & $4.0\pm1.3$ & $3.1\pm1.1$ \\
\hline
F-stat & $8.2$ & $74.6$ & $23.3$ & $1.1$ & $33.0$ & $27.3$ & $24.0$ & $143.4$ \\
p-value & $0.0000$ & $0.0000$ & $0.0000$ & $0.3294$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ \\
$\eta^2$ & $0.006$ & $0.051$ & $0.016$ & $0.001$ & $0.023$ & $0.019$ & $0.017$ & $0.093$ \\
MI & $0.015$ & $0.030$ & $0.013$ & $0.005$ & $0.020$ & $0.017$ & $0.016$ & $0.060$ \\
\hline
\end{tabular}
\end{table}
```


SUGGESTED PAPER TEXT:
─────────────────────
To validate expert chemical specialization quantitatively, we computed mutual
information (MI) between dominant expert assignment and seven RDKit physicochemical
descriptors across all 4200 molecules in the lipophilicity_astrazeneca dataset,
and performed one-way ANOVA across expert groups. Significant between-expert
variation was observed for ArRings, LogP, TPSA (all p < 0.001,
ANOVA), confirming that expert routing captures meaningful physicochemical
structure. Effect sizes (η²) indicate that expert identity explains
9% of ArRings variance, 5% of LogP variance,
consistent with spontaneous learning of Lipinski-like chemical space partitioning.
