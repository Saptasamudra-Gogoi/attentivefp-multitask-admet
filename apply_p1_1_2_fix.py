import sys, shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_p1_1_2.bak")
content = open(path, encoding="utf-8").read()

old = ("**Table 5 | Effect of expert membership on all eight physicochemical "
       "descriptors across four ADMET datasets.** \u03b7\u00b2 effect size "
       "(fraction of descriptor variance explained by expert assignment; "
       "higher = stronger specialization), from one-way ANOVA; all entries "
       "*P* < 0.001 by both ANOVA and Kruskal\u2013Wallis except "
       "Lipophilicity-HBD (\u2020, *P* = 0.33, not significant). Solubility, "
       "AqSolDB (*n* = 9,980); Caco-2, Wang (*n* = 910); LD50, Zhu "
       "(*n* = 7,385); Lipophilicity, AstraZeneca (*n* = 4,200). Three of "
       "the four datasets show effect sizes of 0.03\u20130.445; the "
       "Lipophilicity dataset's effects are statistically significant at "
       "this sample size but small in magnitude (no descriptor above "
       "\u03b7\u00b2 = 0.09) \u2014 no substantial specialization by effect "
       "size, not the absence of a statistically detectable one. Directly "
       "measured diversity metrics for this dataset do not cleanly explain "
       "this pattern (Table S2; see Results and Discussion). Mean column "
       "is the row average across all four datasets.")

new = ("**Table 5 | Effect of expert membership on all eight physicochemical "
       "descriptors across four ADMET datasets.** \u03b7\u00b2 effect size "
       "(fraction of descriptor variance explained by expert assignment; "
       "higher = stronger specialization), from one-way ANOVA. All 32 "
       "ANOVA and Kruskal\u2013Wallis tests were Benjamini\u2013Hochberg "
       "corrected for multiple comparisons (\u03b1 = 0.05); 31 of 32 remain "
       "significant after correction (max adjusted *P* = 0.68), the sole "
       "exception being Lipophilicity-HBD (\u2020, adjusted *P* = 0.68, not "
       "significant either before or after correction). Solubility, "
       "AqSolDB (*n* = 9,980); Caco-2, Wang (*n* = 910); LD50, Zhu "
       "(*n* = 7,385); Lipophilicity, AstraZeneca (*n* = 4,200). Three of "
       "the four datasets show effect sizes of 0.03\u20130.445; the "
       "Lipophilicity dataset's effects are statistically significant at "
       "this sample size but small in magnitude (no descriptor above "
       "\u03b7\u00b2 = 0.09) \u2014 no substantial specialization by effect "
       "size, not the absence of a statistically detectable one. Every "
       "\u03b7\u00b2 value in this table has a bootstrap 95% confidence "
       "interval (2,000 resamples) that excludes zero, and the "
       "less-biased omega-squared estimate is within 0.005 of "
       "\u03b7\u00b2 in every cell (full values in "
       "`p1_stats_results.json`, code repository), so neither small-sample "
       "bias nor multiple-comparison inflation changes any conclusion "
       "drawn from this table. Directly measured diversity metrics for "
       "the Lipophilicity dataset do not cleanly explain its suppressed "
       "specialization (Table S2; see Results and Discussion). Mean "
       "column is the row average across all four datasets.")

count = content.count(old)
print(f"Found {count} occurrence(s)")
assert count == 1, f"Expected 1, found {count}"
content = content.replace(old, new)
open(path, "w", encoding="utf-8").write(content)
print("OK - Table 5 footnote updated")
