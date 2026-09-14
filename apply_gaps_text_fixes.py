import shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_gaps_fix.bak")
content = open(path, encoding="utf-8").read()

replacements = [
    # --- P1-3: Tanimoto uninformative-at-this-range caveat ---
    (
        "| Lipophilicity (AstraZeneca) | 4,200 | 2,443 | 0.582 | 0.881 |"
        ,
        "| Lipophilicity (AstraZeneca) | 4,200 | 2,443 | 0.582 | 0.881 |\n\n"
        "Mean pairwise Tanimoto distance spans only 0.881\u20130.922 across all "
        "four datasets \u2014 a four-percentage-point range on a metric that "
        "saturates near its ceiling for typical drug-like libraries and is "
        "correspondingly uninformative for discriminating between these "
        "four datasets specifically; the scaffold-to-molecule ratio, which "
        "spans a much wider 0.206\u20130.582, is the metric that actually "
        "carries discriminative signal here, and is the one on which the "
        "low-diversity account fails (Results, Discussion)."
    ),

    # --- P0-6: Methods, name the load-balancing formulation explicitly ---
    (
        "L_bal = *E* \u00b7 \u03a3_{i=1}^{*E*} ( mean_{x \u2208 B} w_i(x) )\u00b2\n\n"
        "and optimized the total objective L = L_task + \u03bb L_bal with "
        "\u03bb = 0.01"
        ,
        "L_bal = *E* \u00b7 \u03a3_{i=1}^{*E*} ( mean_{x \u2208 B} w_i(x) )\u00b2\n\n"
        "This is the sum of squared mean *gate weights*, not the standard "
        "Switch-Transformer formulation [10], which multiplies the "
        "*dispatch fraction* (the hard fraction of tokens actually routed "
        "to each expert) by the *mean gate probability* for that expert. "
        "Our formulation is minimized at uniform gate weights and is "
        "therefore directionally correct, but it sees only soft gate mass "
        "and is blind to hard top-1 dispatch imbalance \u2014 it can be "
        "small even when one expert receives a disproportionate share of "
        "molecules under top-1 routing, provided the *runner-up* gate "
        "weights are still spread evenly. We optimized the total objective "
        "L = L_task + \u03bb L_bal with \u03bb = 0.01"
    ),

    # --- P0-5a: cross-reference the random-partition null into Table 5's description ---
    (
        "Every \u03b7\u00b2 value in this table has a bootstrap 95% confidence "
        "interval (2,000 resamples) that excludes zero, and the "
        "less-biased omega-squared estimate is within 0.005 of "
        "\u03b7\u00b2 in every cell (full values in "
        "`p1_stats_results.json`, code repository), so neither small-sample "
        "bias nor multiple-comparison inflation changes any conclusion "
        "drawn from this table."
        ,
        "Every \u03b7\u00b2 value in this table has a bootstrap 95% confidence "
        "interval (2,000 resamples) that excludes zero, and the "
        "less-biased omega-squared estimate is within 0.005 of "
        "\u03b7\u00b2 in every cell (full values in "
        "`p1_stats_results.json`, code repository), so neither small-sample "
        "bias nor multiple-comparison inflation changes any conclusion "
        "drawn from this table. Every \u03b7\u00b2 in this table and in "
        "Fig. 3A also clears its dataset-specific random-partition null "
        "ceiling (1,000 same-size-group shuffles, 95th percentile; full "
        "results in `random_partition_null.json`, code repository), for "
        "all four datasets and all seeds tested \u2014 none of the effects "
        "reported here are indistinguishable from a same-size random "
        "partition."
    ),
]

applied, failed = 0, []
for i, (old, new) in enumerate(replacements, 1):
    c = content.count(old)
    if c == 1:
        content = content.replace(old, new)
        applied += 1
        print(f"[OK] #{i} applied")
    else:
        failed.append((i, c))
        print(f"[FAIL] #{i}: found {c} occurrence(s)")

open(path, "w", encoding="utf-8").write(content)
print(f"\n{applied}/{len(replacements)} applied")
