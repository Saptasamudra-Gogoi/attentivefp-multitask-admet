"""
apply_p1_3_fix.py
==================
Adds the P1-3 endpoint-correlation-vs-eta2 finding to the manuscript.

Result (from endpoint_correlation_vs_eta2.py, n=160 dataset x descriptor
pairs across 20 active-routing TDC datasets): the hypothesis that a
descriptor driving the endpoint becomes suppressed as a specialization
axis (signal vs nuisance) is REFUTED -- the pooled relationship runs the
OPPOSITE direction (Spearman rho=0.41, P<0.0001): descriptors more
correlated with the endpoint are specialized on MORE, not less.
AstraZeneca Lipophilicity's LogP is a striking exception to this trend
(above-average endpoint correlation, but the single lowest LogP eta2
across all 21 active datasets) -- this sharpens rather than resolves the
open question about why this dataset suppresses specialization.

Adds:
  - A new paragraph in Results (after the existing Tanimoto/scaffold-ratio
    discussion) reporting the pooled test and the AstraZeneca exception.
  - A sentence in Limitations noting this as a second falsified hypothesis.
  - A new Methods subsection describing the test.
  - Table S3 with the pooled result and the specific AstraZeneca numbers.
  - A reference in Datasets/Statistical evaluation is not needed since this
    is self-contained.

Run: python apply_p1_3_fix.py Sapta_MoE-ADMET_manuscript_REVISED.md
"""

import sys, shutil

REPLACEMENTS = [
    # --- 1. Results: append the endpoint-correlation finding after the
    #        existing fourth-dataset paragraph ---
    (
        "Given this disagreement, we do not consider the low-diversity account "
        "established, and we report it as an open question rather than a "
        "settled mechanism: something about this dataset suppresses "
        "specialization, but our diversity metrics do not cleanly explain "
        "what. The finding therefore replicates robustly across three "
        "datasets and is much weaker on a fourth, for reasons this study "
        "does not fully resolve."
        ,
        "Given this disagreement, we do not consider the low-diversity account "
        "established, and we report it as an open question rather than a "
        "settled mechanism: something about this dataset suppresses "
        "specialization, but our diversity metrics do not cleanly explain "
        "what.\n\n"
        "We tested a second candidate mechanism: that a descriptor strongly "
        "correlated with the prediction endpoint becomes task-relevant "
        "signal the network must preserve, rather than a nuisance axis free "
        "for the router to partition on \u2014 predicting a negative "
        "relationship between |correlation(endpoint, descriptor)| and that "
        "descriptor's \u03b7\u00b2. Across all 22 TDC datasets with active "
        "routing (160 dataset-descriptor pairs; Table S3), we found the "
        "opposite: |correlation(endpoint, descriptor)| correlates "
        "*positively* with \u03b7\u00b2 (Spearman \u03c1 = 0.41, *P* < 0.0001) "
        "\u2014 descriptors that drive the endpoint are specialized on more "
        "strongly, not less, refuting this hypothesis at the pooled level. "
        "AstraZeneca Lipophilicity is a striking exception to this pooled "
        "trend: LogP's endpoint correlation there (0.415) sits above the "
        "22-dataset average for LogP (0.244), yet its \u03b7\u00b2 (0.019) is "
        "the lowest LogP effect size observed on any dataset. This sharpens "
        "rather than resolves the open question: suppression on this "
        "dataset is not explained by chemical diversity (Table S2) nor by "
        "endpoint-descriptor correlation (Table S3), and on the latter "
        "measure the dataset runs counter to the pooled relationship rather "
        "than merely failing to fit it. The finding therefore replicates "
        "robustly across three datasets and is much weaker on a fourth, for "
        "reasons this study does not fully resolve."
    ),

    # --- 2. Limitations: note this as a second falsified hypothesis ---
    (
        "Finally, our replication spans four datasets chosen to span a "
        "range of chemical diversity by design, though our direct "
        "measurement of that diversity (Table S2) did not cleanly separate "
        "the fourth dataset from the other three; extending this to more "
        "datasets with independently characterized diversity would better "
        "map the boundary conditions under which the recovered axes emerge."
        ,
        "Finally, our replication spans four datasets chosen to span a "
        "range of chemical diversity by design, though our direct "
        "measurement of that diversity (Table S2) did not cleanly separate "
        "the fourth dataset from the other three, and a second candidate "
        "mechanism \u2014 suppression of specialization on descriptors "
        "correlated with the endpoint \u2014 was refuted at the pooled level "
        "and does not explain this dataset's anomaly either (Table S3); "
        "extending this to more datasets with independently characterized "
        "diversity, and to a direct causal test of what this dataset's "
        "representation is doing differently, would better map the "
        "boundary conditions under which the recovered axes emerge."
    ),

    # --- 3. Methods: add a new subsection describing the test, placed
    #        after the diversity-metrics description ---
    (
        "One of these four (AstraZeneca Lipophilicity) was selected as a "
        "putative low-diversity comparator based on the source literature; "
        "we measured chemical diversity directly with two standard metrics "
        "(Bemis\u2013Murcko scaffold-to-molecule ratio and mean pairwise "
        "ECFP4 Tanimoto distance, Table S2), which disagree on whether this "
        "dataset is in fact the least diverse of the four. This design "
        "allows us to assess replication and to report, honestly, the "
        "limits of what our diversity measurements can explain about where "
        "specialization does and does not emerge."
        ,
        "One of these four (AstraZeneca Lipophilicity) was selected as a "
        "putative low-diversity comparator based on the source literature; "
        "we measured chemical diversity directly with two standard metrics "
        "(Bemis\u2013Murcko scaffold-to-molecule ratio and mean pairwise "
        "ECFP4 Tanimoto distance, Table S2), which disagree on whether this "
        "dataset is in fact the least diverse of the four. This design "
        "allows us to assess replication and to report, honestly, the "
        "limits of what our diversity measurements can explain about where "
        "specialization does and does not emerge.\n\n"
        "As a second candidate mechanism, for all 22 TDC datasets with "
        "active routing (excluding datasets where routing collapsed to a "
        "single expert), we computed the absolute Pearson correlation "
        "between the prediction endpoint and each of the eight "
        "physicochemical descriptors, and correlated this against that "
        "descriptor's \u03b7\u00b2 (from the same dominant-expert assignment "
        "used throughout) using Spearman's rank correlation across all "
        "resulting dataset-descriptor pairs (Table S3). This tests whether "
        "descriptors driving the prediction endpoint are suppressed as "
        "specialization axes, as would be expected if such descriptors "
        "become task-relevant signal rather than a nuisance axis available "
        "for the router to partition on."
    ),

    # --- 4. New Table S3, inserted after Table S2 ---
    (
        "| Dataset | *n* molecules | Unique scaffolds | Scaffold-to-molecule ratio | Mean pairwise Tanimoto distance |\n"
        "|---|---|---|---|---|\n"
        "| Solubility (AqSolDB) | 9,982 | 2,056 | 0.206 | 0.922 |\n"
        "| Caco-2 (Wang) | 910 | 488 | 0.536 | 0.892 |\n"
        "| LD50 (Zhu) | 7,385 | 1,677 | 0.227 | 0.919 |\n"
        "| Lipophilicity (AstraZeneca) | 4,200 | 2,443 | 0.582 | 0.881 |"
        ,
        "| Dataset | *n* molecules | Unique scaffolds | Scaffold-to-molecule ratio | Mean pairwise Tanimoto distance |\n"
        "|---|---|---|---|---|\n"
        "| Solubility (AqSolDB) | 9,982 | 2,056 | 0.206 | 0.922 |\n"
        "| Caco-2 (Wang) | 910 | 488 | 0.536 | 0.892 |\n"
        "| LD50 (Zhu) | 7,385 | 1,677 | 0.227 | 0.919 |\n"
        "| Lipophilicity (AstraZeneca) | 4,200 | 2,443 | 0.582 | 0.881 |\n"
        "\n---\n\n"
        "**Table S3 | Endpoint-descriptor correlation versus specialization "
        "strength, all 22 TDC datasets with active routing.** For each "
        "dataset-descriptor pair, |Pearson correlation(endpoint, "
        "descriptor)| against that descriptor's \u03b7\u00b2 on the router's "
        "dominant-expert assignment (n = 160 pairs across 20 datasets; "
        "PPBR-AZ and Clearance-Microsome-AZ excluded, routing collapsed to "
        "a single expert). Pooled Spearman \u03c1 = 0.41, *P* < 0.0001 "
        "(positive: descriptors correlated with the endpoint are "
        "specialized on more strongly, not less \u2014 the opposite of the "
        "signal-suppression hypothesis this test was designed to check). "
        "AstraZeneca Lipophilicity's LogP row is a marked exception: "
        "|corr| = 0.415 (above the 22-dataset mean of 0.244 for LogP) "
        "paired with \u03b7\u00b2 = 0.019, the lowest LogP effect size "
        "observed on any dataset. Full per-dataset, per-descriptor values "
        "in `endpoint_correlation_analysis.json` in the code repository.\n\n"
        "| Dataset | Descriptor | \\|corr(endpoint, descriptor)\\| | \u03b7\u00b2 |\n"
        "|---|---|---|---|\n"
        "| Lipophilicity (AstraZeneca) | LogP | 0.415 | 0.019 |\n"
        "| Lipophilicity (AstraZeneca) | ArRings | 0.298 | 0.038 |\n"
        "| Lipophilicity (AstraZeneca) | RingCount | 0.280 | 0.076 |\n"
        "| Lipophilicity (AstraZeneca) | MW | 0.159 | 0.098 |\n"
        "| Lipophilicity (AstraZeneca) | TPSA | 0.151 | 0.049 |\n"
        "| Lipophilicity (AstraZeneca) | HBD | 0.196 | 0.000 |\n"
        "| Lipophilicity (AstraZeneca) | HBA | 0.039 | 0.073 |\n"
        "| Lipophilicity (AstraZeneca) | RotBonds | 0.030 | 0.029 |\n"
        "| *All 22 datasets, pooled* | *(all 8 descriptors)* | *Spearman \u03c1 = 0.41* | *P < 0.0001* |"
    ),
]


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_p1_3_fix.py <manuscript.md>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    backup_path = path + ".before_p1_3.bak"
    shutil.copy(path, backup_path)
    print(f"Backup saved -> {backup_path}")

    applied, failed = 0, []
    for i, (old, new) in enumerate(REPLACEMENTS, 1):
        count = content.count(old)
        if count == 0:
            failed.append((i, "NOT FOUND", old[:80]))
            continue
        if count > 1:
            failed.append((i, f"FOUND {count} TIMES (ambiguous)", old[:80]))
            continue
        content = content.replace(old, new)
        applied += 1
        print(f"  [OK] Replacement {i}/{len(REPLACEMENTS)} applied")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n{applied}/{len(REPLACEMENTS)} replacements applied successfully")

    if failed:
        print(f"\n{len(failed)} replacement(s) FAILED -- manual check needed:")
        for i, reason, snippet in failed:
            print(f"  #{i}: {reason}")
            print(f"      \"{snippet}...\"")
    else:
        print("All replacements applied cleanly.")


if __name__ == "__main__":
    main()
