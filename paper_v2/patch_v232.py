"""
patch_v232.py - builds manuscript_v2.32_DRAFT.md from v2.31 using only numbers that trace to files:
  results_randsplit/*.json (--summarize), s3_recompute_output.txt, gain_corr_output.txt, rf_audit_*.txt

Never modifies the source. Every edit is matched by regex and COUNTED; a mismatch is reported in
the FAILED list (and not applied) so you fix it by hand instead of editing the wrong place.

Run from paper_v2:   python patch_v232.py
Then:  git diff --no-index --word-diff manuscript_v2.31_DRAFT.md manuscript_v2.32_DRAFT.md
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "manuscript_v2.31_DRAFT.md"
DST = "manuscript_v2.32_DRAFT.md"
text = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
done, fails = [], []


def asc(s):
    return s.encode("ascii", "replace").decode()


def sub(name, pattern, repl, count=1):
    global text
    n = len(re.findall(pattern, text, re.S))
    if n == 0 or (count is not None and n != count):
        fails.append(f"{name}: matched {n}, expected {count}")
        return
    text = re.sub(pattern, repl, text, flags=re.S)
    done.append(f"{name} ({n})")


def repline(name, prefix, new):
    global text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        fails.append(f"{name}: {len(idx)} lines start with the anchor, expected 1")
        return
    lines[idx[0]] = new
    text = "\n".join(lines)
    done.append(name)


def edit_line(name, prefix, fn):
    global text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        fails.append(f"{name}: {len(idx)} lines start with the anchor, expected 1")
        return
    lines[idx[0]] = fn(lines[idx[0]])
    text = "\n".join(lines)
    done.append(name)


def insert_before(name, prefix, new_block):
    global text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        fails.append(f"{name}: {len(idx)} lines start with the anchor, expected 1")
        return
    lines[idx[0]:idx[0]] = [new_block, ""]
    text = "\n".join(lines)
    done.append(name)


def insert_after(name, prefix, new_line):
    global text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        fails.append(f"{name}: {len(idx)} lines start with the anchor, expected 1")
        return
    lines.insert(idx[0] + 1, new_line)
    text = "\n".join(lines)
    done.append(name)


def delete_line(name, prefix):
    global text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        fails.append(f"{name}: {len(idx)} lines start with the anchor, expected 1")
        return
    del lines[idx[0]]
    text = "\n".join(lines)
    done.append(name)


# ---------------------------------------------------------------- status block
delete_line("status block: recompute bullet", "> - The pooled five-dataset classification statistics")

# ---------------------------------------------------------------- abstract
sub("abstract: classification wording",
    r"classification gains were inconsistent, and an ablation found",
    "classification showed no consistent benefit over ten randomized scaffold splits "
    "(higher mean AUROC on three of five datasets; pooled sign test P = 0.50), and an ablation found")
sub("abstract: 21 -> 19 datasets, rho 0.38",
    r"Across 21 datasets, endpoint-relevant descriptors were specialized more strongly "
    r"\(Spearman (\S+) = 0\.43, 95% CI 0\.21(.)0\.59\)",
    r"Across 19 datasets with non-collapsed routers, endpoint-relevant descriptors were specialized "
    r"more strongly (Spearman \g<1> = 0.38, 95% CI 0.14\g<2>0.56)")
sub("abstract: gain rho",
    r"\((\S+) = ([^\d\s])0\.17, 95% CI ([^\d\s])0\.60 to 0\.31\)",
    r"(\g<1> = \g<2>0.17, 95% CI \g<3>0.57 to 0.27)")
sub("ten -> eight MoleculeNet datasets", r"ten MoleculeNet datasets", "eight MoleculeNet datasets", count=None)
sub("datasets paragraph: classification list",
    r"seven classification datasets \(BBBP, BACE, Tox21, ToxCast, SIDER, ClinTox, HIV\)",
    "five classification datasets (BBBP, BACE, Tox21, SIDER, ClinTox)")

# ---------------------------------------------------------------- regression sign test
sub("regression: sign test sentence",
    r"gives 7 of 9 \(\*P\* = 0\.090; Table 4\)",
    "gives 7 of 9 (*P* = 0.090; Table 4; under the single collapse rule defined in Methods, which also "
    "removes hepatocyte clearance, 7 of 8, *P* = 0.035, a rule written down after both values had been seen)")

# ---------------------------------------------------------------- classification paragraph
L69 = (
    "The regression trend was not mirrored in classification, where MoE-GCN showed no consistent benefit. "
    "On five MoleculeNet datasets (BBBP, BACE, ClinTox, SIDER, Tox21) we retrained both models under one "
    "protocol: hyperparameters were tuned once per model on split 0 and the frozen configuration was "
    "evaluated on ten randomized scaffold splits (Methods), so the two models are paired on identical test "
    "molecules. ToxCast and HIV were not rerun and are not reported. MoE-GCN had a higher mean AUROC on "
    "three datasets (ClinTox +0.093, Tox21 +0.006, SIDER +0.001) and a lower one on two (BACE −0.058, "
    "BBBP −0.020; Table 1). Paired 95% confidence intervals (*t*-based, ten splits, uncorrected) excluded "
    "zero for ClinTox (+0.008 to +0.179) and BACE (−0.106 to −0.009) only. The ClinTox difference reflects "
    "instability of the plain GCN rather than strength of MoE-GCN: GCN scored 0.63 and 0.49 on splits 6 and "
    "7, which supply 61% of the summed difference, whereas MoE-GCN stayed between 0.79 and 0.88 on every "
    "split. On BACE, MoE-GCN was lower on all ten splits (largest gap −0.24 on split 7; excluding it, the "
    "mean difference is −0.037); we did not test why, although 16 experts on about 1,200 training molecules "
    "is a candidate. Pooled over the five datasets the mean difference was +0.004 AUROC (95% CI −0.065 to "
    "+0.073; exact one-sided sign test *P* = 0.50; Wilcoxon *P* = 0.50), and −0.018 excluding ClinTox. With "
    "five datasets the smallest attainable one-sided sign-test *P* is 0.031, so these tests have little "
    "power, and all five comparisons are exploratory; per-dataset TDC classification results are in "
    "Table 2. We do not report a single pooled statistic across all 22 TDC datasets, since that would mix "
    "classification and regression metrics on the same incommensurate scales the sign test was designed to "
    "avoid for the regression-only pool; the per-dataset breakdown in Table 2 is the appropriate way to "
    "read the full TDC picture. We report these null results directly rather than selectively: in our "
    "benchmarks, the observed gains were concentrated in regression tasks, whereas classification showed no "
    "consistent benefit."
)
repline("results: classification paragraph",
        "[v2.31: the BBBP and BACE values below came from the withdrawn stratified scaffold split", L69)

# ---------------------------------------------------------------- Methods: splits
sub("methods: split description",
    r"Except for the dense-baseline ablation described below, all datasets were split by .*?"
    r"eliminating scaffold leakage across splits\.",
    "Except for the dense-baseline ablation described below, splits followed three procedures, all based "
    "on Bemis–Murcko scaffolds [17] with no scaffold shared across training, validation and test sets. "
    "The three MoleculeNet regression datasets (ESOL, FreeSolv, Lipophilicity) used one deterministic "
    "80/10/10 split in which scaffold groups, sorted by size, were assigned largest-first to training, "
    "then validation, then test; the same split was used for every seed and both models, so the reported "
    "standard deviations reflect initialization and batch order only and exclude split variance. The 22 "
    "TDC datasets used the TDC default scaffold split (seed 42). The five MoleculeNet classification "
    "datasets used ten randomized scaffold splits (below).")
sub("methods: randomized split protocol (replaces withdrawn-split bracket)",
    r"\[v2\.31: the stratified scaffold split.*?rerun the same way\.\]",
    "Classification (BBBP, BACE, ClinTox, SIDER, Tox21) used ten randomized Bemis–Murcko scaffold splits "
    "(80/10/10) following the chemprop scaffold-balanced procedure: scaffold groups larger than half the "
    "validation or test size go to training, the remaining groups are shuffled with the split seed (0–9) "
    "and filled greedily with the size limit checked before each group is added "
    "(`randomized_scaffold.py`; indices saved in `paper_v2/splits/`). Molecules that PyTorch Geometric "
    "could not featurize (11 in BBBP, 4 in ClinTox, 8 in Tox21) were excluded before splitting. Each model "
    "was tuned once (Optuna TPE, seed 42, 30 trials, validation AUROC on split 0; same search space, the "
    "plain GCN omitting only expert count and top-*K*) and the frozen configuration was trained on splits "
    "0–9 with one initialization per split (seed = split seed), early stopping on validation AUROC "
    "(patience 15, at most 100 epochs, batch size 64), with test AUROC taken at the best validation epoch "
    "(`run_classif_randsplit.py`). Standard deviations are over splits (n − 1) and are not comparable to "
    "the seed-based values reported for regression. Because validation molecules of split 0 can appear in "
    "other splits' test sets, splits 1–9 are not fully independent of tuning; the test set of split 0 was "
    "not used for tuning. The stratified scaffold split previously used for BBBP and BACE, which "
    "reassigned scaffolds using class labels, was withdrawn: a Morgan-fingerprint random forest scored "
    "0.984 test AUROC on the BACE split versus 0.881 ± 0.044 over the ten randomized splits (BBBP "
    "0.881 ± 0.043; ClinTox 0.820 ± 0.067; `rf_audit_*.txt`).")
sub("methods: classification GCN clause",
    r"for classification, GCN was rerun for BBBP.*?have no GCN value in Table 1;",
    "for classification, both models were retrained under the randomized-split protocol described above "
    "(`run_classif_randsplit.py`; ToxCast and HIV were not rerun and are not reported);")
sub("methods: seed counts",
    r"over five random seeds on MoleculeNet and three seeds on TDC",
    "over five random seeds on MoleculeNet regression, three seeds on TDC, and ten split seeds on "
    "MoleculeNet classification")

# ---------------------------------------------------------------- Table 4
sub("table 4: classification-only P",
    r"(Classification-only \*P\* \(exact one-sided sign test, AUROC, 5 MoleculeNet datasets with matched "
    r"GCN; non-significant\) \| )0\.81",
    r"\g<1>0.50 (3 of 5 datasets; ten randomized scaffold splits)")
insert_after("table 4: second regression sensitivity row",
             "| Sensitivity: also excluding collapsed-routing datasets",
             "| Sensitivity: also excluding hepatocyte clearance under the collapse rule defined in Methods "
             "(7 / 1 of 8; the rule was written down after this value was seen), exact one-sided *P* | 0.035 |")

# ---------------------------------------------------------------- collapse rule and S3 numbers
sub("results: 17 of 21 -> 16 of 19",
    r"Across the 21 TDC ADMET datasets with a non-collapsed router, aromatic-ring count had the largest "
    r"(\S+) of the eight descriptors on 17 \(mean \S+ 0\.25, against 0\.13 for LogP and 0\.16 for ring "
    r"count; Table S3\)",
    r"Across the 19 TDC ADMET datasets with a non-collapsed router, aromatic-ring count had the largest "
    r"\g<1> of the eight descriptors on 16 (mean \g<1> 0.27, against 0.14 for LogP and 0.18 for ring count; "
    r"Table S3; 17 of 21 including the two collapsed clearance sets)")
sub("results: S3 pairs",
    r"Across the 21 TDC ADMET datasets with a non-collapsed router \(168 dataset(.)descriptor pairs",
    r"Across the 19 TDC ADMET datasets with a non-collapsed router (152 dataset\g<1>descriptor pairs")
sub("results: pooled rho 0.38",
    r"pooled Spearman (\S+) = 0\.43; 95% CI 0\.21(.)0\.59 from a bootstrap",
    r"pooled Spearman \g<1> = 0.38; 95% CI 0.14\g<2>0.56 from a bootstrap")
sub("results: within-dataset counts", r"positive in 16 of 21 \(median 0\.33; zero in one\)",
    "positive in 14 of 19 (median 0.33; zero in one)")
sub("results: S3 sensitivity sentence",
    r"Excluding the three datasets with fewer than three active experts in any seed gave (\S+) = 0\.38 "
    r"\(95% CI 0\.16.0\.57\), and dropping aromatic-ring count, ring count or both gave 0\.45.0\.50\.",
    r"Including the two collapsed clearance sets gave \g<1> = 0.43 (95% CI 0.22–0.59; 21 datasets), and "
    r"additionally excluding CYP2C9 substrate (limited routing) gave \g<1> = 0.38 (0.15–0.57; 18 "
    r"datasets); leaving out aromatic-ring count, ring count or both gave \g<1> = 0.41, 0.41 and 0.46 "
    r"(19 datasets, point estimates).")
sub("results: gain correlation, 19 datasets",
    r"For the 21 TDC ADMET datasets with a non-collapsed router we took the largest",
    "For the 19 TDC ADMET datasets with a non-collapsed router we took the largest")
sub("results: gain rho",
    r"was ([^\d\s])0\.17 \(95% CI ([^\d\s])0\.60 to 0\.31, bootstrap over datasets; \*P\* = 0\.48\)",
    r"was \g<1>0.17 (95% CI \g<2>0.57 to 0.27, bootstrap over datasets; *P* = 0.48)")
sub("results: gain rho, mean eta2",
    r"gave ([^\d\s])0\.15 \(([^\d\s])0\.61 to 0\.33\)",
    r"gave \g<1>0.17 (\g<2>0.58 to 0.29; including the two collapsed clearance sets, \g<1>0.14, "
    r"\g<2>0.56 to 0.33, 21 datasets; also excluding CYP2C9 substrate, \g<1>0.22, \g<2>0.61 to 0.25, "
    r"18 datasets)")
sub("results: with 21 datasets", r"cannot be excluded with 21 datasets", "cannot be excluded with 19 datasets")
sub("discussion: 17 of 21",
    r"aromatic rings had the largest (\S+) on 17 of 21 datasets",
    r"aromatic rings had the largest \g<1> on 16 of 19 datasets, or 17 of 21 including the collapsed "
    r"clearance sets")
sub("methods: S3 analysis dataset set",
    r"for the 21 TDC ADMET datasets with a non-collapsed router \(PPBR-AZ excluded because routing "
    r"collapsed to a single expert\)",
    "for the 19 TDC ADMET datasets with a non-collapsed router (PPBR-AZ, hepatocyte clearance and "
    "microsome clearance excluded as collapsed; definition above)")
sub("methods: gain analysis dataset set",
    r"Across the 21 TDC ADMET datasets with a non-collapsed router, we correlated each dataset",
    "Across the 19 TDC ADMET datasets with a non-collapsed router, we correlated each dataset")
insert_before(
    "methods: collapse definition paragraph",
    "As a second candidate mechanism, for the 19 TDC ADMET datasets",
    "A router was called *collapsed* on a dataset if no seed had more than two active experts (dominant "
    "expert for at least 10 molecules): PPBR-AZ, hepatocyte clearance and microsome clearance. A dataset "
    "had *limited routing* if at least one seed had fewer than three active experts but it was not "
    "collapsed (CYP2C9 substrate: 10, 2, 11, 11 and 3 active experts across seeds). The rule was derived "
    "from the active-expert counts, but it was written down after some outcome tests (the regression sign "
    "tests) had been seen, so we report results with and without each exclusion "
    "(`s3_recompute.py`, `gain_corr_recompute.py`). In the released code and result files the fixed random "
    "assignment arm is keyed `fixed_hash`; it is the seeded random assignment described below, not a hash "
    "of the molecule.")
repline("supplement: Table S3 caption", "**Table S3 |",
        "**Table S3 | Endpoint–descriptor correlation versus specialization strength, 19 TDC ADMET datasets "
        "with a non-collapsed router.** For each dataset–descriptor pair, |Pearson correlation(endpoint, "
        "descriptor)| against that descriptor's η² on the dominant-expert assignment, averaged over five "
        "seeds (152 pairs). A router was called collapsed if no seed had more than two active experts "
        "(PPBR-AZ, hepatocyte clearance, microsome clearance); these were excluded. Pooled Spearman ρ = "
        "0.38 (95% CI 0.14–0.56, dataset-level bootstrap, 2,000 resamples; pairs are nested within "
        "datasets, so a pair-level *P* value is not reported). Within-dataset ρ was positive in 14 of 19 "
        "datasets (median 0.33). Sensitivity: including both clearance sets, ρ = 0.43 (0.22–0.59; 21 "
        "datasets, 168 pairs); additionally excluding CYP2C9 substrate (limited routing: at least one seed "
        "with fewer than three active experts), ρ = 0.38 (0.15–0.57; 18 datasets, 144 pairs). Leaving out "
        "aromatic-ring count, ring count or both gave ρ = 0.41, 0.41 and 0.46 (19 datasets, point "
        "estimates). The full per-dataset, per-descriptor table, including mutual information (Table S1), "
        "is in `tableS3.csv` and `tableS1_S3.json` (`exp0c_tableS1_S3_all_datasets.py`); the numbers here "
        "are reproduced by `s3_recompute.py` (output in `s3_recompute_output.txt`).")

# ---------------------------------------------------------------- limitations
sub("limitations: two -> three of five",
    r"two of five matched MoleculeNet comparisons favor MoE-GCN",
    "three of five matched MoleculeNet comparisons favor MoE-GCN, and the ClinTox gain reflects plain-GCN "
    "instability rather than MoE strength")
edit_line("limitations: new caveats", "Several limitations bound these conclusions.",
          lambda l: l.rstrip() + " Eighth, the classification comparison tunes each model once, on the "
          "validation set of split 0, so splits 1–9 are not fully independent of tuning, and its standard "
          "deviations are over splits and not comparable to the seed-based values for regression. Ninth, "
          "the MoleculeNet regression results rest on one deterministic scaffold split, so their standard "
          "deviations exclude split variance, and the router-collapse rule used in sensitivity analyses "
          "was written down after some outcome tests had been seen.")

# ---------------------------------------------------------------- write outputs
open(DST, "w", encoding="utf-8", newline="\n").write(text)

TABLE = """\
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
"""
open("table1_classification_v2.32.md", "w", encoding="utf-8", newline="\n").write(TABLE)

print("APPLIED:")
for d in done:
    print("  ok  ", asc(d))
print("\nFAILED (do these by hand):" if fails else "\nNo failures.")
for f in fails:
    print("  FAIL", asc(f))

print("\nLEFTOVER MARKERS to review by hand (line number: start of line):")
pats = [r"\[v2\.31", r"withdrawn", r"must be recomputed", r"stratified", r"21 TDC", r"17 of 21", r"168",
        r"0\.43", r"ToxCast", r"HIV", r"seven classification", r"zenodo", r"DOI"]
for i, l in enumerate(text.split("\n"), 1):
    hit = [p for p in pats if re.search(p, l)]
    if hit:
        print(f"  {i}: [{', '.join(h.strip(chr(92)) for h in hit)}] {asc(l[:110])}")
print(f"\nWrote {DST} and table1_classification_v2.32.md")
