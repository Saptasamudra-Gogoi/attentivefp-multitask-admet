\## Issue 6 — Audit specialization statistics \[RESOLVED]

His words: "The 160 dataset–descriptor pairs in Table S3 are nested within 20 datasets after two

routing-collapse exclusions... state the actual 20-dataset denominator for the 160-pair analysis."



FINDING: Manuscript's "160 pairs / 20 datasets / 2 collapsed" is incorrect. Full 22-dataset

checkpoint-mode run (run\_all\_specialization\_full22.py, git commit <hash>) confirms 3 datasets

collapsed to 1 active expert: ppbr\_az, clearance\_hepatocyte\_az, clearance\_microsome\_az.

Correct figures: 19 datasets, 8 descriptors each = 152 pairs, not 160.

This is independently confirmed by sss\_results\_final.json (also shows 3 collapses, same 3 datasets).

Root cause of manuscript error: ppbr\_az's collapse was not being flagged as "collapsed" in an

earlier version of the pipeline due to a normalization bug in the regression-dataset code path

(fixed data exists, script itself still has this minor labeling bug, low priority).



ACTION TAKEN: Table S3 and all "20 datasets"/"160 pairs" references in manuscript must be

corrected to "19 datasets"/"152 pairs". eta-squared values for Lipophilicity and other reconciled

descriptors should be re-pulled from expert\_specialization\_SUMMARY.json (this run) as the single

source of truth going forward.

