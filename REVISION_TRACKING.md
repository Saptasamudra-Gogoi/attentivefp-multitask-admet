\## Issue 6 — Audit specialization statistics

Status: TRACED, UNRESOLVED

Finding: No file in the project currently contains the full 20-dataset x 8-descriptor computation

(160 pairs) cited in Table S3. Checked and ruled out: sss\_results\_final.json (22 datasets but only

1 best-descriptor each), expanded\_descriptor\_results.json (5 datasets, 12 descriptors),

specialization\_results.json (5-6 datasets, 2 descriptors), expert\_specialization\_SUMMARY.json

(5 datasets, 8 descriptors — right shape, wrong scale). The generating script full\_eta2\_table.py

exists and would write full\_eta2\_8x4\_table.csv, but that output file does not exist on disk —

either never run to completion at full scale, or deleted after generation.

Action needed: locate full\_eta2\_table.py's input path, confirm it can run against all 20-22

datasets, rerun it once at full scale, save output under a clear name, git commit immediately.

If it cannot be rerun (missing input data), this must be disclosed to Prof. Li as an unresolved

provenance gap rather than silently patched.

