\# Revision Tracking — MoE-ADMET

Source: 260917\_0900\_意见\_Sapta修订校正\_EN.docx (Prof. Li)



\## Issue 4 — Identify the actual clustering experiment

His words: "Methods instead says that clustering used the MoE-GCN checkpoint and the identical

vector received by its router... Match each comparison to its checkpoint, representation-extraction

script and result file."

Status: NOT STARTED

Finding: —



\## Issue 5 — Reconcile statistical units and seed records

His words: "Statistical evaluation says the MoleculeNet GCN results come from a single run without

repeated-seed estimates, whereas the training paragraph and Table 1 describe five matched seeds."

Status: IN PROGRESS

Finding: baselines/gin\_gcn\_gat/ confirms 5 real seeds (seed0–seed4) per dataset with logs+checkpoints.

Need: locate which script generated the "single run" statistical evaluation text and check if it

matches these 5-seed logs or a separate older run.



\## Issue 6 — Audit specialization statistics

His words: "The 160 dataset–descriptor pairs in Table S3 are nested within 20 datasets after two

routing-collapse exclusions... Lipophilicity LogP η² is 0.051 in Table 5 and 0.019 in Table S3."

Status: IN PROGRESS

Finding: sss\_results\_final.json (confirmed canonical, Aug 6) shows 3 collapsed datasets

(ppbr\_az, clearance\_microsome\_az, clearance\_hepatocyte\_az) → 19-dataset denominator, not 20.

Manuscript's "160 pairs / 20 datasets" is stale — likely predates the third collapse.

Need: confirm descriptors-per-dataset count (8?) to recompute correct pair total (19×8=152?).

Check whether clearance\_hepatocyte\_az collapsed in a later rerun than the other two.



\## Issue 7 — Local inconsistencies

His words: "Table 1 labels AttentiveFP ESOL with †... Table 5 includes η²=0.093 while Results says

every effect is at or below 0.09... Table S2 gives 0.892 vs Results prints 0.882... masked-softmax

equation makes every selected weight equal to one when K=1."

Status: NOT STARTED

Finding: GROVER† column header in final\_results\_table.csv has mojibake (ΓÇá) — needs UTF-8 fix,

may be related to the † provenance marker confusion he's flagging.



\## Files superseded this audit (DO NOT cite in manuscript):

\- final\_results\_table.csv (root copy, May 28) → superseded by attentivefp-multitask-admet copy (Aug 6)

\- ablation\_optuna\_results.json (subfolder copy) → superseded by root copy (Sept 13)

\- ablation\_routing\_results.json (subfolder copy) → superseded by root copy (Sept 13)

\- sss\_results.json, sss\_results\_exact.json → both superseded by sss\_results\_final.json (placeholder data)

