@echo off
:: ============================================================
:: MoE-ADMET GitHub Update Script
:: Run this from D:\molprop_project
:: ============================================================

cd /d D:\molprop_project

echo.
echo [1/4] Copying .gitignore into project...
copy /Y "%~dp0.gitignore" ".gitignore"
echo Done.

echo.
echo [2/4] Staging files...

git add .gitignore

:: Core model files
git add generic_moe_module.py
git add moe_attentivefp.py
git add optuna_moe.py
git add optuna_moe_remaining.py
git add phase3_fixed.py
git add baseline_runner_fresh.py

:: Baseline scripts
git add dmpnn_baseline.py
git add attentivefp_baseline.py
git add gnn_baselines.py
git add fingerprint_baselines.py

:: TDC benchmark scripts
git add run_gcn_tdc_baseline.py
git add run_moegcn_tdc_benchmark.py

:: Statistical analysis
git add compute_wilcoxon.py
git add compute_statistics.py
git add statistical_tester.py

:: Expert specialization
git add expert_specialization_stats.py
git add run_expert_specialization.py
git add routing_analyzer.py
git add expert_specialization_caco2_wang.json
git add expert_specialization_caco2_wang.md
git add expert_specialization_solubility_aqsoldb.json
git add expert_specialization_solubility_aqsoldb.md

:: GROVER baseline
git add grover_results.json
git add grover/

:: Results & data
git add results/
git add baselines/
git add experiments/
git add results_gcn_tdc.json
git add results_moegcn_tdc_v2.json
git add param_timing.json

:: Docs
git add REVISED_ACTION_PLAN.md

echo Done staging.

echo.
echo [3/4] Current status:
git status

echo.
echo [4/4] Committing and pushing...
git commit -m "Add Phase 1-3 benchmarks, TDC results, Wilcoxon analysis, expert specialization stats"
git push origin main

echo.
echo ============================================================
echo DONE. Check above for any errors.
echo ============================================================
pause
