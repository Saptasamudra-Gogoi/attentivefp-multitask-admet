"""
CHEMPROP BASELINE COMPARISON GUIDE
===================================

This guide helps you run D-MPNN (Chemprop) baselines to compare against your MoE-AttentiveFP model.

PREREQUISITES:
--------------
1. Install Chemprop in your molprop environment:
   conda activate molprop
   pip install chemprop --break-system-packages

2. Prepare datasets in correct format (see inspect_datasets.py output)

STEP 1: Single Dataset Training
--------------------------------
"""

import subprocess
import os
from pathlib import Path

# Configuration
DATA_DIR = r"D:\molprop_project\temp\chemprop_data"  # Change this if needed
OUTPUT_DIR = r"D:\molprop_project\baselines\chemprop"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    # Regression tasks
    'ESOL': {'task_type': 'regression', 'metric': 'rmse', 'target_columns': ['measured log solubility in mols per litre']},
    'FreeSolv': {'task_type': 'regression', 'metric': 'rmse', 'target_columns': ['expt']},
    'Lipo': {'task_type': 'regression', 'metric': 'rmse', 'target_columns': ['exp']},
    
    # Classification tasks
    'BACE': {'task_type': 'classification', 'metric': 'auc', 'target_columns': ['Class']},
    'BBBP': {'task_type': 'classification', 'metric': 'auc', 'target_columns': ['p_np']},
    'HIV': {'task_type': 'classification', 'metric': 'auc', 'target_columns': ['HIV_active']},
    'ClinTox': {'task_type': 'classification', 'metric': 'auc', 'target_columns': ['CT_TOX', 'FDA_APPROVED']},
    'Tox21': {'task_type': 'classification', 'metric': 'auc', 'target_columns': [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]},
    'SIDER': {'task_type': 'classification', 'metric': 'auc', 'target_columns': [
        # 27 side effect categories - adjust based on your actual column names
    ]},
}

def train_chemprop_single(dataset_name, config, num_folds=3, seed=0):
    """
    Train Chemprop on a single dataset with scaffold split
    
    Example command:
    chemprop_train 
        --data_path data.csv 
        --dataset_type classification 
        --save_dir model_output 
        --split_type scaffold_balanced
        --num_folds 3
        --seed 0
    """
    
    data_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    save_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_seed{seed}")
    
    cmd = [
        "chemprop_train",
        "--data_path", data_path,
        "--dataset_type", config['task_type'],
        "--save_dir", save_dir,
        "--split_type", "scaffold_balanced",
        "--num_folds", str(num_folds),
        "--seed", str(seed),
        "--metric", config['metric'],
        "--epochs", "50",  # Adjust as needed
        "--batch_size", "50",
    ]
    
    # Add target columns if multi-task
    if len(config['target_columns']) > 1:
        cmd.extend(["--target_columns"] + config['target_columns'])
    
    print(f"\n{'='*80}")
    print(f"Training Chemprop on {dataset_name} (seed={seed})")
    print(f"{'='*80}")
    print("Command:", " ".join(cmd))
    print()
    
    # Run training
    subprocess.run(cmd, check=True)
    
    # Results will be saved to: {save_dir}/fold_{0,1,2}/test_scores.csv
    print(f"\n✓ Training complete. Results saved to: {save_dir}")

def train_all_datasets_multiple_seeds(seeds=[0, 1, 2]):
    """Train all datasets with multiple seeds for statistical significance"""
    
    for dataset_name, config in DATASETS.items():
        for seed in seeds:
            try:
                train_chemprop_single(dataset_name, config, num_folds=3, seed=seed)
            except Exception as e:
                print(f"\n❌ Error training {dataset_name} with seed {seed}")
                print(f"   {type(e).__name__}: {e}")
                continue

# STEP 2: Extract and Compare Results
# ------------------------------------

def extract_chemprop_results():
    """
    Extract test scores from all Chemprop runs and create comparison table
    """
    import pandas as pd
    import numpy as np
    
    results = []
    
    for dataset_name in DATASETS.keys():
        dataset_results = []
        
        # Check all seed runs
        for seed_dir in Path(OUTPUT_DIR).glob(f"{dataset_name}_seed*"):
            # Each fold has a test_scores.csv
            fold_scores = []
            for fold_dir in seed_dir.glob("fold_*"):
                scores_file = fold_dir / "test_scores.csv"
                if scores_file.exists():
                    df = pd.read_csv(scores_file)
                    # Chemprop saves metric in different formats, adapt as needed
                    score = df.iloc[0][DATASETS[dataset_name]['metric']]
                    fold_scores.append(score)
            
            if fold_scores:
                # Average across folds for this seed
                dataset_results.append(np.mean(fold_scores))
        
        if dataset_results:
            results.append({
                'Dataset': dataset_name,
                'Task': DATASETS[dataset_name]['task_type'],
                'Metric': DATASETS[dataset_name]['metric'].upper(),
                'Mean': np.mean(dataset_results),
                'Std': np.std(dataset_results),
                'Seeds': len(dataset_results),
            })
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, "chemprop_baseline_results.csv")
    df_results.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("CHEMPROP BASELINE RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: {output_file}")
    
    return df_results

# USAGE EXAMPLES
# --------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Chemprop baselines")
    parser.add_argument("--mode", choices=["single", "all", "extract"], required=True,
                        help="single: train one dataset, all: train all datasets, extract: extract results")
    parser.add_argument("--dataset", type=str, help="Dataset name (for single mode)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Multiple seeds for 'all' mode")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if args.dataset not in DATASETS:
            print(f"❌ Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASETS.keys())}")
        else:
            train_chemprop_single(args.dataset, DATASETS[args.dataset], num_folds=3, seed=args.seed)
    
    elif args.mode == "all":
        train_all_datasets_multiple_seeds(seeds=args.seeds)
    
    elif args.mode == "extract":
        extract_chemprop_results()

"""
QUICK START COMMANDS (run in molprop environment):
===================================================

1. Inspect your datasets first:
   python inspect_datasets.py

2. Train on single dataset:
   python chemprop_baseline.py --mode single --dataset ESOL --seed 0

3. Train on all datasets with 3 seeds:
   python chemprop_baseline.py --mode all --seeds 0 1 2

4. Extract results to CSV:
   python chemprop_baseline.py --mode extract

NOTES:
------
- Chemprop uses scaffold split by default (same as your MoE model)
- Adjust --epochs, --batch_size as needed
- For GPU training, Chemprop will auto-detect CUDA
- Results are averaged across 3 folds per seed

COMPARISON WITH YOUR MOE MODEL:
-------------------------------
After running, compare chemprop_baseline_results.csv with your MoE results
from Screenshot_20260329_205935.png (the final MoE K=4 results)
"""
