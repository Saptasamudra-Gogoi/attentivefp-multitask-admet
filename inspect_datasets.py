"""
Dataset Format Inspector for Chemprop Data
Run this script on your Windows machine in the molprop conda environment
"""

import pandas as pd
import os
from pathlib import Path

# Adjust this path to your actual data location
DATA_DIR = r"D:\molprop_project\temp\chemprop_data"  # Change this if needed

# Datasets to inspect
DATASETS = ['ESOL', 'FreeSolv', 'Lipo', 'BACE', 'BBBP', 'HIV', 'ClinTox', 'Tox21', 'SIDER']

print("=" * 80)
print("CHEMPROP DATASET FORMAT INSPECTION")
print("=" * 80)

for ds_name in DATASETS:
    csv_path = os.path.join(DATA_DIR, f"{ds_name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ {ds_name}: FILE NOT FOUND at {csv_path}")
        continue
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 80}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for SMILES column
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        
        if smiles_col:
            print(f"✓ SMILES column found: '{smiles_col}'")
        else:
            print(f"⚠️  No 'smiles' column found! Available: {df.columns.tolist()}")
        
        # Identify target columns (numeric columns that aren't the SMILES)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"Numeric columns (potential targets): {numeric_cols}")
        
        # Show first row
        print(f"\nFirst row:")
        for col in df.columns:
            val = df.iloc[0][col]
            if isinstance(val, str) and len(val) > 50:
                val = val[:50] + "..."
            print(f"  {col}: {val}")
        
        # Check for missing values in target columns
        if numeric_cols:
            print(f"\nMissing values in numeric columns:")
            for col in numeric_cols:
                n_missing = df[col].isna().sum()
                pct = 100 * n_missing / len(df)
                print(f"  {col}: {n_missing} ({pct:.1f}%)")
    
    except Exception as e:
        print(f"\n❌ {ds_name}: ERROR reading file")
        print(f"   {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

# Summary of expected formats
print("""
EXPECTED CHEMPROP FORMAT:
- CSV file with header row
- Must have a 'smiles' column (case-insensitive)
- Target columns should be numeric (float or int)
- For classification: targets should be 0/1 or similar integers
- For regression: targets should be continuous floats

COMMON ISSUES:
1. Column name mismatch (e.g., 'SMILES' vs 'smiles' vs 'Smiles')
2. Missing header row
3. Wrong delimiter (tab instead of comma)
4. Extra whitespace in column names
""")
