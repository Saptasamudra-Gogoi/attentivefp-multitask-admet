"""
Fix for review item 2.7 (GROVER results): grover/util/utils.py load_features()
was changed to read the 'fps' key instead of 'features'. This script:
  1. Checks what key your actual cached .npz feature files use
  2. Patches load_features() to match reality (not guess)
  3. Prints next steps to re-run GROVER eval

Run from D:\\molprop_project\\grover\\ with moe_admet or grover_env activated.
"""
import numpy as np
import glob
import os

UTILS_PATH = "grover/util/utils.py"

# 1. Find cached feature files and check their actual key
npz_candidates = glob.glob(os.path.join("**", "*.npz"), recursive=True)
print(f"Found {len(npz_candidates)} .npz files")

real_key = None
for f in npz_candidates[:20]:
    try:
        data = np.load(f)
        keys = list(data.files)
        print(f"  {f}: keys={keys}")
        if real_key is None and keys:
            real_key = keys[0]
    except Exception as e:
        print(f"  {f}: could not open ({e})")

if not npz_candidates:
    print("No .npz files found here -- check you're in the right directory")
    print("(should be run from inside the grover/ folder or point this at exampledata/)")
    raise SystemExit(1)

if real_key is None:
    print("Could not determine key from any file. Inspect manually.")
    raise SystemExit(1)

print(f"\nDetected real key in your cached features: '{real_key}'")

# 2. Read utils.py, check current key, patch if mismatched
with open(UTILS_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if f"['{real_key}']" in content:
    print(f"utils.py already uses the correct key '{real_key}' -- no patch needed.")
else:
    # Replace whichever wrong key is currently there
    for wrong_key in ["features", "fps"]:
        old_line = f"features = np.load(path)['{wrong_key}']"
        if old_line in content:
            new_line = f"features = np.load(path)['{real_key}']"
            content = content.replace(old_line, new_line)
            with open(UTILS_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched utils.py: '{wrong_key}' -> '{real_key}'")
            break
    else:
        print("Could not find the expected load_features() line to patch -- check utils.py manually.")

print("""
Next steps to actually fix 2.7's GROVER numbers:
1. Confirm the patch above is correct (git diff grover/util/utils.py)
2. Re-run feature extraction + eval for GROVER on all 9 MoleculeNet datasets:
     python main.py eval --data_path exampledata/finetune/<dataset>.csv \\
                          --features_path exampledata/finetune/<dataset>.npz \\
                          --checkpoint_dir model/finetune/<dataset> \\
                          --dataset_type <classification|regression> \\
                          --split_type scaffold_balanced \\
                          --ensemble_size 1 --num_folds 3 \\
                          --metric <auc|rmse|mae> --no_features_scaling
3. Confirm BBBP/BACE stop returning NaN and std is nonzero
4. Regenerate grover_results.json from the eval outputs (not by hand)
5. Only then update Table 1's GROVER column
""")
