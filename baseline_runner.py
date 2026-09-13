# How to Modify baseline_runner.py to Capture Test Metrics

## Why You Need This
Your current baseline logs **training loss only**. For a Q1 publication, you need:
- **Test ROC-AUC** (classification tasks)
- **Test RMSE/MAE** (regression tasks)
- **Mean ± std across 3 seeds**
- **Per-dataset breakdowns**

---

## Step 1: Open baseline_runner.py in VS Code

```
Right-click on D:\molprop_project\baseline_runner.py → Open with Code
```

Or from terminal:
```bash
code D:\molprop_project\baseline_runner.py
```

---

## Step 2: Add imports at the top

Find the imports section (around line 1-15) and add:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import json
from datetime import datetime
```

---

## Step 3: Add evaluation function

Find where your `train_epoch()` function is defined. After that function, add this new function:

```python
def evaluate_epoch(model, loaders, device):
    """Evaluate on test set, return ROC-AUC/RMSE/MAE per task"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    task_metrics = {}
    
    for task_idx in range(all_preds.shape[1]):
        pred_task = all_preds[:, task_idx]
        label_task = all_labels[:, task_idx]
        
        mask = ~np.isnan(label_task)
        if np.sum(mask) == 0:
            continue
            
        pred_task = pred_task[mask]
        label_task = label_task[mask]
        
        # ROC-AUC for binary classification
        if len(np.unique(label_task)) == 2:
            try:
                auc = roc_auc_score(label_task, pred_task)
                task_metrics[f'task_{task_idx}_auc'] = round(auc, 4)
            except:
                pass
        
        # RMSE and MAE
        rmse = np.sqrt(mean_squared_error(label_task, pred_task))
        mae = mean_absolute_error(label_task, pred_task)
        task_metrics[f'task_{task_idx}_rmse'] = round(rmse, 4)
        task_metrics[f'task_{task_idx}_mae'] = round(mae, 4)
    
    model.train()
    return task_metrics
```

---

## Step 4: Modify run_seed() function

Find your `run_seed()` function and modify it to call `evaluate_epoch()` at the end of each dataset:

**Before (current code):**
```python
def run_seed(model_class, model_name, seed):
    results = {}
    for dataset_name in DATASETS:
        # ... train loop ...
        # Nothing here to capture metrics
    return results
```

**After (modified):**
```python
def run_seed(model_class, model_name, seed):
    results = {}
    for dataset_name in DATASETS:
        # ... your existing training code ...
        
        # ADD THIS at the end of the dataset loop, before moving to next dataset:
        test_metrics = evaluate_epoch(model, loaders, device)
        
        results[dataset_name] = {
            'final_loss': train_loss,
            'test_metrics': test_metrics
        }
        
        print(f"    Final Loss: {train_loss:.4f}")
        for metric_name, value in test_metrics.items():
            print(f"    {metric_name}: {value}")
    
    return results
```

---

## Step 5: Modify main execution block

Find the bottom of your file where you call `run_seed()`. Replace:

**Before:**
```python
if __name__ == "__main__":
    for seed in [42, 123, 7]:
        print(f"\nSEED {seed}")
        for name, score in run_seed(model_class, model_name, seed).items():
            print(f"{name}: {score}")
```

**After:**
```python
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}
    
    for seed in [42, 123, 7]:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")
        
        seed_results = run_seed(model_class, model_name, seed)
        all_results[f'seed_{seed}'] = seed_results
    
    # Save to JSON
    output_file = f"results/Baseline_{model_name}_metrics_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nMetrics saved to: {output_file}")
```

---

## Step 6: Save and test

1. **Save the file** (Ctrl+S in VS Code)
2. **Run it:**
   ```bash
   python baseline_runner.py
   ```
3. **Check output** — you should see test metrics printing for each dataset
4. **JSON file** will appear in `results/` folder with all metrics

---

## What the JSON output looks like

```json
{
  "seed_42": {
    "ESOL": {
      "final_loss": 0.1234,
      "test_metrics": {
        "task_0_rmse": 0.5621,
        "task_0_mae": 0.4432
      }
    },
    "FreeSolv": {
      "final_loss": 0.2145,
      "test_metrics": {
        "task_0_auc": 0.8234,
        "task_1_rmse": 0.3421
      }
    }
    ...
  },
  "seed_123": { ... },
  "seed_7": { ... }
}
```

---

## Next: Aggregate results for your paper

Once you have metrics across all 3 seeds, create a summary table:

```python
# Compute mean ± std across seeds
for dataset in DATASETS:
    aucs = [all_results[f'seed_{s}'][dataset]['test_metrics'].get('task_0_auc') 
            for s in [42, 123, 7]]
    aucs = [x for x in aucs if x is not None]
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"{dataset}: {mean_auc:.4f} ± {std_auc:.4f}")
```

---

## Troubleshooting

**Error: `loaders['test']` not found**
- Check how you define your loaders in `load_dataset()` — adjust the dict key if needed

**Error: `batch.x`, `batch.edge_index` not recognized**
- Your batch object might have different attribute names — print `batch` to inspect

**No metrics values (all None)**
- Check if your model output shape matches expected (should be `[batch_size, num_tasks]`)
- Print `all_preds.shape` and `all_labels.shape` to debug

---

## Questions?

Paste your `run_seed()` function and I'll help you integrate this directly.
