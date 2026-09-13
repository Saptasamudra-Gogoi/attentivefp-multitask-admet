# Add this to your baseline_runner.py after your train_epoch() function
# This shows how to extract test metrics for each seed/dataset

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

def evaluate_epoch(model, loaders, device):
    """Evaluate model on test set and return metrics per task"""
    model.eval()
    
    all_preds = []
    all_labels = []
    task_metrics = {}
    
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N_samples, N_tasks)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics per task
    for task_idx in range(all_preds.shape[1]):
        pred_task = all_preds[:, task_idx]
        label_task = all_labels[:, task_idx]
        
        # Filter out missing values (NaN)
        mask = ~np.isnan(label_task)
        pred_task = pred_task[mask]
        label_task = label_task[mask]
        
        if len(pred_task) == 0:
            continue
        
        # Determine metric type based on task (classification vs regression)
        # For classification tasks (binary): use ROC-AUC
        if len(np.unique(label_task)) == 2:
            try:
                auc = roc_auc_score(label_task, pred_task)
                task_metrics[f'task_{task_idx}_auc'] = auc
            except:
                task_metrics[f'task_{task_idx}_auc'] = None
        
        # For regression tasks: use RMSE and MAE
        rmse = np.sqrt(mean_squared_error(label_task, pred_task))
        mae = mean_absolute_error(label_task, pred_task)
        task_metrics[f'task_{task_idx}_rmse'] = rmse
        task_metrics[f'task_{task_idx}_mae'] = mae
    
    model.train()
    return task_metrics


def run_seed(model_class, model_name, seed):
    """Modified run_seed to capture test metrics"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {}
    
    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}...")
        
        # Load data (your existing code)
        loaders = load_dataset(dataset_name)
        
        model = model_class(in_channels, out_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_test_auc = 0
        best_metrics = {}
        
        # Training loop
        for epoch in range(1, 201):
            train_loss = train_epoch(model, loaders, optimizer)
            
            # Evaluate on test set every 10 epochs
            if epoch % 10 == 0:
                test_metrics = evaluate_epoch(model, loaders, device)
                
                # Track best performance
                if 'task_0_auc' in test_metrics and test_metrics['task_0_auc'] is not None:
                    if test_metrics['task_0_auc'] > best_test_auc:
                        best_test_auc = test_metrics['task_0_auc']
                        best_metrics = test_metrics.copy()
        
        # Final test evaluation
        final_metrics = evaluate_epoch(model, loaders, device)
        
        # Store results with both loss and metrics
        results[dataset_name] = {
            'final_loss': train_loss,
            'final_metrics': final_metrics,
            'best_metrics': best_metrics
        }
        
        # Print summary
        print(f"    Loss: {train_loss:.4f}")
        if 'task_0_auc' in final_metrics:
            print(f"    Test AUC: {final_metrics['task_0_auc']:.4f}")
        if 'task_0_rmse' in final_metrics:
            print(f"    Test RMSE: {final_metrics['task_0_rmse']:.4f}")
    
    return results


# Modified main loop to save results to JSON for paper analysis
if __name__ == "__main__":
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_all_seeds = {}
    
    for seed in [42, 123, 7]:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")
        
        seed_results = run_seed(GINModel, "GIN", seed)
        results_all_seeds[f'seed_{seed}'] = seed_results
    
    # Save to JSON for analysis
    output_file = f"results/Baseline_GIN_metrics_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results_all_seeds, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
