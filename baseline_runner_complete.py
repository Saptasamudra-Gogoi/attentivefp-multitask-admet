"""
Baseline GNN Benchmark: GIN, GCN, GAT on MoleculeNet datasets
Logs training loss and test metrics (ROC-AUC, RMSE, MAE) per seed
Saves results to JSON for Q1 paper publication
"""

import os
import json
import warnings
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = [
    'ESOL', 'FreeSolv', 'Lipophilicity',
    'PCBA', 'MUV', 'HIV',
    'BACE', 'BBPB', 'Tox21'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.001

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GINModel(nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels, hidden_dim=128, num_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )) for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.float())
        
        for conv in self.convs:
            x = conv(x, edge_index) + x
            x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GCNModel(nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels, hidden_dim=64, num_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_dim)
        
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.float())
        
        for conv in self.convs:
            x = conv(x, edge_index) + x
            x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GATModel(nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels, hidden_dim=64, num_layers=3, heads=4):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_dim)
        
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads) for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.float())
        
        for conv in self.convs:
            x = conv(x, edge_index) + x
            x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_name, root='./data'):
    """Load MoleculeNet dataset and return train/val/test loaders"""
    dataset = MoleculeNet(root=root, name=dataset_name)
    
    # Split: 80% train, 10% val, 10% test
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, dataset


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, loaders, optimizer):
    """Train for one epoch, return loss"""
    model.train()
    total_loss = 0.0
    
    for batch in loaders['train']:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Multi-task loss: handle missing values with masking
        loss = 0.0
        num_tasks = batch.y.shape[1]
        
        for task_idx in range(num_tasks):
            mask = ~torch.isnan(batch.y[:, task_idx])
            if mask.sum() == 0:
                continue
            
            pred_task = pred[mask, task_idx]
            y_task = batch.y[mask, task_idx]
            
            # Determine loss function based on task type
            if len(torch.unique(y_task)) == 2:  # Classification
                task_loss = F.binary_cross_entropy_with_logits(pred_task, y_task)
            else:  # Regression
                task_loss = F.mse_loss(pred_task, y_task)
            
            loss += task_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loaders['train'])


def evaluate(model, loaders, dataset):
    """Evaluate on test set, return metrics per task"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(DEVICE)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    task_metrics = {}
    
    for task_idx in range(all_preds.shape[1]):
        pred_task = all_preds[:, task_idx]
        label_task = all_labels[:, task_idx]
        
        # Filter missing values
        mask = ~np.isnan(label_task)
        if np.sum(mask) == 0:
            continue
        
        pred_task = pred_task[mask]
        label_task = label_task[mask]
        
        # Classification: ROC-AUC
        if len(np.unique(label_task)) == 2:
            try:
                auc = roc_auc_score(label_task, pred_task)
                task_metrics[f'task_{task_idx}_auc'] = round(auc, 4)
            except:
                pass
        
        # Regression: RMSE and MAE
        rmse = np.sqrt(mean_squared_error(label_task, pred_task))
        mae = mean_absolute_error(label_task, pred_task)
        task_metrics[f'task_{task_idx}_rmse'] = round(rmse, 4)
        task_metrics[f'task_{task_idx}_mae'] = round(mae, 4)
    
    model.train()
    return task_metrics


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_seed(model_class, model_name, seed):
    """Run one seed across all datasets"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {}
    
    for dataset_name in DATASETS:
        print(f"  {dataset_name:12s}", end=" | ", flush=True)
        
        try:
            # Load dataset
            loaders, dataset = load_dataset(dataset_name)
            
            in_channels = dataset.num_features
            out_channels = dataset.num_tasks if hasattr(dataset, 'num_tasks') else dataset[0].y.shape[1]
            edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
            
            # Create model
            model = model_class(in_channels, edge_dim, out_channels).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            # Training loop
            for epoch in range(1, EPOCHS + 1):
                train_loss = train_epoch(model, loaders, optimizer)
            
            # Final evaluation
            final_metrics = evaluate(model, loaders, dataset)
            
            results[dataset_name] = {
                'final_loss': round(train_loss, 4),
                'final_metrics': final_metrics
            }
            
            # Print metrics
            if 'task_0_auc' in final_metrics:
                print(f"AUC={final_metrics['task_0_auc']:.3f}", end="")
            elif 'task_0_rmse' in final_metrics:
                print(f"RMSE={final_metrics['task_0_rmse']:.3f}", end="")
            print()
        
        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            results[dataset_name] = {'error': str(e)}
    
    return results


def main():
    """Run all baselines on all seeds"""
    print(f"\nDevice: {DEVICE}\n")
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    all_results = {}
    
    for model_class, model_name in [(GINModel, "GIN"), (GCNModel, "GCN"), (GATModel, "GAT")]:
        print(f"\n{'='*70}")
        print(f"  {model_name} Baseline — {len(DATASETS)} Datasets")
        print(f"{'='*70}\n")
        
        model_results = {}
        
        for seed in [42, 123, 7]:
            print(f"SEED {seed:3d} | ", end="", flush=True)
            seed_results = run_seed(model_class, model_name, seed)
            model_results[f'seed_{seed}'] = seed_results
        
        all_results[model_name] = model_results
    
    # Save results to JSON
    output_file = f"results/Baseline_Metrics_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
