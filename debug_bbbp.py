"""
Debug BBBP — check labels, predictions, and data integrity
Run: python debug_bbbp.py
"""
import torch
import numpy as np
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = "./data"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load dataset ─────────────────────────────────────────────────────────────
dataset = MoleculeNet(root=DATA_ROOT, name="BBBP")
print(f"Dataset size: {len(dataset)}")
print(f"num_node_features: {dataset.num_node_features}")
print(f"num_tasks: {dataset.num_classes}")

# ── Check first few samples ───────────────────────────────────────────────────
print("\n--- First 10 samples ---")
for i in range(10):
    d = dataset[i]
    print(f"  [{i}] y={d.y.numpy()}, y.shape={d.y.shape}, x.shape={d.x.shape}, edges={d.edge_index.shape[1]}")

# ── Check label distribution ──────────────────────────────────────────────────
all_labels = []
for d in dataset:
    all_labels.append(d.y.numpy().flatten())
all_labels = np.concatenate(all_labels)

print(f"\n--- Label stats ---")
print(f"Total labels: {len(all_labels)}")
print(f"NaN count: {np.isnan(all_labels).sum()}")
print(f"Unique values: {np.unique(all_labels[~np.isnan(all_labels)])}")
print(f"Class 0: {(all_labels == 0).sum()}, Class 1: {(all_labels == 1).sum()}")

# ── Check what a batch looks like ─────────────────────────────────────────────
loader = DataLoader(dataset, batch_size=32, shuffle=False)
batch  = next(iter(loader))
print(f"\n--- Batch check ---")
print(f"batch.y.shape: {batch.y.shape}")
print(f"batch.y[:10]: {batch.y[:10]}")
print(f"batch.x.shape: {batch.x.shape}")
print(f"batch.x dtype: {batch.x.dtype}")

# ── Try reshape both ways ─────────────────────────────────────────────────────
y = batch.y.numpy()
print(f"\nbatch.y raw shape: {y.shape}")
try:
    r1 = y.reshape(-1, 1)
    print(f"reshape(-1,1): {r1.shape} — first 5: {r1[:5].flatten()}")
except Exception as e:
    print(f"reshape(-1,1) failed: {e}")

# ── Simulate model output ─────────────────────────────────────────────────────
# Fake logits ~ N(0,1)
torch.manual_seed(0)
fake_out = torch.randn(32, 1)
fake_probs = torch.sigmoid(fake_out).numpy()
labels = batch.y.numpy()
if labels.ndim == 1:
    labels = labels.reshape(-1, 1)

from sklearn.metrics import roc_auc_score
mask = ~np.isnan(labels[:, 0])
uniq = np.unique(labels[mask, 0])
print(f"\n--- Simulated AUC check ---")
print(f"mask.sum(): {mask.sum()}, unique labels: {uniq}")
if len(uniq) >= 2:
    auc = roc_auc_score(labels[mask, 0], fake_probs[mask, 0])
    print(f"AUC with random preds: {auc:.4f}  (should be ~0.5)")
else:
    print("Only one class in batch — that's the bug source")

print("\nDone.")
