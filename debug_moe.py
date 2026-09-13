"""
Quick sanity check — runs MoE-GIN on BBBP for 5 epochs
Prints loss and AUC each epoch so we can see if learning happens
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load BBBP ──
dataset = MoleculeNet(root="./data", name="BBBP")
print(f"Dataset: {len(dataset)} molecules")
print(f"x shape: {dataset[0].x.shape}")
print(f"y shape: {dataset[0].y.shape}")
print(f"y values sample: {dataset[0].y}")
print(f"edge_attr shape: {dataset[0].edge_attr.shape}")

# Simple split
n = len(dataset)
train_data = list(dataset[:int(n*0.8)])
val_data   = list(dataset[int(n*0.8):])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)

# Check label distribution
labels = []
for data in dataset:
    y = data.y.numpy().flatten()
    labels.extend(y[~np.isnan(y)].tolist())
labels = np.array(labels)
print(f"\nLabel distribution: {np.bincount(labels.astype(int))} (0s, 1s)")
print(f"Positive rate: {labels.mean():.3f}")

# ── Simple MoE-GIN ──
class SparseMoE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = torch.nn.ModuleList([
            Sequential(Linear(in_dim, out_dim), torch.nn.GELU(), Linear(out_dim, out_dim))
            for _ in range(num_experts)
        ])
        self.gate = Linear(in_dim, num_experts, bias=False)
        self._lb = torch.tensor(0.0)

    def forward(self, x):
        logits = self.gate(x)
        topk_v, topk_i = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_v, dim=-1)
        out = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for k in range(self.top_k):
            idx = topk_i[:, k]
            w   = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])
        probs = F.softmax(logits, dim=-1)
        self._lb = (probs.mean(0) * probs.mean(0)).sum() * self.num_experts
        return out

class MoEGIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 128
        self.conv1 = GINConv(Sequential(Linear(9, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden)), train_eps=True)
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden)), train_eps=True)
        self.moe = SparseMoE(hidden, hidden, num_experts=4, top_k=2)
        self.lin = Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.moe(x)
        return self.lin(x).squeeze(-1)   # shape (N,) not (N,1)

model = MoEGIN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n── Training 10 epochs ──")
for epoch in range(1, 11):
    # Train
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt  = batch.y.float().squeeze(-1)
        mask = ~torch.isnan(tgt)
        if mask.sum() == 0: continue
        loss = F.binary_cross_entropy_with_logits(out[mask], tgt[mask])
        loss = loss + 0.01 * model.moe._lb
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Eval
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
            tgt  = batch.y.float().squeeze(-1)
            mask = ~torch.isnan(tgt)
            all_p.extend(torch.sigmoid(out[mask]).cpu().numpy())
            all_t.extend(tgt[mask].cpu().numpy())

    auc = roc_auc_score(all_t, all_p)
    print(f"  Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

print("\nIf AUC > 0.5 after epoch 3, the model works correctly.")
print("If AUC stays at 0.0 or 1.0, there is still a shape/label issue.")
