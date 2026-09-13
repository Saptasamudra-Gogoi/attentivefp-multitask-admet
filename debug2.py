import torch
from torch_geometric.datasets import MoleculeNet

for dsname, root in [("BBBP","data/BBBP"), ("BACE","data/BACE")]:
    ds = MoleculeNet(root=root, name=dsname)
    print(f"\n{dsname} — {len(ds)} molecules")
    for i in range(5):
        y = ds[i].y
        print(f"  [{i}] y={y}  shape={y.shape}  dtype={y.dtype}")
    # Check unique label values
    all_labels = torch.cat([ds[i].y.float().view(-1) for i in range(len(ds))])
    unique = torch.unique(all_labels)
    print(f"  Unique label values: {unique}")
    print(f"  Has NaN: {torch.isnan(all_labels).any()}")
