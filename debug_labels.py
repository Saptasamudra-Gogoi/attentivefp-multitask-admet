from torch_geometric.datasets import MoleculeNet
import torch

for name, root in [("BBBP","data/BBBP"), ("BACE","data/BACE")]:
    ds = MoleculeNet(root=root, name=name)
    y = ds[0].y
    print(f"\n{name}:")
    print(f"  y:      {y}")
    print(f"  dtype:  {y.dtype}")
    print(f"  shape:  {y.shape}")
    print(f"  has nan: {torch.isnan(y.float()).any()}")
    # Check a few more samples
    has_nan_count = sum(1 for i in range(min(50,len(ds))) if torch.isnan(ds[i].y.float()).any())
    print(f"  samples with nan (first 50): {has_nan_count}")
