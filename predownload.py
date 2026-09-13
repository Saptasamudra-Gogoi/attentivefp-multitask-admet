from torch_geometric.datasets import MoleculeNet

datasets = ['ESOL', 'FreeSolv', 'Lipo', 'BBBP', 'BACE', 'Tox21', 'SIDER', 'ClinTox', 'HIV']

for d in datasets:
    try:
        ds = MoleculeNet(root='./data', name=d)
        print(f'{d}: {len(ds)} molecules OK')
    except Exception as e:
        print(f'{d}: FAILED — {e}')
