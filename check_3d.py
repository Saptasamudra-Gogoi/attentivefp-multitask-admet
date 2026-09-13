from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles('CCO')
mol = Chem.AddHs(mol)
r = AllChem.EmbedMolecule(mol, randomSeed=42)
print('Result:', r)
print('Conformers:', mol.GetNumConformers())
