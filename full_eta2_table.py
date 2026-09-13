"""
Full 8 descriptors x 4 datasets eta^2 table (review item 2.5).
Run from D:\\molprop_project\\ with moe_admet activated (needs rdkit, scipy, pandas).

EDIT INPUT_FILES below to point at your per-dataset expert-assignment files.
Each file must have columns: smiles, expert_id  (or edit COL_SMILES/COL_EXPERT)

The 8 descriptors (Methods says all 8 were computed on all 4 datasets):
LogP, aromatic ring count, TPSA, MW, HBD, HBA, rotatable bonds, formal charge
-> extend/rename DESCRIPTORS dict if your Methods section lists different 8.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

COL_SMILES = "smiles"
COL_EXPERT = "expert_id"

INPUT_FILES = {
    "LD50":    r"D:\molprop_project\expert_assignments\ld50.csv",
    "Dataset2": r"D:\molprop_project\expert_assignments\dataset2.csv",
    "Dataset3": r"D:\molprop_project\expert_assignments\dataset3.csv",
    "Dataset4": r"D:\molprop_project\expert_assignments\dataset4.csv",
}

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "LogP": Crippen.MolLogP(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MW": Descriptors.MolWt(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
    }

def eta_squared(values, groups):
    """One-way ANOVA eta^2 = SS_between / SS_total."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    grand_mean = df["v"].mean()
    ss_total = ((df["v"] - grand_mean) ** 2).sum()
    ss_between = sum(
        len(sub) * (sub["v"].mean() - grand_mean) ** 2
        for _, sub in df.groupby("g")
    )
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total

def main():
    results = {}
    for dataset_name, path in INPUT_FILES.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[skip] {dataset_name}: file not found at {path} -- edit INPUT_FILES")
            continue

        desc_rows = df[COL_SMILES].apply(compute_descriptors)
        desc_df = pd.DataFrame(list(desc_rows))
        desc_df[COL_EXPERT] = df[COL_EXPERT].values

        row = {}
        for desc_name in ["LogP", "AromaticRings", "TPSA", "MW", "HBD", "HBA", "RotBonds", "FormalCharge"]:
            row[desc_name] = eta_squared(desc_df[desc_name], desc_df[COL_EXPERT])
        results[dataset_name] = row

    out = pd.DataFrame(results).T
    out.to_csv("full_eta2_8x4_table.csv")
    print(out.round(3))
    print("\nSaved: full_eta2_8x4_table.csv")
    print("Next: run Kruskal-Wallis P-value per cell too (scipy.stats.kruskal) if Li wants both stats.")

if __name__ == "__main__":
    main()
