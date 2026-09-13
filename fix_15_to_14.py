import shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_1416_fix.bak")
content = open(path, encoding="utf-8").read()

replacements = [
    (
        "Two controls temper this: k-means clustering of the same trained "
        "representation recovers this organization at least as strongly on "
        "15 of 16 comparisons, and across all 22 TDC datasets, "
        "specialization strength does not predict the size of MoE "
        "performance gain (Spearman r = 0.019, P = 0.934)."
        ,
        "Two controls temper this: k-means clustering of the same trained "
        "representation recovers this organization at least as strongly on "
        "14 of 16 comparisons, and across all 22 TDC datasets, "
        "specialization strength does not predict the size of MoE "
        "performance gain (Spearman r = 0.019, P = 0.934)."
    ),
    (
        "Across these 8 datasets \u00d7 2 descriptors (16 comparisons), "
        "k-means matched or exceeded the router's \u03b7\u00b2 on 15 of 16. "
        "This result points to the representation, not the router:"
        ,
        "Across these 8 datasets \u00d7 2 descriptors (16 comparisons; "
        "Fig. 3C), k-means matched or exceeded the router's \u03b7\u00b2 on "
        "14 of 16, with the two exceptions (Caco-2 aromatic-ring count and "
        "AstraZeneca Lipophilicity LogP) both cases where the router's own "
        "\u03b7\u00b2 was itself modest (0.40 and 0.02 respectively) and the "
        "gap between router and k-means was small in absolute terms. This "
        "result points to the representation, not the router:"
    ),
]

for i, (old, new) in enumerate(replacements, 1):
    count = content.count(old)
    print(f"Replacement {i}: found {count} occurrence(s)")
    if count == 1:
        content = content.replace(old, new)
        print(f"  Applied.")
    else:
        print(f"  SKIPPED -- expected exactly 1 match.")

open(path, "w", encoding="utf-8").write(content)
print("\nFile saved.")
