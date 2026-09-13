"""
Find whatever script produced the k-means null control result (15/16
comparisons, Spearman r=0.019, P=0.934) so we can document the real
methodology instead of guessing it.

Run from D:\\molprop_project\\ with moe_admet activated.
"""
import glob
import re

# Candidate filenames based on naming patterns already seen in this project
candidates = set()
for pattern in ["*kmeans*", "*k_means*", "*cluster*", "*null_control*", "*sss*"]:
    candidates.update(glob.glob(pattern) + glob.glob(f"**/{pattern}", recursive=True))

py_candidates = sorted(f for f in candidates if f.endswith(".py"))
other_candidates = sorted(f for f in candidates if not f.endswith(".py") and "." in f.split("\\")[-1])

print(f"Found {len(py_candidates)} candidate Python scripts:\n")
for f in py_candidates:
    print(" ", f)

print(f"\nFound {len(other_candidates)} candidate data/output files:\n")
for f in other_candidates:
    print(" ", f)

print("\n" + "=" * 70)
print("Scanning each .py file for k-means / Spearman signatures...")
print("=" * 70)

for f in py_candidates:
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except Exception as e:
        print(f"\n[skip] {f}: {e}")
        continue

    has_kmeans = "KMeans" in content or "k_means" in content.lower()
    has_spearman = "spearman" in content.lower()
    if not (has_kmeans or has_spearman):
        continue

    print(f"\n--- {f} ---")
    print(f"  Has KMeans: {has_kmeans} | Has Spearman: {has_spearman}")

    # Pull out key parameter lines
    for pattern, label in [
        (r"KMeans\([^)]*\)", "KMeans call"),
        (r"n_init\s*=\s*\w+", "n_init"),
        (r"random_state\s*=\s*\w+", "random_state"),
        (r"StandardScaler|standardiz|normali[sz]e", "standardization mention"),
        (r"spearmanr\([^)]*\)", "spearmanr call"),
        (r"n_clusters\s*=\s*\w+", "n_clusters"),
    ]:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            print(f"  {label}: {matches[:5]}")

print("""
--- Next step ---
Open whichever file(s) above look most relevant and paste me:
1. The KMeans(...) call with its exact arguments
2. Whether StandardScaler (or similar) is applied before clustering
3. What array is passed to spearmanr(...) -- the actual variable names/what
   they contain
4. Where the embeddings being clustered come from (which model's forward pass)
""")
