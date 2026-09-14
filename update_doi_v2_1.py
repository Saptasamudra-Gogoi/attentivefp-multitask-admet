import shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_doi_v2_1.bak")
content = open(path, encoding="utf-8").read()

old_doi = "10.5281/zenodo.22746214"
new_doi = "10.5281/zenodo.22746900"

count = content.count(old_doi)
print(f"Found {count} occurrence(s) of old DOI")
assert count >= 1, "Old DOI not found -- check manuscript content"

content = content.replace(old_doi, new_doi)
open(path, "w", encoding="utf-8").write(content)
print(f"Replaced {count} occurrence(s). New DOI: {new_doi}")
