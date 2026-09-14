import shutil

path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
shutil.copy(path, path + ".before_doi_update.bak")
content = open(path, encoding="utf-8").read()

old_doi = "10.5281/zenodo.22162186"
new_doi = "10.5281/zenodo.22746214"

count = content.count(old_doi)
print(f"Found {count} occurrence(s) of old DOI")
assert count >= 1, "Old DOI not found -- check manuscript content"

content = content.replace(old_doi, new_doi)
open(path, "w", encoding="utf-8").write(content)
print(f"Replaced {count} occurrence(s). New DOI: {new_doi}")