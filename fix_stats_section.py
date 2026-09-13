path = "Sapta_MoE-ADMET_manuscript_REVISED.md"
content = open(path, encoding="utf-8").read()

old = ("for each dataset we used the mean MoE-GCN value across seeds and the "
       "corresponding GCN value as reported in its source table \u2014 a "
       "three-seed mean for the nine TDC datasets, and a single "
       "scaffold-split run for the three MoleculeNet datasets, which do "
       "not have a repeated-seed GCN estimate (see Backbone architectures, "
       "above) \u2014 giving one paired observation per dataset.")

new = ("for each dataset we used the mean MoE-GCN value across seeds and "
       "the corresponding mean GCN value as reported in its source table "
       "\u2014 a five-seed mean for the three MoleculeNet datasets and a "
       "three-seed mean for the nine TDC datasets, both under matched "
       "protocols (see Backbone architectures, above) \u2014 giving one "
       "paired observation per dataset.")

count = content.count(old)
print(f"Found {count} occurrence(s)")
assert count == 1, f"Expected exactly 1 match, found {count}"

content = content.replace(old, new)
open(path, "w", encoding="utf-8").write(content)
print("OK - file updated")
