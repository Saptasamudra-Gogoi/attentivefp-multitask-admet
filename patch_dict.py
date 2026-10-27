old_file = "run_all_specialization_full22.py"

new_dict = '''MAE_DATASETS = {
    "solubility_aqsoldb": {
        "tdc_name": "Solubility_AqSolDB",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "solubility_aqsoldb",
    },
    "caco2_wang": {
        "tdc_name": "Caco2_Wang",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "caco2_wang",
    },
    "lipophilicity_astrazeneca": {
        "tdc_name": "Lipophilicity_AstraZeneca",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "lipophilicity_astrazeneca",
    },
    "ld50_zhu": {
        "tdc_name": "LD50_Zhu",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "ld50_zhu",
    },
    "ppbr_az": {
        "tdc_name": "PPBR_AZ",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "ppbr_az",
    },
    "ames": {
        "tdc_name": "AMES",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "ames",
    },
    "bbb_martins": {
        "tdc_name": "BBB_Martins",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "bbb_martins",
    },
    "bioavailability_ma": {
        "tdc_name": "Bioavailability_Ma",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "bioavailability_ma",
    },
    "clearance_hepatocyte_az": {
        "tdc_name": "Clearance_Hepatocyte_AZ",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "clearance_hepatocyte_az",
    },
    "clearance_microsome_az": {
        "tdc_name": "Clearance_Microsome_AZ",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "clearance_microsome_az",
    },
    "cyp2c9_substrate_carbonmangels": {
        "tdc_name": "CYP2C9_Substrate_CarbonMangels",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp2c9_substrate_carbonmangels",
    },
    "cyp2c9_veith": {
        "tdc_name": "CYP2C9_Veith",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp2c9_veith",
    },
    "cyp2d6_substrate_carbonmangels": {
        "tdc_name": "CYP2D6_Substrate_CarbonMangels",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp2d6_substrate_carbonmangels",
    },
    "cyp2d6_veith": {
        "tdc_name": "CYP2D6_Veith",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp2d6_veith",
    },
    "cyp3a4_substrate_carbonmangels": {
        "tdc_name": "CYP3A4_Substrate_CarbonMangels",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp3a4_substrate_carbonmangels",
    },
    "cyp3a4_veith": {
        "tdc_name": "CYP3A4_Veith",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "cyp3a4_veith",
    },
    "dili": {
        "tdc_name": "DILI",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "dili",
    },
    "half_life_obach": {
        "tdc_name": "Half_Life_Obach",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "half_life_obach",
    },
    "herg": {
        "tdc_name": "hERG",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "herg",
    },
    "hia_hou": {
        "tdc_name": "HIA_Hou",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "hia_hou",
    },
    "pgp_broccatelli": {
        "tdc_name": "Pgp_Broccatelli",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "pgp_broccatelli",
    },
    "vdss_lombardo": {
        "tdc_name": "VDss_Lombardo",
        "result_file": "results_moegcn_tdc_v2.json",
        "result_key": "vdss_lombardo",
    },
}'''

with open(old_file, encoding="utf-8", errors="replace") as f:
    content = f.read()

start = content.find("MAE_DATASETS = {")
end = content.find("\n}\n", start) + 3

old_dict = content[start:end]
patched = content.replace(old_dict, new_dict)

with open(old_file, "w", encoding="utf-8") as f:
    f.write(patched)

print("Patched. Old dict length:", len(old_dict), "New dict length:", len(new_dict))
print("File size now:", len(patched))