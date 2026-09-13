import json, os, io
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\molprop_project")
OUT = io.open("evidence_report_round3.txt", "w", encoding="utf-8")

def w(s=""):
    print(s)
    OUT.write(str(s) + "\n")

def ts(p):
    try:
        return datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "N/A"

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}

def section(t):
    w("\n" + "=" * 70); w(t); w("=" * 70)

section("2.1 (final) - full content of pharma_moe_gcn.py")
p = ROOT / "pharma_moe_gcn.py"
if p.exists():
    w(f"[{ts(p)}] {p}\n")
    with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
        w(f.read())
else:
    w("** FILE NOT FOUND **")

section("2.2 (final) - caco2_wang inside results_moegcn_tdc_v2.json")
data = load_json(ROOT / "results_moegcn_tdc_v2.json")
if "__error__" not in data and "caco2_wang" in data:
    w(json.dumps(data["caco2_wang"], indent=2))
else:
    w(f"Not found or error: {data.get('__error__')}")

section("2.7-part-2 - AttentiveFP: local run vs final_results_table.csv")
data = load_json(ROOT / "results_attentivefp.json")
if "__error__" not in data:
    w(json.dumps(data, indent=2))
else:
    w(f"Error: {data['__error__']}")

w("\nCompare against final_results_table.csv AttentiveFP_mean/std columns:")
w("BBBP=0.908/0.05  BACE=0.852/0.053  Tox21=0.7458/0.0049  ToxCast=0.6746/0.0029")
w("SIDER=0.5937/0.0062  ClinTox=0.7294/0.1006  HIV=0.7152/0.0074")
w("ESOL=1.0606/0.0295  FreeSolv=2.5729/0.1272  Lipo=0.6849/0.015")

OUT.close()
