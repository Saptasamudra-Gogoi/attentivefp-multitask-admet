import io
p = r"D:\molprop_project\attentivefp-multitask-admet\moedmpnn_regr.py"
with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

print(f"File: {p}\n")
# Check for backbone markers
for marker in ["GCNConv", "NNConv", "GRUCell", "true_dmpnn", "backbone"]:
    count = content.count(marker)
    print(f"  '{marker}' appears {count}x")

print("\n--- Lines mentioning Conv/GRU (context) ---")
for i, line in enumerate(content.splitlines(), 1):
    if any(k in line for k in ["GCNConv", "NNConv", "GRUCell", "class ", "backbone"]):
        print(f"  L{i}: {line.strip()}")
