"""
inspect_tdc.py
--------------
Deeply inspects your TDC result files to show exact structure,
keys, and what model names / scores are actually stored.

Run from your project root:
    python inspect_tdc.py
"""

import json
from pathlib import Path

TARGET_FILES = [
    "results_tdc.json",
    "results_phase3_fixed.json",
    "results_moegcn_classif.json",
    "results_moegcn_regr.json",
    "results_dmpnn_classif.json",
    "results_dmpnn_regr.json",
    "results_moedmpnn_classif.json",
    "results_moedmpnn_regr.json",
    "results_attentivefp.json",
]

def pretty_val(v, depth=0):
    indent = "  " * depth
    if isinstance(v, dict):
        if len(v) == 0:
            return "{}"
        lines = []
        for k, val in list(v.items())[:8]:  # show up to 8 keys
            lines.append(f"{indent}  {repr(k)}: {pretty_val(val, depth+1)}")
        if len(v) > 8:
            lines.append(f"{indent}  ... ({len(v) - 8} more keys)")
        return "{\n" + "\n".join(lines) + f"\n{indent}}}"
    elif isinstance(v, list):
        if len(v) == 0:
            return "[]"
        sample = pretty_val(v[0], depth+1)
        return f"[{len(v)} items, first: {sample}]"
    elif isinstance(v, float):
        return f"{v:.4f}"
    else:
        return repr(v)

def inspect_file(path: Path):
    print(f"\n{'='*60}")
    print(f"  FILE: {path.name}")
    print(f"{'='*60}")

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ Could not load: {e}")
        return

    print(f"  Type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"  Top-level keys ({len(data)}): {list(data.keys())}\n")
        print("  Structure:")
        print(pretty_val(data, depth=1))

        # Try to find numeric leaf values and report them
        print("\n  ── Numeric values found (dataset/key → value) ──")
        found = []
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                found.append((path, obj))
        walk(data)
        for p, v in found[:40]:
            print(f"    {p:<60} = {v:.4f}" if isinstance(v, float) else f"    {p:<60} = {v}")
        if len(found) > 40:
            print(f"    ... ({len(found) - 40} more numeric values)")

    elif isinstance(data, list):
        print(f"  Length: {len(data)}")
        if data:
            print(f"  First item type: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                print(f"  First item keys: {list(data[0].keys())}")
                print(f"  First 3 items:")
                for item in data[:3]:
                    print(f"    {item}")

def main():
    root = Path(".")
    found_any = False
    for fname in TARGET_FILES:
        p = root / fname
        if p.exists():
            inspect_file(p)
            found_any = True

    if not found_any:
        print("None of the target files found in current directory.")
        print("Files present in root:")
        for f in sorted(root.glob("*.json")):
            print(f"  {f.name}")

if __name__ == "__main__":
    main()
