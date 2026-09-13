import sys, os, json
from pathlib import Path

print("="*60)
print("ENVIRONMENT")
print("="*60)
print(f"Python: {sys.version}")

# Core libs
libs = ["torch", "torch_geometric", "torch_scatter", "rdkit", "optuna",
        "sklearn", "numpy", "pandas", "deepchem"]
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "ok")
        print(f"  ✓ {lib}: {ver}")
    except ImportError:
        print(f"  ✗ {lib}: NOT INSTALLED")

# CUDA
try:
    import torch
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
except: pass

print("\n" + "="*60)
print("PROJECT FILES")
print("="*60)

# Search common locations
search_dirs = [
    Path("D:/molprop_project"),
    Path("C:/molprop_project"),
    Path.home() / "molprop_project",
    Path.home() / "Desktop" / "molprop_project",
    Path("."),
]

project_dir = None
for d in search_dirs:
    if d.exists():
        project_dir = d
        print(f"Found project at: {d}")
        break

if project_dir:
    for f in sorted(project_dir.rglob("*.py"))[:20]:
        print(f"  {f.relative_to(project_dir)}")
    
    print("\nRESULTS:")
    results_dir = project_dir / "results"
    if results_dir.exists():
        files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.csv"))
        if files:
            for f in sorted(files):
                size = f.stat().st_size
                print(f"  {f.name} ({size/1024:.1f} KB)")
                # Peek inside JSON
                if f.suffix == ".json" and size < 5_000_000:
                    try:
                        with open(f) as jf:
                            data = json.load(jf)
                        # Show top-level keys
                        print(f"    Keys: {list(data.keys())[:5]}")
                        # If nested, show depth
                        for k, v in list(data.items())[:2]:
                            if isinstance(v, dict):
                                print(f"    [{k}]: {list(v.keys())[:4]}")
                    except: pass
        else:
            print("  (no result files yet)")
    else:
        print("  No results/ folder found")
else:
    print("Project directory not found. Current dir contents:")
    for f in Path(".").iterdir():
        print(f"  {f}")

print("\n" + "="*60)
print("LAST ERROR CHECK")
print("="*60)
# Try importing the main runner
if project_dir:
    runner = project_dir / "baseline_runner_complete.py"
    runner2 = project_dir / "baseline_runner.py"
    for r in [runner, runner2]:
        if r.exists():
            print(f"Found: {r.name}")
            try:
                with open(r) as f:
                    lines = f.readlines()
                print(f"  Lines: {len(lines)}")
                # Show imports
                imports = [l.strip() for l in lines if l.startswith("import") or l.startswith("from")]
                print(f"  Imports ({len(imports)}):")
                for imp in imports[:10]:
                    print(f"    {imp}")
            except Exception as e:
                print(f"  Error reading: {e}")

print("\n✅ Diagnosis complete. Paste this output back.")
