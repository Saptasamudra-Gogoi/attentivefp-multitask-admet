import subprocess

datasets = [
    "ames", "bbb_martins", "bioavailability_ma", "clearance_hepatocyte_az",
    "clearance_microsome_az", "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith", "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith", "dili", "half_life_obach", "herg", "hia_hou",
    "pgp_broccatelli", "vdss_lombardo",
]

for ds in datasets:
    ckpt = f"checkpoints\\{ds}_seed0.pt"
    print(f"=== Running {ds} ===")
    result = subprocess.run(
        ["python", "expert_specialization_stats.py", "--dataset", ds, "--checkpoint", ckpt],
        capture_output=True, text=True
    )
    print(result.stdout[-500:])
    if "demo" in result.stdout.lower() or "Checkpoint not found" in result.stdout:
        print(f"!!! WARNING: {ds} may have fallen back to demo mode !!!")
    if result.returncode != 0:
        print(f"!!! FAILED: {ds} — {result.stderr[-500:]}")