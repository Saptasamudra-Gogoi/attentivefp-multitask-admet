import os
import zipfile

exclude_dirs = {'.git', '__pycache__', 'data', 'log', 'log_backup', '.idea', 'grover', 'grover_data'}
exclude_exts = {'.pt', '.ckpt', '.pdb', '.mol2', '.smi', '.csv', '.zip', '.npz'}

with zipfile.ZipFile('code_project.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if os.path.splitext(f)[1] in exclude_exts:
                continue
            path = os.path.join(root, f)
            zf.write(path, path)

print("done: code_project.zip")