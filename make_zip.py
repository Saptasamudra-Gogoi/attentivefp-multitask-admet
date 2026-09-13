import zipfile, os

exclude = {'checkpoints','models','grover_data','tdc_data','entropy_results','temp'}
exclude_paths = ['baselines\\chemprop', 'baselines/chemprop']

src = 'D:\\molprop_project'
out = 'D:\\molprop_v2.zip'

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        skip = False
        for e in exclude:
            if e in root:
                skip = True
                break
        for ep in exclude_paths:
            if ep in root:
                skip = True
                break
        if skip:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            filepath = os.path.join(root, f)
            arcname = os.path.join('molprop_project', rel_root, f)
            z.write(filepath, arcname)

print('Done! Saved to D:\\molprop_v2.zip')
