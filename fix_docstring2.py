path = 'ablation_routing.py'
content = open(path, encoding='utf-8').read()

old_block = """  To confirm that performance gains arise from learned routing"""
# We need the full paragraph — print it first to get exact text
lines = content.split('\n')
for i in range(460, 480):
    if i < len(lines):
        print(f"{i}: {lines[i]}")