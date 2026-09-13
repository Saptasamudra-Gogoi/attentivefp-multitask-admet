lines = open('cross_arch_ablation.py', encoding='utf-8').readlines()
for i, l in enumerate(lines[:20]):
    print(f'{i+1}: {l}', end='')