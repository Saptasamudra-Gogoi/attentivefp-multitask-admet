lines = open('ablation_routing.py', encoding='utf-8').readlines()
for i in range(445, 515):
    print(f'{i+1}: {lines[i]}', end='')