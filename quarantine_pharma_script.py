import shutil

old_name = 'cross_arch_ablation.py'
new_name = 'DEPRECATED_pharma_cross_arch_ablation.py'

shutil.move(old_name, new_name)

content = open(new_name, encoding='utf-8').read()
banner = '''"""
================================================================================
DEPRECATED — DO NOT USE FOR CURRENT MANUSCRIPT
================================================================================
This script tests Pharma-MoE (pharmacophore-guided routing) vs standard MoE
on GIN and DMPNN. This is the OLD pharmacophore-guided story, which has been
fully replaced by the unsupervised-routing narrative in the current manuscript
(v4). The current paper makes NO pharmacophore claims anywhere.

Do not use cross_arch_results.json (output of this script) for any figure
or claim in the current manuscript. GIN transferability for the CURRENT
(non-pharma) MoE plug-in has never been tested — see Fig. 4 honesty note.

Kept for archival/reference only. Renamed from cross_arch_ablation.py on
'''
import datetime
banner += datetime.date.today().isoformat() + '.\\n' + '"""\\n\\n'

content = content.replace('"""\nCross-Architecture Ablation: Pharma-MoE on GIN and DMPNN\n"""',
                           banner.rstrip() + '\n\n"""\nCross-Architecture Ablation: Pharma-MoE on GIN and DMPNN (ORIGINAL DOCSTRING, DEPRECATED)\n"""')

open(new_name, 'w', encoding='utf-8').write(content)
print(f"Renamed {old_name} -> {new_name} and added deprecation banner.")