python -c "
import json
with open('sss_results_final.json') as f:
    d = json.load(f)
collapsed = [k for k, v in d.items() if isinstance(v, dict) and v.get('source') == 'collapsed']
print('Collapsed datasets:', collapsed)
print('Count:', len(collapsed))
print('Total top-level dataset keys:', len([k for k,v in d.items() if isinstance(v, dict) and 'sss' in v]))
"