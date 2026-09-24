"""Freeze four prespecified policies using validation outcomes only."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['training','validation','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();training=Path(a.training);validation=Path(a.validation);expert=json.loads((training/'selection.json').read_text());rows=[]
for i in range(6):
 run=validation/f'job_{i}';assert (run/'acceptance.json').exists();r=json.loads((run/'summary.json').read_text());weight=training/f'job_{i}'/'best_weights.pt';h=hashlib.sha256(weight.read_bytes()).hexdigest();assert h==r['hashes']['weights']==expert['all_candidates'][i]['weights_sha256'];assert len(r['cases'])==32
 rows.append({'index':i,'validation_cost':r['aggregate']['block_pose']['native']['selected_cost'],'weights_sha256':h,'validation_summary_sha256':hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()})
assert len({json.loads((validation/f'job_{i}'/'summary.json').read_text())['hashes']['bank_manifest'] for i in range(6)})==1
policies={}
for mode,start in [('teacher_forced',0),('recursive',3)]:
 indices={'expert':expert['selected'][mode]['index'],'branch':min(rows[start:start+3],key=lambda v:(v['validation_cost'],v['index']))['index']}
 for selector,index in indices.items():policies[mode+'_'+selector]={'index':index,'weights_sha256':rows[index]['weights_sha256']}
result={'scope':'frozen before held-out evaluation; privileged-state inference; one training sample/seed','policies':policies,'all_validation_candidates':rows,'heldout_indices':sorted({p['index'] for p in policies.values()}),'selection_label_budget':32*32}
with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
