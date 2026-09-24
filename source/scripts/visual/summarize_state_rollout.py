import json
from pathlib import Path
training=Path('runs/state_rollout_full_v1');selection=json.loads((training/'selection.json').read_text());rows=[]
for i in range(6):
 r=json.loads((training/f'job_{i}'/'summary.json').read_text());row={'index':i,'mode':r['mode'],'lr':r['base_lr'],'epoch':r['best_epoch'],'validation_mse':r['best_validation_mse'],'selected':selection['selected'][r['mode']]['index']==i}
 for cond in ['inview','stress']:
  run=Path('runs/state_rollout_eval_v1')/cond/f'job_{i}';assert (run/'acceptance.json').exists();s=json.loads((run/'summary.json').read_text());assert s['hashes']['weights']==selection['all_candidates'][i]['weights_sha256'];row[cond]=s['aggregate']['block_pose']['native'];row[cond+'_zero']=s['aggregate']['block_pose']['zero_action']['selected_cost']
 rows.append(row)
result={'scope':'one data sample/seed; existing development banks; no post-hoc task checkpoint selection','rows':rows};out=Path('outputs/maintrack/state_rollout_summary');out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Matched rollout supervision: development results','','| Mode | LR | Selected by validation | Best epoch | Recursive validation MSE | In-view cost | Stress cost |','|---|---:|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['mode']} | {r['lr']} | {r['selected']} | {r['epoch']} | {r['validation_mse']:.6f} | {r['inview']['selected_cost']:.2f} | {r['stress']['selected_cost']:.2f} |")
lines.extend(['',f"Zero-action costs: in-view {rows[0]['inview_zero']:.2f}; stress {rows[0]['stress_zero']:.2f}.",'',result['scope']]);out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
