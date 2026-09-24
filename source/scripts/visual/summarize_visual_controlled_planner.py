"""Prespecified planner contrasts; no model or search hyperparameter selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.runs);draws=np.random.default_rng(955002).integers(0,128,(10000,128));tables=[];contrasts=[];per_case={};reference_seeds=None;zero=None;bank_hash=None
for mi,index in enumerate(range(5)):
 costs={}
 for pi,(algorithm,parameterization) in enumerate([('random','held'),('random','full'),('cem','held'),('cem','full')]):
  run=root/f'job_{mi*4+pi}';r=json.loads((run/'summary.json').read_text());acc=json.loads((run/'acceptance.json').read_text());assert len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']<=1e-7
  assert r['model_index']==index and r['algorithm']==algorithm and r['parameterization']==parameterization
  for name,h in json.loads((run/'source/manifest.json').read_text()).items():assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==h
  seeds=[v['seed'] for v in r['cases']]
  if reference_seeds is None:reference_seeds=seeds;zero=np.array([v['zero_cost'] for v in r['cases']]);bank_hash=r['hashes']['bank_manifest']
  assert seeds==reference_seeds and r['hashes']['bank_manifest']==bank_hash;np.testing.assert_array_equal(zero,[v['zero_cost'] for v in r['cases']]);assert all(v['scored_candidates']==9000 and v['additional_verification_forwards']==1 for v in r['cases'])
  values=np.array([v['realized_cost'] for v in r['cases']]);name=algorithm+'_'+parameterization;costs[name]=values;per_case[f'{index}_{name}']=values.tolist();tables.append({'model_index':index,'arm':r['arm'],'score_space':r['score_space'],'algorithm':algorithm,'parameterization':parameterization,'mean_cost':float(values.mean()),'success':float(np.mean([v['success'] for v in r['cases']])),'mean_native_score':float(np.mean([v['predicted_cost'] for v in r['cases']])),'mean_boundary_steps':float(np.mean([v['boundary_steps'] for v in r['cases']])),'mean_planning_seconds':float(np.mean([v['planning_seconds'] for v in r['cases']]))})
 for left,right in [('cem_held','random_held'),('cem_full','random_full'),('random_full','random_held'),('cem_full','cem_held')]:
  delta=costs[left]-costs[right];contrasts.append({'model_index':index,'contrast':left+' minus '+right,'mean_difference':float(delta.mean()),'paired_95_percentile_interval':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()})
result={'scope':'previously examined scenarios; one training sample/seed; image-only open-loop controlled search; state arms use extra training labels; released reference uses a different training budget, not published LeWM reproduction','cases_per_route':128,'search_scores_per_case':9000,'additional_verification_forwards_per_case':1,'bank_manifest_sha256':bank_hash,'zero_mean_cost':float(zero.mean()),'tables':tables,'contrasts':contrasts,'per_case_costs':per_case}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Budget-matched controlled planners','',result['scope'],'','| Model | Search | Action coordinates | Realized cost | Success | Native score (within-model units) | Boundary steps |','|---|---|---|---:|---:|---:|---:|']
for v in tables:lines.append(f"| {v['arm']} | {v['algorithm']} | {v['parameterization']} | {v['mean_cost']:.2f} | {v['success']:.1%} | {v['mean_native_score']:.2f} | {v['mean_boundary_steps']:.2f} |")
lines.extend(['',f'Zero-action mean cost: {zero.mean():.2f}.','', 'Negative paired cost difference favors the first policy.',''])
for v in contrasts:lines.append(f"- Model {v['model_index']}, {v['contrast']}: {v['mean_difference']:.2f}, nominal paired 95% interval {v['paired_95_percentile_interval']}.")
out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
