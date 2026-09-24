"""Held-out policy contrasts with scenario-paired uncertainty; no reselection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['runs','bank','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();root=Path(a.runs);bank=Path(a.bank);selection=json.loads((root/'frozen_selection.json').read_text());manifest=json.loads((bank/'manifest.json').read_text());assert len(manifest['cases'])==128
costs={};success={};model_rows={}
for index in selection['heldout_indices']:
 run=root/'heldout'/f'job_{index}';assert (run/'acceptance.json').exists();r=json.loads((run/'summary.json').read_text());assert len(r['cases'])==128
 assert r['hashes']['bank_manifest']==hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
 for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
 model_rows[index]=r
for name,policy in selection['policies'].items():
 index=policy['index'];r=model_rows[index];assert r['hashes']['weights']==policy['weights_sha256'];costs[name]=[];success[name]=[]
 for row,item in zip(r['cases'],manifest['cases']):
  assert row['index']==item['index'];i=item['index'];path=bank/f'case_{i:03d}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);pred=np.load(root/'heldout'/f'job_{index}'/f'case_{i:03d}_predictions.npz')
  states,goal=z['terminal_states'],z['goal_state'];distance=np.linalg.norm(states[:,2:4]-goal[2:4],axis=1);angle=np.angle(np.exp(1j*(states[:,4]-goal[4])));actual=distance**2+900*angle**2;ok=(distance<20)&(abs(angle)<np.pi/9)
  scores=pred['score_native_block_pose'];ties=np.flatnonzero(scores==scores.min());value=float(actual[ties].mean());np.testing.assert_allclose(value,row['tasks']['block_pose']['native']['selected_cost'],rtol=1e-9,atol=1e-7);costs[name].append(value);success[name].append(float(ok[ties].mean()))
  if name==next(iter(selection['policies'])):
   for baseline,values in [('zero',(float(actual[0]),float(ok[0]))),('uniform',(float(actual.mean()),float(ok.mean()))),('sample_oracle',(float(actual.min()),float(ok[actual==actual.min()].mean())))]:costs.setdefault(baseline,[]).append(values[0]);success.setdefault(baseline,[]).append(values[1])
costs={k:np.array(v) for k,v in costs.items()};assert all(len(v)==128 for v in costs.values());draws=np.random.default_rng(953001).integers(0,128,(10000,128));summary={}
for name,values in costs.items():summary[name]={'mean_cost':float(values.mean()),'mean_success':float(np.mean(success[name])),'mean_regret':float(np.mean(values-costs['sample_oracle'])),'gain_relative_to_aggregate_zero':float(1-values.mean()/costs['zero'].mean())}
contrasts={}
for left,right in [('recursive_expert','teacher_forced_expert'),('recursive_branch','teacher_forced_branch'),('teacher_forced_branch','teacher_forced_expert'),('recursive_branch','recursive_expert')]:
 delta=costs[left]-costs[right];contrasts[left+' minus '+right]={'mean_cost_difference':float(delta.mean()),'paired_95_percentile_interval':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
result={'scope':'one training sample and seed; paired intervals quantify held-out scenario uncertainty only','selection':selection,'cases':128,'summary':summary,'contrasts':contrasts,'per_case_costs':{k:v.tolist() for k,v in costs.items()},'bank_manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Prospective PushT selector comparison','',result['scope'],'','| Policy | Mean block cost | Success | Gain relative to zero |','|---|---:|---:|---:|']
for k,v in summary.items():lines.append(f"| {k} | {v['mean_cost']:.2f} | {v['mean_success']:.1%} | {v['gain_relative_to_aggregate_zero']:.1%} |")
lines.extend(['','Cost differences below: negative favors the first policy.',''])
for name,v in contrasts.items():lines.append(f"- {name}: {v['mean_cost_difference']:.2f}; paired 95% interval {v['paired_95_percentile_interval']}.")
out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
