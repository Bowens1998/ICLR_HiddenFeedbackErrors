"""Require all48 accepted development routes before comparing PushT scores."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pose_planning_statistics import analyze,INTERFACES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['runs','plan','engineering-gate','bank-manifest','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();plan=json.loads(Path(a.plan).read_text());gate=json.loads(Path(a.engineering_gate).read_text());bank=json.loads(Path(a.bank_manifest).read_text())
 assert plan['layout']=='pusht_nonlinear_pose' and len(plan['models'])==24 and len(plan['routes'])==48
 assert gate['status']=='PASS' and len(gate['rows'])==48 and gate['plan_sha256']==sha(a.plan)
 assert plan['bank_manifest_sha256']==sha(a.bank_manifest) and len(bank['cases'])==128
 seeds=[v['seed'] for v in bank['cases']];assert len(set(seeds))==128
 assert gate['runner_sha256']==sha(Path(__file__).with_name('run_pose_planner.py'))
 assert gate['planner_verifier_sha256']==sha(Path(__file__).with_name('accept_pose_planner.py'))
 costs=[];success=[];rows=[];bindings=[]
 for i,route in enumerate(plan['routes']):
  d=Path(a.runs)/f'job_{i}';r=json.loads((d/'summary.json').read_text());acc=json.loads((d/'acceptance.json').read_text());art=json.loads((d/'artifact_manifest.json').read_text())
  assert (d/'COMPLETE').exists() and art['summary.json']==sha(d/'summary.json')
  assert acc['cases']==len(r['cases'])==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']<=1e-7
  assert acc['verifier_sha256']==gate['planner_verifier_sha256']
  source=json.loads((d/'source/manifest.json').read_text())
  assert source['run_pose_planner.py']==gate['runner_sha256']
  for name in ['controlled_search.py','nonlinear_pose_cost.py','image_planner_cost.py','factorial_model.py','lewm_adapter.py','evaluation_precision.py']:
   assert source[name]==sha(Path(__file__).with_name(name)),name
  mi=route['model_index'];entry=plan['models'][mi];rep,arch=divmod(i//8,2)
  assert (entry['replica'],entry['backbone_index'],entry['score'],route['algorithm'])==(rep,rep*2+arch,INTERFACES[(i%8)//2],['random','cem'][i%2])
  assert r['route_index']==i and r['model_index']==mi and r['score_space']==entry['score'] and r['arm']==entry['arm']
  assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['weights']==entry['weights_sha256'] and r['hashes']['bank_manifest']==sha(a.bank_manifest)
  assert r['checkpoint']==entry['checkpoint']=='last' and r['algorithm']==route['algorithm'] and r['parameterization']=='full'
  assert [v['seed'] for v in r['cases']]==seeds and [v['index'] for v in r['cases']]==[v['index'] for v in bank['cases']]
  assert all(v['scored_candidates']==9000 for v in r['cases'])
  c=[v['realized_cost'] for v in r['cases']];s=[v['success'] for v in r['cases']];costs.append(c);success.append(s)
  rows.append(dict(route_index=i,replica=rep,arch=arch,score=entry['score'],algorithm=route['algorithm'],mean_cost=float(np.mean(c)),success_rate=float(np.mean(s)),cost=c,success=s))
  bindings.append(dict(index=i,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),artifact_manifest_sha256=sha(d/'artifact_manifest.json')))
 result=dict(status='COMPLETE_DEVELOPMENT',goal_seeds=seeds,rows=rows,analysis=analyze(costs,success),bindings=bindings,
             plan_sha256=sha(a.plan),engineering_gate_sha256=sha(a.engineering_gate),bank_manifest_sha256=sha(a.bank_manifest),
             source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['summarize_pose_planning.py','pose_planning_statistics.py']},
             scope='Previously examined128-goal development bank; conditional paired-goal estimates over fixed three training pools. No fresh-bank confirmation, training-population inference, cross-task dimensional cost pooling, or method novelty claim. Raw trace verification was remote; local metadata alone does not re-verify traces.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('COMPLETE48_ROUTES_190_CONTRASTS')


if __name__=='__main__':main()
