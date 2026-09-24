"""Bind all48 engineering routes before the full PushT development run."""
import argparse,hashlib,json
from pathlib import Path


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['runs','plan','official','config','simulator','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();plan=json.loads(Path(a.plan).read_text());assert plan['layout']=='pusht_nonlinear_pose'
 assert len(plan['models'])==24 and len(plan['routes'])==48
 rows=[];common=None
 for index,route in enumerate(plan['routes']):
  d=Path(a.runs)/f'job_{index}';r=json.loads((d/'summary.json').read_text());acc=json.loads((d/'acceptance.json').read_text())
  assert (d/'COMPLETE').exists() and len(r['cases'])==acc['cases']==2
  assert acc['model_free_simulator_replay'] and acc['max_replay_state_difference']<=1e-7
  assert acc['verifier_sha256']==sha(Path(__file__).with_name('accept_pose_planner.py'))
  assert r['route_index']==index and r['model_index']==route['model_index']
  assert r['algorithm']==route['algorithm'] and r['parameterization']==route['parameterization']=='full'
  e=plan['models'][route['model_index']]
  assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['weights']==e['weights_sha256']
  assert r['hashes']['bank_manifest']==plan['bank_manifest_sha256']
  assert r['score_space']==e['score'] and r['checkpoint']==e['checkpoint']=='last'
  seeds=[x['seed'] for x in r['cases']]
  if common is None:common=seeds
  assert seeds==common
  for name,h in json.loads((d/'artifact_manifest.json').read_text()).items():assert sha(d/name)==h
  sources=json.loads((d/'source/manifest.json').read_text())
  for name,h in sources.items():
   assert sha(d/'source'/name)==h
   if name in ['jepa.py','module.py']:current=Path(a.official)/name
   elif name=='env.py':current=Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py'
   elif name==Path(a.config).name:current=Path(a.config)
   else:current=Path(__file__).with_name(name)
   assert sha(current)==h,name
  rows.append(dict(index=index,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),source_manifest_sha256=sha(d/'source/manifest.json')))
 result=dict(status='PASS',rows=rows,plan_sha256=sha(a.plan),engineering_seeds=common,verifier_sha256=sha(__file__),
             runner_sha256=sha(Path(__file__).with_name('run_pose_planner.py')),planner_verifier_sha256=sha(Path(__file__).with_name('accept_pose_planner.py')),
             scope='All48 engineering routes with two goals: artifact/source/model binding and completed full-score reconstruction/physics acceptance; no scientific outcome.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('PASS_ALL48_ENGINEERING')


if __name__=='__main__':main()
