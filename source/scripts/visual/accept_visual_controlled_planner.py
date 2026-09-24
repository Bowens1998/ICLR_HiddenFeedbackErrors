"""Independent image-score reconstruction and model-free physical replay."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['run','bank','model-manifest']:p.add_argument('--'+k,required=True)
p.add_argument('--simulator');p.add_argument('--expected-cases',type=int,default=128);a=p.parse_args();run=Path(a.run);bank=Path(a.bank);r=json.loads((run/'summary.json').read_text());manifest=json.loads((bank/'manifest.json').read_text());selection=json.loads(Path(a.model_manifest).read_text())
assert (run/'COMPLETE').exists() and len(r['cases'])==a.expected_cases
assert r['hashes']['weights']==selection['models'][r['model_index']]['weights_sha256'];assert r['hashes']['bank_manifest']==hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
if a.simulator:
 sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
assert r['hashes']['model_manifest']==hashlib.sha256(Path(a.model_manifest).read_bytes()).hexdigest()
assert r['arm']==selection['models'][r['model_index']]['arm']
if r['score_space']=='state':mn=np.array(r['target_normalization']['mean']);sd=np.array(r['target_normalization']['std'])
max_replay=0.;max_score=0.
for row,item in zip(r['cases'],manifest['cases']):
 i=row['index'];assert i==item['index'] and row['seed']==item['seed'];path=bank/f'case_{i:03d}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];truth=np.load(path);z=np.load(run/f'case_{i:03d}_predictions.npz')
 dim=2 if r['parameterization']=='held' else 10;params=z['parameters'];costs=z['costs'];assert params.shape==(30,300,5,dim) and costs.shape==(30,300) and z['tokens'].shape==(30,300,6 if r['score_space']=='state' else 192);assert np.isfinite(costs).all() and np.max(abs(params))<=np.float32(.35)
 goal=truth['goal_state'];tokens=z['tokens'].astype(float);goal_token=z['goal_tokens'].astype(float)
 if r['score_space']=='latent':rebuilt=np.sum((tokens-goal_token)**2,-1);atol=1e-7
 else:
  state=tokens*sd+mn;estimated_goal=goal_token*sd+mn;angle=np.arctan2(state[...,4],state[...,5])-np.arctan2(estimated_goal[4],estimated_goal[5]);angle=(angle+np.pi)%(2*np.pi)-np.pi;rebuilt=np.sum((state[...,2:4]-estimated_goal[2:4])**2,-1)+900*angle**2;atol=2e-3
 np.testing.assert_allclose(rebuilt,costs,rtol=5e-5,atol=atol);max_score=max(max_score,float(np.max(abs(rebuilt-costs))))
 mean=np.zeros((5,dim));std=np.full((5,dim),.2)
 for t in range(30):
  np.testing.assert_allclose(z['mean'][t],mean,rtol=3e-5,atol=2e-7);np.testing.assert_allclose(z['std'][t],std,rtol=3e-5,atol=2e-7);np.testing.assert_array_equal(params[t,0],z['mean'][t])
  if r['algorithm']=='cem':
   elites=params[t,np.argsort(costs[t],kind='stable')[:30]].astype(float);mean=elites.mean(0);std=np.maximum(elites.std(0),1e-3)
 np.testing.assert_allclose(z['final_distribution_mean'],mean,rtol=3e-5,atol=2e-7)
 ti,ci=np.unravel_index(np.argmin(costs),costs.shape);assert (ti,ci)==(row['selected_iteration'],row['selected_candidate']);assert float(costs[ti,ci])==row['predicted_cost'];chosen=params[ti,ci]
 actions=np.repeat(chosen,5,axis=0) if dim==2 else chosen.reshape(25,2);np.testing.assert_array_equal(actions,z['selected_actions']);assert row['scored_candidates']==9000 and row['additional_verification_forwards']==1
 def physical(s):
  d=np.linalg.norm(s[2:4]-goal[2:4]);angle=(s[4]-goal[4]+np.pi)%(2*np.pi)-np.pi;return float(d*d+900*angle*angle),bool(d<20 and abs(angle)<np.pi/9)
 for label,state in [('realized',z['selected_states'][-1]),('zero',truth['terminal_states'][0])]:
  c,ok=physical(state);np.testing.assert_allclose(c,row[label+'_cost'],rtol=1e-9,atol=1e-7);assert ok==row['success' if label=='realized' else 'zero_success']
 assert z['selected_states'].shape==(26,7) and int(z['boundary'].sum())==row['boundary_steps']
 if a.simulator:
  env=PushT(resolution=224);obs,_=env.reset(seed=int(row['seed']));np.testing.assert_array_equal(obs['state'],truth['history_states'][0]);hi=1
  for t,action in enumerate(truth['prefix']):
   obs,*_=env.step(action)
   if (t+1)%5==0:np.testing.assert_array_equal(obs['state'],truth['history_states'][hi]);hi+=1
  replay=[obs['state'].copy()];boundary=[]
  for action in actions:
   obs,*_=env.step(action);replay.append(obs['state'].copy());outside=False
   for body in [env.agent,env.block]:
    for shape in body.shapes:
     bb=shape.cache_bb();outside|=min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504
   boundary.append(outside)
  replay=np.array(replay);np.testing.assert_allclose(replay,z['selected_states'],rtol=0,atol=1e-7);max_replay=max(max_replay,float(np.max(abs(replay-z['selected_states']))));np.testing.assert_array_equal(env.render(),z['terminal_pixels']);np.testing.assert_array_equal(boundary,z['boundary']);env.close()
result={'cases':len(r['cases']),'candidate_scores_per_case':9000,'scores_CEM_updates_choices_actions_metrics_hashes':'independently reconstructed','max_absolute_float32_score_reconstruction_difference':max_score,'model_free_simulator_replay':bool(a.simulator),'max_replay_state_difference':max_replay if a.simulator else None,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(run/('acceptance.json' if a.simulator else 'numeric_acceptance.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
