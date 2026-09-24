"""Check deterministic full-history replay before simulator action-ranking studies."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();sys.path.insert(0,a.source)
from stable_worldmodel.envs.pusht.env import PushT
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
rows=[]
for seed in [730001,730002,730003,730004]:
 rng=np.random.default_rng(seed);prefix=rng.uniform(-.3,.3,(15,2)).astype('float32')
 def replay(branch):
  env=PushT(resolution=224);obs,info=env.reset(seed=seed)
  for action in prefix:obs,reward,term,trunc,info=env.step(action)
  start=obs['state'].copy();frame=env.render().copy();states=[]
  for action in branch:obs,reward,term,trunc,info=env.step(action);states.append(obs['state'].copy())
  env.close();return start,frame,np.array(states)
 branches=[np.tile(action,(25,1)).astype('float32') for action in [[-.3,0],[0,0],[.3,0]]]
 results=[replay(b) for b in branches];repeat=replay(branches[0])
 for item in results:np.testing.assert_array_equal(item[0],results[0][0])
 np.testing.assert_array_equal(results[0][2],repeat[2]);np.testing.assert_array_equal(results[0][1],repeat[1])
 assert results[0][1].shape==(224,224,3)
 Image.fromarray(results[0][1]).save(out/f'start_{seed}.png')
 np.savez_compressed(out/f'branches_{seed}.npz',start=results[0][0],states=np.array([v[2] for v in results]),actions=np.array(branches),prefix=prefix)
 rows.append({'seed':seed,'deterministic_replay':True,'branches':3,'steps':25})
(out/'summary.json').write_text(json.dumps({'purpose':'simulator plumbing only, not learned-model evaluation','rows':rows},indent=2)+'\n')
print('SIMULATOR_SMOKE_PASSED',flush=True)
