"""Fresh paired-trajectory bank: action means fixed, within-group timing varied."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from paired_action_plans import paired_plans
p=argparse.ArgumentParser()
for k in ['source','output']:p.add_argument('--'+k,required=True)
p.add_argument('--cases',type=int,default=64);a=p.parse_args();sys.path.insert(0,a.source)
from stable_worldmodel.envs.pusht.env import PushT
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
for condition in ['held','varied']:(out/condition).mkdir()
rows={k:[] for k in ['held','varied']};rejected=[];pair_rows=[]
def visible(env):
 for body in [env.agent,env.block]:
  for shape in body.shapes:
   bb=shape.cache_bb()
   if min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504:return False
 return True
for seed in range(930001,932049):
 if len(pair_rows)==a.cases:break
 env=PushT(resolution=224);obs,_=env.reset(seed=seed);prefix=[]
 for _ in range(10):
  action=np.clip((obs['state'][2:4]+[0,65]-obs['state'][:2])/100,-.7,.7).astype('float32');prefix.append(action);obs,*_=env.step(action)
 env.close();prefix=np.array(prefix)
 def replay(plan):
  env=PushT(resolution=224);obs,_=env.reset(seed=seed);frames=[env.render().copy()];states=[obs['state'].copy()];future=[]
  if not visible(env):env.close();return None
  for t,action in enumerate(prefix):
   obs,*_=env.step(action)
   if not visible(env):env.close();return None
   if (t+1)%5==0:frames.append(env.render().copy());states.append(obs['state'].copy())
  contacts=0;terms=0
  for t,action in enumerate(plan):
   obs,_,term,_,_=env.step(action);contacts+=int(env.n_contact_points>0);terms+=int(term)
   if not visible(env):env.close();return None
   if (t+1)%5==0:future.append(obs['state'].copy())
  result={'history_pixels':np.array(frames),'history_states':np.array(states),'terminal_pixels':env.render().copy(),'terminal_states':obs['state'].copy(),'future_states':np.array(future),'contacts':contacts,'terminations':terms};env.close();return result
 zero=np.zeros((25,2),dtype='float32');first=replay(zero)
 if first is None:rejected.append({'seed':seed,'reason':'history_or_zero_visibility'});continue
 goal_rng=np.random.default_rng(np.random.SeedSequence([seed,1]));plan_rng=np.random.default_rng(np.random.SeedSequence([seed,2]));goal=None
 for ga in range(1,129):
  options=paired_plans(goal_rng);goal_type=int(goal_rng.integers(2));goal_plan=options[goal_type];goal=replay(goal_plan)
  if goal is not None and np.linalg.norm(goal['terminal_states'][2:4]-goal['history_states'][-1,2:4])>=40:break
  goal=None
 if goal is None:rejected.append({'seed':seed,'reason':'no_visible_displaced_goal'});continue
 plans={k:[zero] for k in rows};results={k:[first] for k in rows};attempts=[];failed=False
 for _ in range(16):
  pair=None
  for attempt in range(1,129):
   hp,vp=paired_plans(plan_rng);hr,vr=replay(hp),replay(vp)
   if hr is not None and vr is not None:pair=(hp,vp,hr,vr);break
  if pair is None:failed=True;break
  for k,plan,res in [('held',pair[0],pair[2]),('varied',pair[1],pair[3])]:plans[k].append(plan);results[k].append(res)
  attempts.append(attempt)
 if failed:rejected.append({'seed':seed,'reason':'paired_slot_exhausted'});continue
 repeat=replay(zero)
 for key in ['history_pixels','history_states','terminal_pixels','terminal_states','future_states']:np.testing.assert_array_equal(first[key],repeat[key])
 np.testing.assert_allclose(np.array(plans['held']).reshape(17,5,5,2).mean(2,dtype=float),np.array(plans['varied']).reshape(17,5,5,2).mean(2,dtype=float),rtol=0,atol=5e-8)
 index=len(pair_rows)
 for k in rows:
  for res in results[k]:
   for key in ['history_pixels','history_states']:np.testing.assert_array_equal(res[key],goal[key])
  path=out/k/f'case_{index:03d}.npz';np.savez_compressed(path,seed=seed,prefix=prefix,actions=np.array(plans[k]),goal_actions=goal_plan,goal_pixels=goal['terminal_pixels'],goal_state=goal['terminal_states'],history_pixels=goal['history_pixels'],history_states=goal['history_states'],**{key:np.array([r[key] for r in results[k]]) for key in ['terminal_pixels','terminal_states','future_states','contacts','terminations']})
  rows[k].append({'index':index,'seed':seed,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 pair_rows.append({'index':index,'seed':seed,'goal_attempts':ga,'goal_type':['held','varied'][goal_type],'pair_attempts':attempts});print(json.dumps(pair_rows[-1]),flush=True)
assert len(pair_rows)==a.cases
source_hashes={str(path.name):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('paired_action_plans.py'),Path(a.source)/'stable_worldmodel/envs/pusht/env.py']}
for k in rows:
 manifest={'cases':rows[k],'candidates':17,'condition':k,'scope':'paired action-timing mechanism diagnostic; not an official benchmark','source_hashes':source_hashes};(out/k/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/k/'COMPLETE').write_text('paired visibility and replay checked\n')
(out/'paired_manifest.json').write_text(json.dumps({'pairs':pair_rows,'rejected_seeds':rejected,'seed_start':930001,'max_seeds':2048,'source_hashes':source_hashes},indent=2)+'\n');(out/'COMPLETE').write_text('paired bank complete\n')
