"""Build fixed-horizon, full-prefix-replayed PushT candidate banks."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',type=int,default=32);p.add_argument('--candidates',type=int,default=32);a=p.parse_args()
    sys.path.insert(0,a.source)
    from stable_worldmodel.envs.pusht.env import PushT
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    for index in range(a.cases):
        seed=870001+index;rng=np.random.default_rng(seed)
        env=PushT(resolution=224);obs,_=env.reset(seed=seed);prefix=[]
        for _ in range(10):
            # Approach from below; fixed scenario construction, not model input.
            target=obs['state'][2:4]+np.array([0.,65.])
            action=np.clip((target-obs['state'][:2])/100.,-.7,.7).astype('float32')
            prefix.append(action);obs,*_=env.step(action)
        env.close();prefix=np.array(prefix)
        plans=np.repeat(rng.uniform(-.7,.7,(a.candidates,5,2)),5,axis=1).astype('float32');plans[0]=0
        goal_plan=np.repeat(rng.uniform(-.7,.7,(5,2)),5,axis=0).astype('float32')
        def replay(plan):
            e=PushT(resolution=224);o,_=e.reset(seed=seed);frames=[e.render().copy()];states=[o['state'].copy()]
            for t,action in enumerate(prefix):
                o,*_=e.step(action)
                if (t+1)%5==0:frames.append(e.render().copy());states.append(o['state'].copy())
            contacts=0;terminations=0
            for action in plan:
                o,_,term,_,_=e.step(action);contacts+=int(e.n_contact_points>0);terminations+=int(term)
            result=(np.array(frames),np.array(states),e.render().copy(),o['state'].copy(),contacts,terminations)
            e.close();return result
        goal=replay(goal_plan);results=[replay(plan) for plan in plans];repeat=replay(plans[0])
        for result in results+[repeat]:
            np.testing.assert_array_equal(result[0],goal[0]);np.testing.assert_array_equal(result[1],goal[1])
        np.testing.assert_array_equal(results[0][2],repeat[2]);np.testing.assert_array_equal(results[0][3],repeat[3])
        path=out/f'case_{index:03d}.npz'
        np.savez_compressed(path,seed=seed,history_pixels=goal[0],history_states=goal[1],prefix=prefix,actions=plans,goal_actions=goal_plan,
                            goal_pixels=goal[2],goal_state=goal[3],terminal_pixels=np.array([r[2] for r in results]),terminal_states=np.array([r[3] for r in results]),
                            contacts=np.array([r[4] for r in results]),terminations=np.array([r[5] for r in results]))
        rows.append({'index':index,'seed':seed,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contact_candidates':sum(r[4]>0 for r in results),
                     'moving_block_candidates':int(sum(np.linalg.norm(r[3][2:4]-goal[1][-1,2:4])>1e-6 for r in results))})
        print(json.dumps(rows[-1]),flush=True)
    manifest={'scope':'development only; privileged shared approach prefix','cases':rows,'candidates':a.candidates,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'simulator_env_sha256':hashlib.sha256((Path(a.source)/'stable_worldmodel/envs/pusht/env.py').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'COMPLETE').write_text('candidate bank only\n')
if __name__=='__main__':main()
