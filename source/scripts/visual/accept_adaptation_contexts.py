"""Independent full replay of new adaptation context/reference branches."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ['bank','simulator']:p.add_argument('--'+k,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--seed-start',type=int,required=True);a=p.parse_args()
    sys.path.insert(0,a.simulator)
    from stable_worldmodel.envs.pusht.env import PushT
    bank=Path(a.bank);m=json.loads((bank/'manifest.json').read_text());assert len(m['cases'])==a.count and m['seed_start']==a.seed_start
    assert m['script_sha256']==sha(Path(__file__).with_name('prepare_prospective_pusht.py')) and m['simulator_env_sha256']==sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py')
    rows=[]
    for i,item in enumerate(m['cases']):
        path=bank/f'case_{i:03d}.npz';assert item['index']==i and sha(path)==item['sha256'];z=dict(np.load(path));seed=int(z['seed'])
        assert seed==item['seed'] and a.seed_start<=seed<a.seed_start+m['max_seeds']
        assert z['actions'].shape==(32,25,2) and not z['actions'][0].any()
        assert np.max(abs(z['actions']))<=.35 and np.max(abs(z['goal_actions']))<=.35
        for branch in range(33):
            goal=branch==32;actions=z['goal_actions'] if goal else z['actions'][branch]
            env=PushT(resolution=224);obs,_=env.reset(seed=seed);frames=[env.render().copy()];states=[obs['state'].copy()]
            def visible():
                for body in [env.agent,env.block]:
                    for shape in body.shapes:
                        bb=shape.cache_bb();assert min(bb.left,bb.bottom)>=8 and max(bb.right,bb.top)<=504
            visible()
            for j,action in enumerate(z['prefix']):
                obs,*_=env.step(action);visible()
                if (j+1)%5==0:frames.append(env.render().copy());states.append(obs['state'].copy())
            np.testing.assert_array_equal(frames,z['history_pixels']);np.testing.assert_array_equal(states,z['history_states'])
            contacts=0;terms=0
            for action in actions:
                obs,_,term,_,_=env.step(action);visible();contacts+=int(env.n_contact_points>0);terms+=int(term)
            np.testing.assert_array_equal(obs['state'],z['goal_state'] if goal else z['terminal_states'][branch])
            np.testing.assert_array_equal(env.render(),z['goal_pixels'] if goal else z['terminal_pixels'][branch]);env.close()
            if not goal:assert contacts==z['contacts'][branch] and terms==z['terminations'][branch]
        displacement=float(np.linalg.norm(z['goal_state'][2:4]-z['history_states'][-1,2:4]));assert displacement>=40
        np.testing.assert_allclose(displacement,item['initial_goal_block_distance'],rtol=1e-12)
        rows.append(dict(index=i,seed=seed,sha256=sha(path),independently_replayed_branches=33))
    r=dict(status='PASS_FULL_REFERENCE_AND_GOAL_REPLAY',rows=rows,manifest_sha256=sha(bank/'manifest.json'),source_sha256=sha(__file__),scope='Full admitted context/reference replay; no learned-planner transitions, fitting, or independent scientific result yet. Rejected seed admission is retained in generator manifest, not independently re-enumerated.')
    with (bank/'acceptance.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('PASS',a.count,'contexts',33*a.count,'branches')

if __name__=='__main__':main()
