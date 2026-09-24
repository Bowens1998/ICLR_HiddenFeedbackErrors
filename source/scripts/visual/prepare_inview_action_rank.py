"""Prespecified visibility-conditioned bank; no model scores used for selection."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',type=int,default=32);a=p.parse_args();sys.path.insert(0,a.source)
    from stable_worldmodel.envs.pusht.env import PushT
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];rejected=[]
    def visible(e):
        for body in [e.agent,e.block]:
            for shape in body.shapes:
                bb=shape.cache_bb()
                if min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504:return False
        return True
    for seed in range(880001,880001+256):
        if len(rows)==a.cases:break
        env=PushT(resolution=224);obs,_=env.reset(seed=seed);prefix=[]
        for _ in range(10):
            target=obs['state'][2:4]+np.array([0.,65.]);action=np.clip((target-obs['state'][:2])/100.,-.7,.7).astype('float32');prefix.append(action);obs,*_=env.step(action)
        env.close();prefix=np.array(prefix)
        def replay(plan):
            e=PushT(resolution=224);o,_=e.reset(seed=seed);frames=[e.render().copy()];states=[o['state'].copy()]
            if not visible(e):e.close();return None
            for t,action in enumerate(prefix):
                o,*_=e.step(action)
                if not visible(e):e.close();return None
                if (t+1)%5==0:frames.append(e.render().copy());states.append(o['state'].copy())
            contacts=0;terminations=0
            for action in plan:
                o,_,term,_,_=e.step(action);contacts+=int(e.n_contact_points>0);terminations+=int(term)
                if not visible(e):e.close();return None
            result=(np.array(frames),np.array(states),e.render().copy(),o['state'].copy(),contacts,terminations);e.close();return result
        zero=np.zeros((25,2),dtype='float32');first=replay(zero)
        if first is None:rejected.append({'seed':seed,'reason':'history_or_zero_branch_not_fully_visible'});continue
        rng_goal=np.random.default_rng(np.random.SeedSequence([seed,1]));rng_plans=np.random.default_rng(np.random.SeedSequence([seed,2]))
        def sample(rng):return np.repeat(rng.uniform(-.35,.35,(5,2)),5,axis=0).astype('float32')
        goal=None
        for goal_attempt in range(1,129):
            goal_plan=sample(rng_goal);goal=replay(goal_plan)
            if goal is not None:break
        if goal is None:rejected.append({'seed':seed,'reason':'no_visible_goal_in_128_attempts'});continue
        plans=[zero];results=[first];attempts=[1];failed=False
        for _ in range(31):
            result=None
            for attempt in range(1,129):
                plan=sample(rng_plans);result=replay(plan)
                if result is not None:break
            if result is None:failed=True;break
            plans.append(plan);results.append(result);attempts.append(attempt)
        if failed:rejected.append({'seed':seed,'reason':'candidate_slot_exhausted_128_attempts'});continue
        repeat=replay(zero);assert repeat is not None
        for result in results+[repeat]:
            np.testing.assert_array_equal(result[0],goal[0]);np.testing.assert_array_equal(result[1],goal[1])
        np.testing.assert_array_equal(first[2],repeat[2]);np.testing.assert_array_equal(first[3],repeat[3])
        index=len(rows);path=out/f'case_{index:03d}.npz'
        np.savez_compressed(path,seed=seed,history_pixels=goal[0],history_states=goal[1],prefix=prefix,actions=np.array(plans),goal_actions=goal_plan,
                            goal_pixels=goal[2],goal_state=goal[3],terminal_pixels=np.array([r[2] for r in results]),terminal_states=np.array([r[3] for r in results]),
                            contacts=np.array([r[4] for r in results]),terminations=np.array([r[5] for r in results]))
        row={'index':index,'seed':seed,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'goal_attempts':goal_attempt,'candidate_attempts':attempts,
             'contact_candidates':sum(r[4]>0 for r in results),'all_shapes_visible_every_step':True};rows.append(row);print(json.dumps(row),flush=True)
    assert len(rows)==a.cases,'Visibility quota not reached; do not silently reduce number of cases'
    manifest={'scope':'development; fully visible feasibility-conditioned proposals; not the official task distribution','cases':rows,'candidates':32,'rejected_seeds':rejected,
              'primitive_action_limit':.35,'visibility_box':[8,504],'seed_start':880001,'max_seeds':256,'max_attempts_per_slot':128,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'simulator_env_sha256':hashlib.sha256((Path(a.source)/'stable_worldmodel/envs/pusht/env.py').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'COMPLETE').write_text('full-shape visibility and exact replay checked\n')
if __name__=='__main__':main()
