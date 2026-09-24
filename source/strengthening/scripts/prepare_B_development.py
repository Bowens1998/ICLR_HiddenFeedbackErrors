"""Fresh native development episodes, fixed model-free controls, and exact replay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha, atomic_json
from artifact_io import atomic_npz, array_sha
from native_simulator import load_native_pusht
from input_lock import add_design_arguments,input_context


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=['diagnostic_development_B','donor_development_B','confirmation_B','donor_bank_B'],required=True)
    p.add_argument('--environment',required=True);p.add_argument('--output',required=True)
    add_design_arguments(p);a=p.parse_args()
    settings,binding=input_context(a,a.role,ROOT,__file__)
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    native=ROOT/'strengthening/external/dino_wm';PushT=load_native_pusht(native)
    envpath=Path(a.environment);env=json.loads((envpath/'report.json').read_text())
    assert env['status']=='PASS_NATIVE_SIMULATOR_DETERMINISTIC_REPLAY'
    assert env['simulator_source_sha256']==sha(native/'env/pusht/pusht_env.py')
    spec=ROOT/'strengthening/configs/B_bank_generation.json'
    design=ROOT/'strengthening/configs/bank_design.draft.json'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];rejects=[]
    def visible(env):
        for body in [env.agent,env.block]:
            for shape in body.shapes:
                bb=shape.cache_bb()
                if min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504:return False
        return True
    def reset(seed):
        env=PushT(with_velocity=True,with_target=True);env.seed(seed);obs,state=env.reset()
        return env,obs,state
    def replay(seed,actions):
        env,obs,state=reset(seed);states=[state.copy()];pixels=[obs['visual'].copy()];proprio=[obs['proprio'].copy()]
        contacts=[];boundary=[];reward=[];done=[]
        for t,u in enumerate(actions):
            obs,r,d,info=env.step(u);states.append(info['state'].copy());proprio.append(obs['proprio'].copy())
            if (t+1)%5==0:pixels.append(obs['visual'].copy())
            contacts.append(env.n_contact_points>0);boundary.append(not visible(env));reward.append(r);done.append(d)
        env.close()
        return dict(pixels=np.stack(pixels),states=np.stack(states),proprio=np.stack(proprio),actions=actions.copy(),
            contacts=np.asarray(contacts),boundary=np.asarray(boundary),rewards=np.asarray(reward),terminations=np.asarray(done))
    for seed in range(settings['seed_start'],settings['seed_start']+settings['max_seeds']):
        env,obs,state=reset(seed);admitted=visible(env);prefix=[]
        for _ in range(10):
            target=state[2:4]+np.array([0.,65.]);u=np.clip((target-state[:2])/100.,-.7,.7).astype(np.float32)
            prefix.append(u);obs,_,_,info=env.step(u);state=info['state'];admitted &= visible(env)
        env.close()
        if not admitted:
            rejects.append(dict(seed=seed,reason='history_not_fully_visible'));continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,51493]))
        future=np.repeat(rng.uniform(-.35,.35,(5,2)),5,axis=0).astype(np.float32)
        actions=np.concatenate([np.stack(prefix),future]);left=replay(seed,actions);right=replay(seed,actions)
        for key in left:
            if key=='rewards':
                # Shapely polygon intersection may change the final rounding bit;
                # rewards are not inputs or scores of the feedback diagnostic.
                np.testing.assert_allclose(left[key],right[key],rtol=0,atol=1e-12)
            else:np.testing.assert_array_equal(left[key],right[key])
        np.testing.assert_array_equal(left['proprio'],left['states'][:,[0,1,5,6]])
        assert left['pixels'].shape==(8,224,224,3) and left['states'].shape==(36,7)
        assert not left['boundary'][:10].any() and np.isfinite(left['states']).all()
        i=len(rows);target=out/f'case_{i:03d}.npz';atomic_npz(target,**left,seed=np.asarray(seed))
        rows.append(dict(index=i,seed=seed,file=target.name,sha256=sha(target),
            initial_state_sha256=array_sha(left['states'][0]),trajectory_sha256=array_sha(left['states']),
            action_sha256=array_sha(actions),history_pixels_sha256=array_sha(left['pixels'][:3]),
            two_physics_pixel_proprio_replays_exact=True,proprio_channel_check=True,
            unused_reward_max_replay_difference=float(np.max(abs(left['rewards']-right['rewards'])))))
        atomic_json(out/'generation_progress.json',dict(cases=rows,rejected_seeds=rejects))
        if len(rows)==settings['count']:break
    if len(rows)!=settings['count']:raise RuntimeError('Fixed seed budget exhausted; do not reduce sample count')
    assert len({x['trajectory_sha256'] for x in rows})==len(rows)
    atomic_json(out/'manifest.json',dict(**binding,status='PASS_NATIVE_INPUT_BANK',role=a.role,cases=rows,rejected_seeds=rejects,
        settings=settings,generation_contract_sha256=sha(spec),bank_design_sha256=sha(design),
        simulator_environment_sha256=sha(envpath/'report.json'),simulator_source_sha256=sha(native/'env/pusht/pusht_env.py'),
        loader_source_sha256=sha(ROOT/'strengthening/adapters/native_simulator.py'),source_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-start,scope='Fresh native simulator input trajectories, one fixed model-free action source; all future outcomes retained. No model evaluated.'))
    atomic_json(out/'role.json',dict(**binding,role=a.role,parent_manifest_sha256=sha(out/'manifest.json'),count=len(rows),
        generation_contract_sha256=sha(spec),source_sha256=sha(__file__)))
    (out/'DONE').write_text('accepted_native_input_bank\n')


if __name__=='__main__':main()
