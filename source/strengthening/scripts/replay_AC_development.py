"""Independent complete simulator replay of compact development action archives."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha, atomic_json, require_role
from artifact_io import atomic_npz
from input_lock import add_design_arguments,input_context,check_parent_design


def main():
    p=argparse.ArgumentParser()
    for key in ['base','bank','actions','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int)
    add_design_arguments(p)
    a=p.parse_args();index=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index
    sim=Path(a.base)/'releases/visual-v1/stable-worldmodel';sys.path.insert(0,str(sim))
    from stable_worldmodel.envs.pusht.env import PushT
    bank=Path(a.bank);role=require_role(json.loads((bank/'role.json').read_text()),
        ['diagnostic_development_A_C','donor_development_A_C','confirmation_A_C','donor_bank_A_C'])
    settings,phase_binding=input_context(a,role['role'],ROOT,__file__)
    check_parent_design(role,phase_binding);count=settings['count']
    manifest=json.loads((bank/'manifest.json').read_text());assert len(manifest['cases'])==role['count']==count
    assert [c['index'] for c in manifest['cases']]==list(range(count))
    assert role['parent_manifest_sha256']==sha(bank/'manifest.json') and (bank/'DONE').exists()
    source=Path(a.actions)/f'route_{index}';report=json.loads((source/'report.json').read_text())
    assert report['status']=='PASS_COMPACT_FIXED_REFERENCE_SEARCH' and (source/'DONE').exists()
    assert report['binding']['bank_manifest_sha256']==sha(bank/'manifest.json')
    check_parent_design(report['binding'],phase_binding)
    assert report['binding']['source_sha256']==sha(Path(__file__).with_name('plan_AC_development.py'))
    out=Path(a.output)/f'route_{index}';out.mkdir(parents=True,exist_ok=True)
    binding=dict(**phase_binding,role=role['role'],bank_role_sha256=sha(bank/'role.json'),parent_manifest_sha256=sha(bank/'manifest.json'),
        action_report_sha256=sha(source/'report.json'),source_sha256=sha(__file__),
        simulator_sha256=sha(sim/'stable_worldmodel/envs/pusht/env.py'))
    if (out/'binding.json').exists():assert json.loads((out/'binding.json').read_text())==binding
    else:atomic_json(out/'binding.json',binding)
    def replay(seed,actions):
        env=PushT(resolution=224);obs,_=env.reset(seed=seed)
        states=[obs['state'].copy()];pixels=[env.render().copy()];contacts=[];terminated=[];boundary=[]
        for t,u in enumerate(actions):
            obs,_,term,_,_=env.step(u);states.append(obs['state'].copy())
            if (t+1)%5==0:pixels.append(env.render().copy())
            outside=False
            for body in [env.agent,env.block]:
                for shape in body.shapes:
                    bb=shape.cache_bb();outside|=min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504
            boundary.append(outside);contacts.append(env.n_contact_points>0);terminated.append(term)
        env.close()
        return dict(pixels=np.stack(pixels),states=np.stack(states),actions=actions.copy(),
            contacts=np.asarray(contacts),terminations=np.asarray(terminated),boundary=np.asarray(boundary))
    rows=[];start=time.monotonic()
    for c,s in zip(manifest['cases'],report['cases'],strict=True):
        i=c['index'];assert s['index']==i and s['seed']==c['seed']
        context=bank/f'case_{i:03d}.npz';ap=source/s['file']
        assert sha(context)==c['sha256'] and sha(ap)==s['file_sha256']
        target=out/f'case_{i:03d}.npz';result=out/f'case_{i:03d}.json'
        if result.exists():
            row=json.loads(result.read_text());assert row['file_sha256']==sha(target) and row['binding_sha256']==sha(out/'binding.json')
            rows.append(row);continue
        z=dict(np.load(context));az=dict(np.load(ap));j=int(az['selected_index'])
        np.testing.assert_array_equal(az['population_actions'][j],az['selected_actions'])
        assert az['population_actions'].shape==(300,25,2)
        assert np.unravel_index(np.argmin(az['cost_trace']),az['cost_trace'].shape)==(int(az['selected_iteration']),j)
        full=np.concatenate([z['prefix'],az['selected_actions']]);left=replay(c['seed'],full);right=replay(c['seed'],full)
        for key in left:np.testing.assert_array_equal(left[key],right[key])
        assert left['pixels'].shape==(8,224,224,3) and left['states'].shape==(36,7)
        np.testing.assert_array_equal(left['pixels'][:3],z['history_pixels'])
        np.testing.assert_array_equal(left['states'][[0,5,10]],z['history_states'])
        atomic_npz(target,**left,seed=np.asarray(c['seed']))
        row=dict(index=i,seed=c['seed'],file=target.name,file_sha256=sha(target),input_sha256=sha(context),
            action_file_sha256=sha(ap),binding_sha256=sha(out/'binding.json'),all_two_replays_exact=True,
            boundary_steps=int(left['boundary'][10:].sum()),scope='All selected futures retained, including contacts and exits.')
        atomic_json(result,row);rows.append(row)
    assert len(rows)==count
    atomic_json(out/'report.json',dict(status='PASS_COMPLETE_SELECTED_PHYSICS',binding=binding,cases=rows,
        elapsed_seconds=time.monotonic()-start,independent_replays=2*len(rows)))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
