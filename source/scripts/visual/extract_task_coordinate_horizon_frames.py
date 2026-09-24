"""Complete observed sequences of accepted selected plans; never filter outcomes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ['run','bank','plan','simulator','output','protocol']:p.add_argument('--'+k,required=True)
    a=p.parse_args();sys.path.insert(0,a.simulator)
    from stable_worldmodel.envs.pusht.env import PushT
    d=Path(a.run);r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text())
    plan=json.loads(Path(a.plan).read_text());bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert ac['model_free_simulator_replay'] and ac['cases']==len(r['cases'])==len(bm['cases'])
    assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['bank_manifest']==sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    protocol_sha=sha(a.protocol)
    assert ac['population_replay_exact_all_cases'] and ac['verifier_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_population_planner.py'))
    assert len(plan['models'])==48 and len(plan['routes'])==96 and len(bm['cases'])==128
    assert am['summary.json']==sha(d/'summary.json') and r['route_index'] in [16*g+k for g in range(6) for k in (0,1,10,11)]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
    for i,(row,item) in enumerate(zip(r['cases'],bm['cases'])):
        assert row['index']==item['index']==i and row['seed']==item['seed']
        bp=bank/f'case_{i:03d}.npz';rp=d/f'case_{i:03d}_predictions.npz'
        assert sha(bp)==item['sha256'] and sha(rp)==am[rp.name]
        with np.load(bp) as b,np.load(rp) as selected:
            actions=np.concatenate([b['prefix'],selected['selected_actions']]);assert actions.shape==(35,2)
            env=PushT(resolution=224);obs,_=env.reset(seed=item['seed']);frames=[env.render().copy()];states=[obs['state'].copy()]
            for t,action in enumerate(actions):
                obs,*_=env.step(action);states.append(obs['state'].copy())
                if (t+1)%5==0:frames.append(env.render().copy())
            frames=np.array(frames);states=np.array(states);env.close()
            assert frames.shape==(8,224,224,3) and frames.dtype==np.uint8
            np.testing.assert_array_equal(frames[:3],b['history_pixels']);np.testing.assert_array_equal(states[[0,5,10]],b['history_states'])
            np.testing.assert_allclose(states[10:],selected['selected_states'],rtol=0,atol=1e-7)
            np.testing.assert_array_equal(frames[-1],selected['terminal_pixels'])
            assert np.isfinite(actions).all() and np.isfinite(states).all()
            file=f'case_{i:03d}.npz';np.savez_compressed(out/file,pixels=frames,states=states,actions=actions,seed=item['seed'])
        records.append(dict(index=i,seed=item['seed'],file=file,file_sha256=sha(out/file),source_archive_sha256=sha(rp),bank_archive_sha256=sha(bp),source_success=row['success'],source_boundary_steps=row['boundary_steps']))
    report=dict(status='PASS_REPLAYED_COMPLETE_SELECTED_TRAJECTORIES',route=r['route_index'],cases=records,plan_sha256=sha(a.plan),summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),source_sha256=sha(__file__),simulator_sha256=sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py'),scope='Eight stride-five observed frames,35 actions,36 states. All admitted selected plans retained regardless of cost/success/boundary. This engineering extraction is not formal training data or a fitted model result.')
    assert protocol_sha==sha(a.protocol)
    report['protocol_sha256']=protocol_sha
    report['scope']='All128 previously executed original-policy reference trajectories, replayed for intermediate image diagnostics. No new fitting, search or independent physical evaluation.'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',r['route_index'],len(records))

if __name__=='__main__':main()
