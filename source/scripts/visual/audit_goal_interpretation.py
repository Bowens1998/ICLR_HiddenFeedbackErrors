"""Rescore fixed executed actions under alternative goal interpretations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def metrics(state,goal):
    d=state-goal;angle=(d[4]+np.pi)%(2*np.pi)-np.pi
    agent=float(np.square(d[:2]).sum());block=float(np.square(d[2:4]).sum())
    return dict(block_cost=block+900*angle**2,agent_position_cost=agent,joint_pose_cost=agent+block+900*angle**2,
                raw_seven_state_l2=float(np.linalg.norm(d)),block_success=bool(block<400 and abs(angle)<np.pi/9),joint_pose_success=bool(agent+block<400 and abs(angle)<np.pi/9))

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
bank=Path('data/pusht_prospective_v1/heldout');bm=json.loads((bank/'manifest.json').read_text());bh=sha(bank/'manifest.json');truth={}
for item in bm['cases']:
    file=bank/f"case_{item['index']:03d}.npz";assert sha(file)==item['sha256']
    with np.load(file,allow_pickle=False) as z:truth[item['index']]=(np.array(z['goal_state']),np.array(z['terminal_states'][0]))
roots=[(3072,Path('runs/visual_controlled_planner_full_v1'))]+[(s,Path(f'runs/optimizer_seed_planning_v1/seed_{s}')) for s in [3073,3074]]
rows=[]
for seed,root in roots:
    for folder in sorted(root.glob('job_*')):
        r=json.loads((folder/'summary.json').read_text())
        if r['parameterization']!='full':continue
        acc=json.loads((folder/'acceptance.json').read_text());art=json.loads((folder/'artifact_manifest.json').read_text())
        assert sha(folder/'summary.json')==art['summary.json'] and r['hashes']['bank_manifest']==bh
        assert len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
        cases=[]
        for item in r['cases']:
            i=item['index'];file=folder/f'case_{i:03d}_predictions.npz';assert sha(file)==art[file.name]
            with np.load(file,allow_pickle=False) as z:state=np.array(z['selected_states'][-1])
            goal,zero=truth[i];m=metrics(state,goal)
            np.testing.assert_allclose(m['block_cost'],item['realized_cost'],rtol=1e-9,atol=1e-7);assert m['block_success']==item['success']
            cases.append(dict(index=i,**m))
        rows.append(dict(seed=seed,arm=r['arm'],algorithm=r['algorithm'],path=str(folder),summary_sha256=sha(folder/'summary.json'),means={k:float(np.mean([c[k] for c in cases])) for k in cases[0] if k!='index'},cases=cases))
        print('VERIFIED',seed,r['arm'],r['algorithm'],flush=True)
assert len(rows)==26
lookup={(r['seed'],r['arm'],r['algorithm']):r for r in rows};assert len(lookup)==26
contrasts=[];draws=np.random.default_rng(972001).integers(0,128,(10000,128))
for r in rows:
    if r['algorithm']!='cem':continue
    other=lookup[(r['seed'],r['arm'],'random')];assert [c['index'] for c in r['cases']]==[c['index'] for c in other['cases']]
    effects={}
    for metric in ['block_cost','agent_position_cost','joint_pose_cost','raw_seven_state_l2','block_success','joint_pose_success']:
        delta=np.array([float(c[metric]) for c in r['cases']])-np.array([float(c[metric]) for c in other['cases']])
        effects[metric]=dict(mean_difference=float(delta.mean()),conditional_scenario_95_percentile_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    contrasts.append(dict(seed=r['seed'],arm=r['arm'],cem_minus_random=effects))
report=dict(rows=rows,contrasts=contrasts,bank_sha256=bh,scope='Fixed executed actions from complete full-action optimizer-seed comparison plus separately budgeted released reference; no policy rescoring or replanning. Task-definition sensitivity on repeatedly examined development cases. Raw seven-state L2 mixes units and is only a legacy diagnostic. Conditional scenario intervals are nominal, unadjusted and do not represent training uncertainty.')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
