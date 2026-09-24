"""Complete, paired development-noise analysis; no checkpoint/condition selection."""
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def noise_pairs(keys):
    """Keys are (sigma,pool,architecture,objective,algorithm); left minus right."""
    pairs=[]
    for key in sorted(keys):
        sigma,pool,arch,mode,algorithm=key
        candidates=[]
        if sigma:candidates.append(((0,pool,arch,mode,algorithm),'noisy minus clean'))
        if algorithm=='cem':candidates.append(((sigma,pool,arch,mode,'random'),'CEM minus random'))
        if mode in ['inverse','inverse_goal']:
            candidates.extend([((sigma,pool,arch,'none',algorithm),'auxiliary minus none'),((sigma,pool,arch,'direct_state',algorithm),'auxiliary minus state')])
        if mode=='inverse_goal':candidates.append(((sigma,pool,arch,'inverse',algorithm),'goal addition'))
        for right,label in candidates:
            assert right in keys
            pairs.append((key,right,label))
    return pairs

def robustness_interactions(values,successes,draws):
    """Compare degradation, separating clean advantage from noise robustness."""
    rows=[];groups=defaultdict(list)
    for sigma in [8,24]:
        for pool in range(3):
            for arch in ['transformer','gru']:
                for algorithm in ['random','cem']:
                    for mode in ['inverse','inverse_goal']:
                        for reference in ['none','direct_state']:
                            keys=[(sigma,pool,arch,mode,algorithm),(0,pool,arch,mode,algorithm),(sigma,pool,arch,reference,algorithm),(0,pool,arch,reference,algorithm)]
                            x,x0,y,y0=[np.asarray(values[k]) for k in keys]
                            delta=(x-x0)-(y-y0)
                            u,u0,v,v0=[np.asarray(successes[k]) for k in keys]
                            row=dict(sigma=sigma,replica=pool,architecture=arch,algorithm=algorithm,mode=mode,reference=reference,
                                mean_cost_degradation_difference=float(delta.mean()),success_change_difference=float(((u-u0)-(v-v0)).mean()),
                                conditional_scenario_95_percentile_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
                            rows.append(row);groups[sigma,arch,algorithm,mode,reference].append(row)
    effects=[]
    for key,rs in sorted(groups.items()):
        assert [r['replica'] for r in rs]==[0,1,2]
        effects.append(dict(zip(['sigma','architecture','algorithm','mode','reference'],key),pools=rs,
            mean_cost_degradation_difference=float(np.mean([r['mean_cost_degradation_difference'] for r in rs])),
            smaller_cost_degradation_pools=sum(r['mean_cost_degradation_difference']<0 for r in rs)))
    assert len(rows)==96 and len(effects)==32
    return dict(rows=rows,effects=effects,interpretation='(auxiliary noisy minus clean) minus (reference noisy minus clean). Negative cost interaction means smaller absolute cost degradation; positive success interaction means a more favorable success change. Does not establish better absolute noisy performance, relative degradation, or population robustness.')

def main():
    p=argparse.ArgumentParser()
    for k in ['plan','runs','output','clean-action-summary','clean-state-summary']:p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=Path(a.plan);entries=json.loads(plan.read_text())['scientific'];assert len(entries)==96
    clean_paths={'action':Path(a.clean_action_summary),'state':Path(a.clean_state_summary)}
    accepted_clean={k:json.loads(v.read_text()) for k,v in clean_paths.items()}
    tables={};values={};successes={};provenance=[];seeds=None;zero=None
    def register(key,run,expected_bank,expected_weights):
        nonlocal seeds,zero
        r=json.loads((run/'summary.json').read_text());artifacts=json.loads((run/'artifact_manifest.json').read_text())
        assert sha(run/'summary.json')==artifacts['summary.json']
        assert r['hashes']['bank_manifest']==expected_bank and r['hashes']['weights']==expected_weights
        assert r['checkpoint']=='last' and len(r['cases'])==128
        ids=[c['seed'] for c in r['cases']];zs=[c['zero_cost'] for c in r['cases']]
        if seeds is None:seeds=ids;zero=zs
        assert ids==seeds and zs==zero
        assert all(c['scored_candidates']==9000 for c in r['cases'])
        for file,h in json.loads((run/'source/manifest.json').read_text()).items():assert sha(run/'source'/file)==h
        v=np.array([c['realized_cost'] for c in r['cases']]);s=np.array([c['success'] for c in r['cases']],dtype=float)
        assert np.isfinite(v).all() and (v>=0).all()
        if key in values:
            np.testing.assert_array_equal(v,values[key]);np.testing.assert_array_equal(s,successes[key]);return
        values[key]=v;successes[key]=s
        tables[key]=dict(zip(['sigma','replica','architecture','mode','algorithm'],key),mean_cost=float(v.mean()),success=float(s.mean()),summary_sha256=sha(run/'summary.json'))
    for e in entries:
        run=Path(a.runs)/f"job_{e['index']}";manifest=plan.parent/e['manifest'];assert sha(manifest)==e['manifest_sha256']
        f=json.loads(manifest.read_text());route=f['routes'][e['route_index']];model=f['models'][route['model_index']]
        n=json.loads((run/'noise_acceptance.json').read_text());r=json.loads((run/'summary.json').read_text())
        assert n['entry']==e and n['phase']=='scientific' and n['cases']==128
        assert n['plan_sha256']==sha(plan) and n['dispatcher_sha256']==sha(Path(__file__).with_name('run_observation_noise.py'))
        assert n['summary_sha256']==sha(run/'summary.json') and r['hashes']['model_manifest']==sha(manifest)
        assert r['route_index']==e['route_index'] and r['model_index']==route['model_index']
        assert r['arm']==e['arm']==model['arm'] and r['algorithm']==e['algorithm']==route['algorithm']
        assert r['training_path']==model['training_path'] and r['parameterization']==route['parameterization']
        ap=run/('acceptance.json' if e['family']=='action' else 'staged/staged_acceptance.json');acc=json.loads(ap.read_text())
        assert acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
        verifier='accept_manifest_planner.py' if e['family']=='action' else 'accept_state_staged.py'
        assert acc['verifier_sha256']==sha(Path(__file__).with_name(verifier))
        if e['family']=='action':
            assert acc['mode']==r['mode']==e['mode']==model['mode']
            assert acc['mode_binding_verifier_sha256']==sha(Path(__file__).with_name('accept_action_manifest_planner.py'))
        else:assert acc['original_summary_sha256']==sha(run/'summary.json')
        key=(e['sigma'],e['replica'],e['arm'].split('_')[0],e['mode'],e['algorithm'])
        register(key,run,f['bank_manifest_sha256'],model['weights_sha256'])
        original=Path(e['original_run']);clean=json.loads((original/'summary.json').read_text())
        assert clean['hashes']['model_manifest']==f['parent_manifest_sha256']
        assert clean['route_index']==route['parent_route_index']
        accepted=accepted_clean[e['family']];assert accepted['manifest_sha256']==f['parent_manifest_sha256'] and accepted['bank_manifest_sha256']==f['clean_bank_manifest_sha256']
        np.testing.assert_array_equal([c['realized_cost'] for c in clean['cases']],accepted['per_case_costs'][str(route['parent_route_index'])])
        table=next(t for t in accepted['tables'] if t['route_index']==route['parent_route_index'])
        assert table['success']==float(np.mean([c['success'] for c in clean['cases']]))
        register((0,*key[1:]),original,f['clean_bank_manifest_sha256'],model['weights_sha256'])
        provenance.append(dict(index=e['index'],noise_acceptance_sha256=sha(run/'noise_acceptance.json'),physical_acceptance_sha256=sha(ap),original_summary_sha256=sha(original/'summary.json')))
    assert len(tables)==144
    draws=np.random.default_rng(1113001).integers(0,128,(10000,128));contrasts=[];groups=defaultdict(list)
    for left,right,label in noise_pairs(tables):
        delta=values[left]-values[right];sd=successes[left]-successes[right]
        row=dict(left=list(left),right=list(right),contrast=label,mean_cost_difference=float(delta.mean()),success_difference=float(sd.mean()),conditional_scenario_95_percentile_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        contrasts.append(row);groups[(label,left[0],*left[2:],right[0],*right[2:])].append(row)
    effects=[]
    for key,rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['left'][1]);assert [r['left'][1] for r in rows]==[0,1,2]
        effects.append(dict(group=list(key),pools=rows,mean_cost_difference=float(np.mean([r['mean_cost_difference'] for r in rows])),cost_improvement_pools=sum(r['mean_cost_difference']<0 for r in rows)))
    interactions=robustness_interactions(values,successes,draws)
    result=dict(per_case_records=[dict(key=list(k),costs=values[k].tolist(),successes=successes[k].tolist()) for k in sorted(values)],goal_seeds=seeds,robustness_interactions=interactions,clean_summary_hashes={k:sha(v) for k,v in clean_paths.items()},plan_sha256=sha(plan),script_sha256=sha(Path(__file__)),cases=128,tables=list(tables.values()),contrasts=contrasts,effects=effects,provenance=provenance,scope='144 cells including48 accepted clean references;96 noisy routes. Fixed-last, shared development goals and one noise draw per image. Nominal intervals conditional on each trained model; three pools do not establish population uncertainty. No universal robustness or novel-method claim.')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('SUMMARIZED',len(tables),'CELLS',len(contrasts),'PAIRS',len(effects),'GROUPS')
if __name__=='__main__':main()
