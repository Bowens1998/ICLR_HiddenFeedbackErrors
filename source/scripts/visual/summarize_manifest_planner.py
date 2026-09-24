"""Complete frozen comparisons with scenario intervals conditional on each model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from planning_acceptance import load_acceptance

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--staged-state-reports');a=p.parse_args();root=Path(a.runs);mp=Path(a.manifest);frozen=json.loads(mp.read_text());mh=hashlib.sha256(mp.read_bytes()).hexdigest();tables=[];costs={};seeds=None;zero=None;bank=None
    for i,route in enumerate(frozen['routes']):
        run=root/f'job_{i}';r=json.loads((run/'summary.json').read_text());acc,acceptance_path=load_acceptance(run,a.staged_state_reports);entry=frozen['models'][route['model_index']]
        artifacts=json.loads((run/'artifact_manifest.json').read_text());assert hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()==artifacts['summary.json']
        assert r['route_index']==i and r['model_index']==route['model_index'] and r['checkpoint']==entry['checkpoint'] and r['training_path']==entry['training_path']
        assert r['hashes']['model_manifest']==mh and r['hashes']['weights']==entry['weights_sha256']
        assert r['arm']==entry['arm']
        if frozen['layout']=='action_auxiliary':
            assert r['mode']==entry['mode']==acc['mode']
            assert acc['verifier_sha256']==hashlib.sha256(Path(__file__).with_name('accept_manifest_planner.py').read_bytes()).hexdigest()
            assert acc['mode_binding_verifier_sha256']==hashlib.sha256(Path(__file__).with_name('accept_action_manifest_planner.py').read_bytes()).hexdigest()
        assert (r['algorithm'],r['parameterization'])==(route['algorithm'],route['parameterization'])
        assert len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
        for file,h in json.loads((run/'source/manifest.json').read_text()).items():assert hashlib.sha256((run/'source'/file).read_bytes()).hexdigest()==h
        ids=[v['seed'] for v in r['cases']];z=np.array([v['zero_cost'] for v in r['cases']])
        if seeds is None:seeds=ids;zero=z;bank=r['hashes']['bank_manifest']
        assert ids==seeds and r['hashes']['bank_manifest']==bank;np.testing.assert_array_equal(z,zero)
        assert all(v['scored_candidates']==9000 for v in r['cases'])
        values=np.array([v['realized_cost'] for v in r['cases']]);costs[i]=values
        tables.append({'route_index':i,'acceptance_sha256':hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),'verification_policy':acc.get('verification_policy','original_full_float64'),**{k:entry[k] for k in ['arm','mode','checkpoint','seed','replica','episodes','updates'] if k in entry},'algorithm':r['algorithm'],'parameterization':r['parameterization'],'score_space':r['score_space'],'mean_cost':float(values.mean()),'success':float(np.mean([v['success'] for v in r['cases']])),'mean_native_score':float(np.mean([v['predicted_cost'] for v in r['cases']]))})
    draws=np.random.default_rng(962001).integers(0,128,(10000,128));contrasts=[]
    def add(left,right,label):
        delta=costs[left['route_index']]-costs[right['route_index']]
        contrasts.append({'contrast':label,'left_route':left['route_index'],'right_route':right['route_index'],'mean_cost_difference':float(delta.mean()),'conditional_scenario_95_percentile_interval':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()})
    def same(a,b,omit):return all(a.get(k)==b.get(k) for k in ['arm','mode','checkpoint','seed','replica','episodes','updates','algorithm','parameterization'] if k not in omit)
    for left in tables:
        for right in tables:
            if left['algorithm']=='cem' and right['algorithm']=='random' and same(left,right,{'algorithm'}):add(left,right,'CEM minus random')
            if frozen['layout']=='scaling':
                if left['episodes']==1024 and right['episodes']==256 and same(left,right,{'episodes'}):add(left,right,'1024 minus256 episodes, fixed updates')
                if left['updates']==21000 and right['updates']==5250 and same(left,right,{'updates'}):add(left,right,'21000 minus5250 updates, fixed episodes')
    if frozen['layout']=='action_auxiliary':
        from action_comparison_pairs import comparison_pairs
        contrasts.clear()
        for left,right,label in comparison_pairs(tables):add(tables[left],tables[right],label)
    result={'layout':frozen['layout'],'scope':'previously examined development scenarios; nominal unadjusted intervals condition on fixed trained models, not independent training uncertainty; native scores only comparable within their own scoring space','manifest_sha256':mh,'bank_manifest_sha256':bank,'cases':128,'zero_cost':float(zero.mean()),'tables':tables,'contrasts':contrasts,'per_case_costs':{str(i):v.tolist() for i,v in costs.items()}}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARIZED',len(tables),'routes',len(contrasts),'prespecified contrasts')
if __name__=='__main__':main()
