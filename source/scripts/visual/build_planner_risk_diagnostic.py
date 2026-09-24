"""Consumed paired-search decision diagnostic; no fitted selector or novelty claim."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha
from diagnose_feature_support_risk import auc


def main():
    p=argparse.ArgumentParser()
    for k in ['pusht-runs','pusht-summary','reacher-runs','reacher-summary','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[];arrays={};sources={}
    for task,root,summary,n in [('pusht',a.pusht_runs,a.pusht_summary,128),('reacher',a.reacher_runs,a.reacher_summary,256)]:
        full=json.loads(Path(summary).read_text());assert len(full['rows'])==48 and len(full['goal_seeds'])==n
        sources[task]=sha(summary)
        for index in range(0,48,2):
            pair=[];provenance=[]
            for j in [index,index+1]:
                d=Path(root)/f'job_{j}';r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
                assert len(r['cases'])==ac['cases']==n
                if task=='pusht':
                    binding=next(v for v in full['bindings'] if v['index']==j)
                    assert binding['summary_sha256']==sha(d/'summary.json') and binding['acceptance_sha256']==sha(d/'acceptance.json')
                else:assert ac['summary_sha256']==sha(d/'summary.json')
                assert [v['seed'] for v in r['cases']]==full['goal_seeds']
                assert r['algorithm']==('random' if j==index else 'cem')
                assert all(v['scored_candidates']==9000 for v in r['cases'])
                actual=np.array([v['realized_cost'] for v in r['cases']]);np.testing.assert_array_equal(actual,full['rows'][j]['cost'])
                pair.append(dict(actual=actual,predicted=np.array([v['predicted_cost'] for v in r['cases']]),success=np.array([v['success'] for v in r['cases']],dtype=float)))
                provenance.append(dict(route=j,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json')))
            random,cem=pair;interface=full['rows'][index].get('interface',r['score_space'])
            gain=(random['predicted']-cem['predicted'])/(abs(random['predicted'])+abs(cem['predicted'])+1e-12)
            delta=cem['actual']-random['actual'];harm=delta>0;key=f'{task}_{index}'
            assert np.isfinite(gain).all() and np.isfinite(delta).all()
            arrays[key+'_inputs']=np.column_stack([random['predicted'],cem['predicted'],gain])
            arrays[key+'_labels']=np.column_stack([delta,harm.astype(float),random['actual'],cem['actual'],random['success'],cem['success']])
            arrays[key+'_seeds']=np.array(full['goal_seeds'])
            oracle=np.minimum(random['actual'],cem['actual'])
            rows.append(dict(task=task,backbone=index//8,interface=interface,goals=n,random_mean_cost=float(random['actual'].mean()),cem_mean_cost=float(cem['actual'].mean()),cem_cost_harm_fraction=float(harm.mean()),oracle_pair_mean_cost=float(oracle.mean()),oracle_improvement_over_better_fixed_mean=float(min(random['actual'].mean(),cem['actual'].mean())-oracle.mean()),predicted_gain_as_harm_risk_auc=auc(~harm,gain),random_success=float(random['success'].mean()),cem_success=float(cem['success'].mean()),provenance=provenance))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'paired_data.npz',**arrays)
    groups=[]
    for task in ['pusht','reacher']:
        for interface in dict.fromkeys(r['interface'] for r in rows if r['task']==task):
            group=[r for r in rows if r['task']==task and r['interface']==interface];assert len(group)==6
            keys=['random_mean_cost','cem_mean_cost','cem_cost_harm_fraction','oracle_pair_mean_cost','oracle_improvement_over_better_fixed_mean']
            groups.append(dict(task=task,interface=interface,**{k:float(np.mean([r[k] for r in group])) for k in keys},gain_risk_auc_by_backbone=[r['predicted_gain_as_harm_risk_auc'] for r in group]))
    result=dict(status='COMPLETE48_PAIRED_SEARCH_DIAGNOSTICS',rows=rows,groups=groups,sources=sources,source_sha256=sha(__file__),data_sha256=sha(out/'paired_data.npz'),input_columns=['random_predicted_cost','cem_predicted_cost','normalized_predicted_gain'],label_columns=['cem_minus_random_realized_cost','cem_higher_cost','random_realized_cost','cem_realized_cost','random_success','cem_success'],scope='Consumed banks; descriptive, no fitted gate or independent confirmation. Inputs available after both searches but before physical execution. Both9000-candidate searches cost18000 total; no claim of9000-budget early stopping. Oracle is unavailable post-execution information and a pairwise lower bound only. Task/interface units not pooled; latent scores not physical costs. Cost harm is distinct from binary task failure. No thresholds tuned.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
