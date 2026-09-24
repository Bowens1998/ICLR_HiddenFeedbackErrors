"""Paired effects of one readout-preserving feedback intervention."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from summarize_task_coordinate_horizon import contrast,select


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());data={};bindings=[];seeds=None;conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];paths=['free','fiber','reset','observed']
    for g in range(6):
        d=Path(a.runs)/f'job_{g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert ac['status']=='PASS_ALL3_MATCHED_FIBER_ROLLOUT_RECONSTRUCTIONS' and ac['source_sha256']==sha(Path(__file__).with_name('accept_matched_fiber_rollout.py')) and ac['report_sha256']==sha(d/'report.json')
        assert ac['index']==r['index']==g and ac['protocol_sha256']==r['protocol_sha256']==sha(a.protocol) and r['plan_sha256']==sha(a.plan) and len(ac['rows'])==3
        assert r['anchor_checks']==r['readout_checks']==1536 and r['goal_checks']==384 and ac['max_initial_normalized_readout_error']<=1e-6
        if seeds is None:seeds=ac['seeds'];assert len(seeds)==128
        else:assert seeds==ac['seeds']
        for c,row in enumerate(ac['rows']):
            mi=8*g+c+2;assert row['model_index']==mi and row['condition']==conditions[c]==plan['models'][mi]['adaptation_condition'] and row['file_sha256']==sha(d/r['rows'][c]['file'])
            data[g,c]={path:{k:np.asarray(v,float) for k,v in row['metrics'][path].items()} for path in paths}
            assert all(v.shape==(4,128,5) and np.isfinite(v).all() for mm in data[g,c].values() for v in mm.values())
        bindings.append(dict(group=g,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
    draws=np.random.default_rng(1364001).integers(0,128,(10000,128));groups=[];effects=[]
    streams=['equal_four_streams','original_jepa_random','original_jepa_cem','original_state_random','original_state_cem']
    for ref,label in enumerate(streams):
        for c,condition in enumerate(conditions):
            pool={(path,k):np.stack([select(data[g,c][path][k],ref) for g in range(6)]) for path in paths for k in data[0,c][path]}
            for path in paths:
                mm={k:pool[path,k] for k in data[0,c][path]};groups.append(dict(reference_stream=label,condition=condition,path=path,means={k:v.mean((0,1)).tolist() for k,v in mm.items()},per_model=[dict(group=g,**{k:v[g].mean(0).tolist() for k,v in mm.items()}) for g in range(6)]))
            for path in ['fiber','reset']:
                effects.append(dict(reference_stream=label,condition=condition,left=path,right='free',metrics={k:contrast(pool[path,k],pool['free',k],draws) for k in data[0,c][path]}))
    interactions=[]
    for c in (1,2):
        for ref,label in enumerate(streams):
            metrics={}
            for metric in data[0,0]['free']:
                coordinate=np.stack([select(data[g,c]['fiber'][metric]-data[g,c]['free'][metric],ref) for g in range(6)])
                latent=np.stack([select(data[g,0]['fiber'][metric]-data[g,0]['free'][metric],ref) for g in range(6)])
                metrics[metric]=contrast(coordinate,latent,draws)
            interactions.append(dict(reference_stream=label,left_objective=conditions[c],right_objective=conditions[0],estimand='(matched minus free) coordinate effect minus latent effect',metrics=metrics))
    result=dict(status='COMPLETE18_MODELS_MATCHED_FEEDBACK_INTERVENTION',groups=groups,contrasts=effects,objective_effect_interactions=interactions,bindings=bindings,seeds=seeds,horizons=[5,10,15,20,25],complete_free_trajectory_anchors=9216,readout_preservation_checks=9216,plan_sha256=sha(a.plan),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),statistics_source_sha256=sha(Path(__file__).with_name('summarize_task_coordinate_horizon.py')),bootstrap=dict(seed=1364001,replicates=10000,scope='Retrospective secondary95% marginal conditional shared-goal intervals;128 goals, not model/stream/horizon counts.',designated_mechanism_endpoint='25-action block-position MSE; earlier horizons describe propagation'),scope='Paired displacement-matched observed-guided feedback change after first five actions. Fiber preserves immediate task readout; full reset also changes it. Neither intervention is a deployable policy.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],flush=True)

if __name__=='__main__':main()
