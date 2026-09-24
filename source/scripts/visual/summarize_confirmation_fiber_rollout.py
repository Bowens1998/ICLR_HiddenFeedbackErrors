"""Paired effects of one readout-preserving feedback intervention."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from summarize_task_coordinate_horizon import contrast,select


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());data={};bindings=[];seeds=None;conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];paths=['free','fiber','shuffled','full','reset','observed']
    for g in range(6):
        d=Path(a.runs)/f'job_{g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert ac['status']=='PASS_ALL3_CONFIRMATION_FIBER_ROLLOUT_RECONSTRUCTIONS' and ac['source_sha256']==sha(Path(__file__).with_name('accept_confirmation_fiber_rollout.py')) and ac['report_sha256']==sha(d/'report.json')
        assert ac['index']==r['index']==g and ac['protocol_sha256']==r['protocol_sha256']==sha(a.protocol) and r['plan_sha256']==sha(a.plan) and len(ac['rows'])==3
        assert r['anchor_checks']==1536 and r['readout_checks']==4608 and r['goal_checks']==384 and ac['max_initial_normalized_readout_error']<=1e-6
        if seeds is None:seeds=ac['seeds'];assert len(seeds)==128
        else:assert seeds==ac['seeds']
        for c,row in enumerate(ac['rows']):
            mi=8*g+c+2;assert row['model_index']==mi and row['condition']==conditions[c]==plan['models'][mi]['adaptation_condition'] and row['file_sha256']==sha(d/r['rows'][c]['file'])
            data[g,c]={path:{k:np.asarray(v,float) for k,v in row['metrics'][path].items()} for path in paths}
            assert all(v.shape==(4,128,5) and np.isfinite(v).all() for mm in data[g,c].values() for v in mm.values())
        bindings.append(dict(group=g,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
    draws=np.random.default_rng(1368001).integers(0,128,(20000,128));groups=[];effects=[]
    streams=['equal_four_streams','original_jepa_random','original_jepa_cem','original_state_random','original_state_cem']
    for ref,label in enumerate(streams):
        for c,condition in enumerate(conditions):
            pool={(path,k):np.stack([select(data[g,c][path][k],ref) for g in range(6)]) for path in paths for k in data[0,c][path]}
            for path in paths:
                mm={k:pool[path,k] for k in data[0,c][path]};groups.append(dict(reference_stream=label,condition=condition,path=path,means={k:v.mean((0,1)).tolist() for k,v in mm.items()},per_model=[dict(group=g,**{k:v[g].mean(0).tolist() for k,v in mm.items()}) for g in range(6)]))
            for path in ['fiber','shuffled','full','reset']:
                effects.append(dict(reference_stream=label,condition=condition,left=path,right='free',metrics={k:contrast(pool[path,k],pool['free',k],draws) for k in data[0,c][path]}))
    guidance_contrasts=[]
    for ref,label in enumerate(streams):
        for c,condition in enumerate(conditions):
            metrics={}
            for metric in data[0,c]['free']:
                guided=np.stack([select(data[g,c]['fiber'][metric],ref) for g in range(6)])
                shuffled=np.stack([select(data[g,c]['shuffled'][metric],ref) for g in range(6)])
                metrics[metric]=contrast(guided,shuffled,draws)
            guidance_contrasts.append(dict(reference_stream=label,condition=condition,left='fiber',right='shuffled',metrics=metrics))
    assert plan['source_plan_sha256']=='1fd34ed8b462afc5901b6d82c9ef119ff689424be03803b92882064a5559c79b' and plan['fiber_confirmation']['protocol_sha256']==sha(a.protocol)
    assert len(set(seeds))==128 and min(seeds)>=1371001 and max(seeds)<1373049
    primary=[]
    for c,left,right in [(1,'fiber','free'),(2,'fiber','free'),(0,'fiber','shuffled'),(1,'fiber','shuffled'),(2,'fiber','shuffled')]:
        delta=np.stack([(data[g,c][left]['position_mse']-data[g,c][right]['position_mse']).mean(0)[:,-1] for g in range(6)])
        paired=delta.mean(0);ci=np.quantile(paired[draws].mean(1),[.005,.995]);primary.append(dict(condition=conditions[c],left=left,right=right,horizon=25,metric='position_mse',mean_difference=float(paired.mean()),primary_99_percentile_interval=ci.tolist(),confirmed_improvement=bool(ci[1]<0),per_model_differences=delta.mean(1).tolist(),paired_goal_differences=paired.tolist()))
    interactions=[]
    for c in (1,2):
        left=np.stack([(data[g,c]['fiber']['position_mse']-data[g,c]['free']['position_mse']).mean(0) for g in range(6)])
        right=np.stack([(data[g,0]['fiber']['position_mse']-data[g,0]['free']['position_mse']).mean(0) for g in range(6)])
        interactions.append(dict(left_objective=conditions[c],right_objective=conditions[0],metrics=dict(position_mse=contrast(left,right,draws))))
    result=dict(status='COMPLETE18_MODELS_FRESH_FIBER_CONFIRMATION',groups=groups,contrasts=effects,primary_contrasts=primary,primary_confirmed=sum(x['confirmed_improvement'] for x in primary),objective_effect_interactions=interactions,guidance_contrasts=guidance_contrasts,bindings=bindings,seeds=seeds,horizons=[5,10,15,20,25],complete_free_trajectory_anchors=9216,readout_preservation_checks=27648,plan_sha256=sha(a.plan),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),statistics_source_sha256=sha(Path(__file__).with_name('summarize_task_coordinate_horizon.py')),bootstrap=dict(seed=1368001,replicates=20000,scope='Five primary 99% two-sided marginal percentile intervals with nominal Bonferroni familywise 95%; secondary 95% intervals. Fresh recipient goals; fixed external donor bank and models. Streams/horizons are repeated observations.',designated_mechanism_endpoint='25-action block-position MSE; earlier horizons describe propagation'),scope='Fresh-goal confirmation of trajectory-matched versus independent external-donor guidance at a common displacement after first five actions. Fiber preserves immediate task readout; full reset also changes it. Neither intervention is a deployable policy.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],flush=True)

if __name__=='__main__':main()
