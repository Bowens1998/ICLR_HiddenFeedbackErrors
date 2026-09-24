"""Shared-goal, retrospective horizon contrasts for all frozen models."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha

CONDITIONS=['original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels','original','clipped_state','unit_state']
STREAMS=['equal_four_streams','original_jepa_random','original_jepa_cem','original_state_random','original_state_cem']
PATHS=['free','teacher','observed']

def select(values,ref):return values.mean(0) if ref==0 else values[ref-1]

def contrast(left,right,draws):
    # Each array is six fixed models by128 shared goals byfive horizons.
    delta=left-right;paired=delta.mean(0)
    ci=np.quantile(paired[draws].mean(1),[.025,.975],axis=0).T
    return dict(mean_difference=paired.mean(0).tolist(),secondary_95_percentile_interval=ci.tolist(),per_model_differences=delta.mean(1).tolist())

def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());data={};bindings=[];seeds=None
    for g in range(6):
        d=Path(a.runs)/f'job_{g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert ac['status']=='PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS' and ac['report_sha256']==sha(d/'report.json') and ac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_horizon.py'))
        assert r['plan_sha256']==sha(a.plan) and ac['protocol_sha256']==r['protocol_sha256']==sha(a.protocol) and ac['index']==r['index']==g and len(ac['rows'])==8
        assert r['anchor_checks']==r['first_step_checks']==r['observed_terminal_checks']==4096 and r['same_image_checks']==16128
        assert ac['horizons']==[5,10,15,20,25]
        if seeds is None:seeds=ac['seeds'];assert len(seeds)==128
        else:assert seeds==ac['seeds']
        for slot,row in enumerate(ac['rows']):
            assert row['model_index']==8*g+slot;e=plan['models'][row['model_index']];assert row['condition']==e['adaptation_condition'] and row['recipe']==e['score'] and row['file_sha256']==sha(d/r['rows'][slot]['file'])
            values={path:{k:np.asarray(v,float) for k,v in row['metrics'][path].items()} for path in PATHS}
            assert all(v.shape==(4,128,5) and np.isfinite(v).all() for m in values.values() for v in m.values());data[g,slot]=values
        bindings.append(dict(group=g,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
    draws=np.random.default_rng(1357001).integers(0,128,(10000,128));groups=[];contrasts=[]
    for ref,label in enumerate(STREAMS):
        pool={(slot,path,k):np.stack([select(data[g,slot][path][k],ref) for g in range(6)]) for slot in range(8) for path in PATHS for k in data[0,slot][path]}
        for slot,condition in enumerate(CONDITIONS):
            recipe='pose_encoded' if slot<5 else 'state'
            for path in PATHS:
                mm={k:pool[slot,path,k] for k in data[0,slot][path]}
                groups.append(dict(reference_stream=label,condition=condition,recipe=recipe,path=path,means={k:v.mean((0,1)).tolist() for k,v in mm.items()},per_model=[dict(group=g,**{k:v[g].mean(0).tolist() for k,v in mm.items()}) for g in range(6)]))
            contrasts.append(dict(reference_stream=label,recipe=recipe,kind='free_minus_teacher',condition=condition,metrics={k:contrast(pool[slot,'free',k],pool[slot,'teacher',k],draws) for k in data[0,slot]['free']}))
        for left,right in [(3,2),(4,3)]:
            for path in ['free','teacher']:
                contrasts.append(dict(reference_stream=label,recipe='pose_encoded',kind='objective_difference',left=CONDITIONS[left],right=CONDITIONS[right],path=path,metrics={k:contrast(pool[left,path,k],pool[right,path,k],draws) for k in data[0,left][path]}))
    result=dict(status='COMPLETE48_MODELS_FIVE_HORIZON_DIAGNOSTICS',groups=groups,contrasts=contrasts,bindings=bindings,seeds=seeds,horizons=[5,10,15,20,25],endpoint_anchor_checks=24576,first_step_checks=24576,observed_terminal_checks=24576,same_image_checks=96768,plan_sha256=sha(a.plan),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),bootstrap=dict(seed=1357001,replicates=10000,unit='128 shared goals; models, streams and horizons not independent replicates',scope='Secondary conditional95% marginal percentile intervals. Retrospective, no primary/simultaneous/fresh confirmation claim.'),scope='Free versus privileged actual-history feedback on fixed original-policy action streams; no new training or search. All six model points retained.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],flush=True)

if __name__=='__main__':main()
