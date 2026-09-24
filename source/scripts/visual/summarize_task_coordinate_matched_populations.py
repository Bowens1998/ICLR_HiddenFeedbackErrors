"""Complete retrospective same-action contrasts; shared goals, no primary claims."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());data={};bindings=[];seeds=None
    for group in range(6):
        d=Path(a.runs)/f'job_{group}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert ac['status']=='PASS_ALL8_MATCHED_ACTION_NUMPY_RECONSTRUCTIONS' and ac['report_sha256']==sha(d/'report.json') and ac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_matched_populations.py'))
        assert r['plan_sha256']==sha(a.plan) and ac['protocol_sha256']==r['protocol_sha256']==sha(a.protocol) and ac['index']==r['index']==group
        assert r['anchor_checks']==512 and r['same_image_checks']==3840 and len(ac['rows'])==8
        if seeds is None:seeds=ac['seeds']
        else:assert seeds==ac['seeds']
        for slot,row in enumerate(ac['rows']):
            assert row['model_index']==8*group+slot;entry=plan['models'][row['model_index']];assert row['condition']==entry['adaptation_condition'] and row['recipe']==entry['score']
            assert row['file_sha256']==sha(d/r['rows'][slot]['file'])
            values={k:np.asarray(v,dtype=float) for k,v in row['metrics'].items()};assert all(v.shape==(4,128) and np.isfinite(v).all() for v in values.values())
            data[(group,slot)]=values
        bindings.append(dict(group=group,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
    rng=np.random.default_rng(1356001);draws=rng.integers(0,128,(10000,128));groups=[];contrasts=[]
    refs=['equal_four_streams','original_jepa_random','original_jepa_cem','original_state_random','original_state_cem']
    def select(v,ref):return v.mean(0) if ref==0 else v[ref-1]
    conditions=['original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels','original','clipped_state','unit_state']
    comparisons=[(3,2),(4,3),(1,0),(2,0),(3,0),(4,0),(6,5),(7,5)]
    for ref,label in enumerate(refs):
        for slot,condition in enumerate(conditions):
            values={k:np.stack([select(data[(g,slot)][k],ref) for g in range(6)]) for k in data[(0,slot)]}
            groups.append(dict(reference_stream=label,condition=condition,recipe='pose_encoded' if slot<5 else 'state',means={k:float(v.mean()) for k,v in values.items()},per_model=[dict(group=g,**{k:float(v[g].mean()) for k,v in values.items()}) for g in range(6)]))
        for left,right in comparisons:
            results={}
            for metric in data[(0,left)]:
                delta=np.stack([select(data[(g,left)][metric]-data[(g,right)][metric],ref) for g in range(6)]);paired=delta.mean(0)
                results[metric]=dict(mean_difference=float(paired.mean()),secondary_95_percentile_interval=np.quantile(paired[draws].mean(1),[.025,.975]).tolist(),per_model_differences=delta.mean(1).tolist())
            contrasts.append(dict(reference_stream=label,left=conditions[left],right=conditions[right],recipe='pose_encoded' if left<5 else 'state',metrics=results))
    result=dict(status='COMPLETE48_MODELS_MATCHED_ACTION_DIAGNOSTICS',groups=groups,contrasts=contrasts,bindings=bindings,seeds=seeds,anchor_checks=3072,same_image_checks=23040,plan_sha256=sha(a.plan),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),bootstrap=dict(seed=1356001,replicates=10000,unit='same128 goal indices shared across all conditions, groups and fixed reference streams',scope='Retrospective secondary95% conditional intervals; no fresh or primary or simultaneous coverage claim.'),scope='All8 conditions/group on identical four original-policy streams.24576 model/reference/goal predictions but128 shared goal units. No new fitting, action search or physical data. Equal-four-stream means and every individual stream retained. Causal scope limited to model behavior on these fixed references.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],flush=True)

if __name__=='__main__':main()
