"""Independent nonlinear readout and fixed-input checks on every saved correction."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def forward(h,z):
    a=(z.astype(float)-h['mean'])/h['scale']
    first=a@h['0.weight'].astype(float).T+h['0.bias'];second=np.maximum(first,0)@h['2.weight'].astype(float).T+h['2.bias']
    result=np.maximum(second,0)@h['4.weight'].astype(float).T+h['4.bias']
    return result,first,second


def main():
    p=argparse.ArgumentParser()
    for k in ('run','inputs','protocol'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());assert r['protocol_sha256']==sha(a.protocol) and r['source_sha256']==sha(Path(__file__).with_name('prepare_readout_fiber_rollout.py')) and r['projection_source_sha256']==sha(Path(__file__).with_name('readout_fiber.py'))
    assert len(r['records'])==1536;verified=[];max_region=0;max_readout=0
    for g in (r['index'],):
        src=Path(a.inputs)/f'job_{g}';source=json.loads((src/'report.json').read_text());binding=next(b for b in r['bindings'] if b.get('group')==g);assert sha(src/'report.json')==binding['report_sha256'] and sha(src/'acceptance.json')==binding['acceptance_sha256']
        for slot in (2,3,4):
            mi=8*g+slot;row=source['rows'][slot];b=next(b for b in r['bindings'] if b.get('model_index')==mi);assert sha(src/row['file'])==b['input_sha256'] and sha(src/row['head_file'])==b['head_sha256'] and sha(d/b['output_file'])==b['output_sha256'];head=dict(np.load(src/row['head_file']));z=np.load(src/row['file']);saved=np.load(d/b['output_file']);records=[x for x in r['records'] if x['model_index']==mi];assert len(records)==512
            for idx,record in enumerate(records):
                ref=idx//128;i=idx%128;h=0
                assert record['reference']==int(z['reference_routes'][ref]) and record['goal_index']==i and record['horizon']==int(z['horizons'][h]) and record['seed']==int(z['seeds'][i])
                pred=saved['predicted'][idx];obs=saved['observed'][idx];np.testing.assert_array_equal(pred,z['free_tokens'][ref,i,h]);np.testing.assert_array_equal(obs,z['observed_tokens'][ref,i,h]);baseline,a0,b0=forward(head,pred)
                vals={}
                for name in ('naive','constrained'):
                    value=saved[name][idx];assert value.dtype==np.float32 and np.isfinite(value).all();output,a1,b1=forward(head,value);error=float(np.max(abs(output-baseline)));valid=error<=1e-6 and (name=='naive' or record['solver']['status']=='solved');assert valid==record[name]['valid']
                    np.testing.assert_allclose(error,record[name]['normalized_readout_max_error'],atol=1e-12,rtol=1e-10)
                    before=np.sum(((pred.astype(float)-obs)/head['scale'])**2);after=np.sum(((value.astype(float)-obs)/head['scale'])**2)
                    np.testing.assert_allclose([before,after],[record[name]['squared_distance_before'],record[name]['squared_distance_after']],rtol=1e-10,atol=1e-10)
                    region=max(0.,float(np.max(-np.where(a0>=0,1.,-1.)*a1)),float(np.max(-np.where(b0>=0,1.,-1.)*b1)))
                    vals[name]=dict(valid=valid,normalized_readout_error=error,distance_reduction=1-float(after/before),region_violation=region)
                    if name=='constrained':max_region=max(max_region,region);max_readout=max(max_readout,error)
                verified.append(dict(model_index=mi,condition=record['condition'],**vals))
    summaries=[]
    for condition in ['unit_latent','unit_decoded_teacher','unit_physical_labels']:
        for method in ['naive','constrained']:
            values=[v[method] for v in verified if v['condition']==condition];distance=np.array([v['distance_reduction'] for v in values])
            summaries.append(dict(condition=condition,method=method,cases=len(values),valid=sum(v['valid'] for v in values),mean_squared_distance_reduction=float(distance.mean()),median_squared_distance_reduction=float(np.median(distance)),min_squared_distance_reduction=float(distance.min()),max_normalized_readout_error=max(v['normalized_readout_error'] for v in values)))
    result=dict(status='COMPLETE1536_FIRST_FEEDBACK_FIBER_CHECKS',index=r['index'],summaries=summaries,constrained_valid=sum(x['constrained']['valid'] for x in verified),max_constrained_region_violation=max_region,max_constrained_readout_error=max_readout,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='Independent saved-input and full nonlinear head checks after FP32 conversion. Region violation reported; no independent QP optimality certificate and no rollout/causal/utility result.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
