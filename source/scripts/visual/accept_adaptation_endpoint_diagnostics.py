"""Independent NumPy decoding/cost reconstruction of every endpoint diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--plan',required=True);a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());plan=json.loads(Path(a.plan).read_text())
    assert r['status']=='EXTRACTED_REQUIRES_NUMERIC_ACCEPTANCE' and r['plan_sha256']==sha(a.plan)
    assert r['source_sha256']==sha(Path(__file__).with_name('extract_adaptation_endpoint_diagnostics.py'))
    assert r['same_image_checks']==1536 and r['max_same_image_difference']==0 and len(r['rows'])==6
    for offset,row in enumerate(r['rows']):
        route=6*r['index']+offset;assert row['route']==route;e=plan['models'][plan['routes'][route]['model_index']];assert row['entry']==e
        fp=d/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp);assert len(z['seed'])==len(row['cases'])==128
        assert z['seed'].tolist()==[c['seed'] for c in row['cases']]
        if row['head_file']:
            assert sha(d/row['head_file'])==row['head_sha256']==e['goal_head']['sha256'];head=dict(np.load(d/row['head_file']))
        for name in ('imagined','real','goal'):
            token=z[name+'_tokens'].astype(np.float64)
            if row['head_file']:
                output=(token-head['mean'])/head['scale']
                for layer in (0,2,4):
                    output=output@head[f'{layer}.weight'].astype(float).T+head[f'{layer}.bias'].astype(float)
                    if layer!=4:output=np.maximum(output,0)
                output=output*head['target_scale']+head['target_mean']
            else:output=token*np.asarray(e['target_normalization']['std'])+np.asarray(e['target_normalization']['mean'])
            np.testing.assert_allclose(output,z[name+'_pose'],rtol=1e-10,atol=1e-8)
        for first,second,key in [('imagined_pose','goal_pose','predicted_cost'),('true_endpoint','true_goal','realized_cost')]:
            x,y=z[first],z[second];angle=np.arctan2(x[:,4],x[:,5])-np.arctan2(y[:,4],y[:,5]);angle=np.arctan2(np.sin(angle),np.cos(angle));cost=np.sum((x[:,2:4]-y[:,2:4])**2,axis=1)+900*angle**2
            np.testing.assert_allclose(cost,z[key],rtol=2e-5 if key=='predicted_cost' else 1e-10,atol=2e-3 if key=='predicted_cost' else 1e-7)
            if key=='realized_cost':np.testing.assert_array_equal((np.linalg.norm(x[:,2:4]-y[:,2:4],axis=1)<20)&(abs(angle)<np.pi/9),z['success'])
    result=dict(status='PASS_ALL_ENDPOINT_DECODE_AND_COST_RECONSTRUCTIONS',index=r['index'],routes=6,cases=768,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='All saved tokens decoded and physical/predicted costs/success reconstructed independently with NumPy. Same-image neural checks and frozen tensors are executed by the bound extraction; this verifier does not rerun neural encoding or physics.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS',r['index'],flush=True)

if __name__=='__main__':main()
