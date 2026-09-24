"""Independent saved-array decoding and common-input checks; no neural replay."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def errors(pred,truth):
    d=np.sum((pred[...,2:4]-truth[...,2:4])**2,-1)
    angle=np.arctan2(pred[...,4],pred[...,5])-np.arctan2(truth[...,4],truth[...,5]);angle=np.arctan2(np.sin(angle),np.cos(angle))
    return dict(position_mse=d,angle_mse=angle**2,precision=(d<400)&(abs(angle)<np.pi/9))


def main():
    p=argparse.ArgumentParser()
    for k in ('run','plan','protocol'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());plan=json.loads(Path(a.plan).read_text());group=r['index']
    assert r['status']=='MATCHED_ACTION_EXTRACTION_REQUIRES_NUMPY_ACCEPTANCE' and r['plan_sha256']==sha(a.plan) and r['protocol_sha256']==sha(a.protocol)
    assert r['source_sha256']==sha(Path(__file__).with_name('run_task_coordinate_matched_native.py'))
    for name,h in r['sources'].items():assert sha(Path(__file__).with_name(name))==h
    assert r['anchor_checks']==512 and r['same_image_checks']==3840 and len(r['rows'])==8
    ref_routes=[16*group+k for k in (0,1,10,11)];assert [b['route'] for b in r['reference_bindings']]==ref_routes
    for b in r['reference_bindings']:
        assert b['model_index']==plan['routes'][b['route']]['model_index'] and [c['case'] for c in b['cases']]==list(range(128))
    common=None;observations={};rows=[]
    for slot,row in enumerate(r['rows']):
        mi=8*group+slot;assert row['model_index']==mi and row['entry']==plan['models'][mi] and row['frozen_tensors_unchanged'];e=row['entry'];fp=d/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp)
        assert z['predicted_pose'].shape==z['true_pose'].shape==z['real_pose'].shape==(4,128,6)
        assert z['predicted_tokens'].shape==z['real_tokens'].shape==(4,128,6 if e['score']=='state' else 192)
        np.testing.assert_array_equal(z['reference_routes'],ref_routes)
        identity=(z['true_pose'].copy(),z['seeds'].copy())
        if common is None:common=identity
        else:
            for x,y in zip(common,identity):np.testing.assert_array_equal(x,y)
        if e['score'] in observations:
            np.testing.assert_array_equal(z['real_tokens'],observations[e['score']][0]);np.testing.assert_array_equal(z['goal_tokens'],observations[e['score']][1])
        else:observations[e['score']]=(z['real_tokens'].copy(),z['goal_tokens'].copy())
        if row['head_file']:
            assert sha(d/row['head_file'])==row['head_sha256']==e['endpoint_head']['sha256'];head=dict(np.load(d/row['head_file']))
        for name in ('predicted','real'):
            token=z[name+'_tokens'].astype(float);assert np.isfinite(token).all()
            if row['head_file']:
                value=(token-head['mean'])/head['scale']
                for k in (0,2,4):
                    value=value@head[f'{k}.weight'].astype(float).T+head[f'{k}.bias'].astype(float)
                    if k!=4:value=np.maximum(value,0)
                value=value*head['target_scale']+head['target_mean']
            else:value=token*np.asarray(e['target_normalization']['std'])+np.asarray(e['target_normalization']['mean'])
            np.testing.assert_allclose(value,z[name+'_pose'],rtol=1e-10,atol=1e-8)
        metrics=errors(z['predicted_pose'],z['true_pose']);metrics['token_mse']=np.mean((z['predicted_tokens'].astype(float)-z['real_tokens'].astype(float))**2,-1)
        real_metrics=errors(z['real_pose'],z['true_pose'])
        rows.append(dict(model_index=mi,condition=e['adaptation_condition'],recipe=e['score'],arm=e['arm'],reference_routes=ref_routes,metrics={k:v.tolist() for k,v in metrics.items()},real_image_metrics={k:v.tolist() for k,v in real_metrics.items()},file_sha256=sha(fp)))
    result=dict(status='PASS_ALL8_MATCHED_ACTION_NUMPY_RECONSTRUCTIONS',index=group,rows=rows,seeds=common[1].tolist(),report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol),scope='All saved token-to-pose decoding reconstructed independently; same true endpoints/seeds/reference streams and unchanged image encodings checked across conditions. Neural original-actor anchors/frozen tensors and reference physical truth retain bound extraction/earlier replay scope; not independent neural or simulator implementation.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS_MATCHED_ACTIONS',group,flush=True)

if __name__=='__main__':main()
