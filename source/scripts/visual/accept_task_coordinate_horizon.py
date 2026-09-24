"""Independent array decoding and exact horizon anchors; no neural replay."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_task_coordinate_matched_native import errors


def main():
    p=argparse.ArgumentParser()
    for k in ('run','plan','protocol','matched'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());plan=json.loads(Path(a.plan).read_text());g=r['index']
    assert r['status']=='HORIZON_EXTRACTION_REQUIRES_NUMPY_ACCEPTANCE' and r['plan_sha256']==sha(a.plan) and r['protocol_sha256']==sha(a.protocol)
    assert r['source_sha256']==sha(Path(__file__).with_name('run_task_coordinate_horizon.py'))
    for name,h in r['sources'].items():assert sha(Path(__file__).with_name(name))==h
    assert r['anchor_checks']==r['first_step_checks']==r['observed_terminal_checks']==4096 and r['same_image_checks']==16128 and len(r['rows'])==8
    routes=[16*g+k for k in (0,1,10,11)];assert [x['route'] for x in r['reference_bindings']]==routes==[x['route'] for x in r['trajectory_bindings']]
    md=Path(a.matched)/f'job_{g}';mr=json.loads((md/'report.json').read_text());mac=json.loads((md/'acceptance.json').read_text())
    assert mac['status']=='PASS_ALL8_MATCHED_ACTION_NUMPY_RECONSTRUCTIONS' and mac['report_sha256']==sha(md/'report.json') and mac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_matched_native.py'))
    common=None;observations={};rows=[]
    for slot,row in enumerate(r['rows']):
        mi=8*g+slot;e=plan['models'][mi];assert row['model_index']==mi and row['entry']==e and row['frozen_tensors_unchanged'];fp=d/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp)
        assert z['true_pose'].shape==(4,128,5,6);np.testing.assert_array_equal(z['reference_routes'],routes);np.testing.assert_array_equal(z['horizons'],[5,10,15,20,25]);assert z['seeds'].shape==(128,)
        mrow=mr['rows'][slot];binding=r['matched_bindings'][slot];assert binding['model_index']==mi and mrow['entry']==e and binding['file_sha256']==mrow['sha256']==sha(md/mrow['file'])
        assert binding['report_sha256']==sha(md/'report.json') and binding['acceptance_sha256']==sha(md/'acceptance.json');m=np.load(md/mrow['file'])
        for key,other in [('seeds','seeds'),('reference_routes','reference_routes'),('goal_tokens','goal_tokens')]:np.testing.assert_array_equal(z[key],m[other])
        np.testing.assert_array_equal(z['free_tokens'][:,:,-1],m['predicted_tokens']);np.testing.assert_array_equal(z['observed_tokens'][:,:,-1],m['real_tokens']);np.testing.assert_array_equal(z['true_pose'][:,:,-1],m['true_pose']);np.testing.assert_array_equal(z['free_tokens'][:,:,0],z['teacher_tokens'][:,:,0])
        identity=(z['true_pose'].copy(),z['seeds'].copy())
        if common is None:common=identity
        else:
            for x,y in zip(common,identity):np.testing.assert_array_equal(x,y)
        if e['score'] in observations:
            np.testing.assert_array_equal(z['observed_tokens'],observations[e['score']][0]);np.testing.assert_array_equal(z['goal_tokens'],observations[e['score']][1])
        else:observations[e['score']]=(z['observed_tokens'].copy(),z['goal_tokens'].copy())
        if row['head_file']:
            assert sha(d/row['head_file'])==row['head_sha256']==e['endpoint_head']['sha256'];head=dict(np.load(d/row['head_file']))
        metrics={}
        for name in ('free','teacher','observed'):
            token=z[name+'_tokens'].astype(float);assert token.shape==(4,128,5,6 if e['score']=='state' else 192) and np.isfinite(token).all()
            if row['head_file']:
                value=(token-head['mean'])/head['scale']
                for k in (0,2,4):
                    value=value@head[f'{k}.weight'].astype(float).T+head[f'{k}.bias'].astype(float)
                    if k!=4:value=np.maximum(value,0)
                value=value*head['target_scale']+head['target_mean']
            else:value=token*np.asarray(e['target_normalization']['std'])+np.asarray(e['target_normalization']['mean'])
            assert z[name+'_pose'].shape==(4,128,5,6);np.testing.assert_allclose(value,z[name+'_pose'],rtol=1e-10,atol=1e-8)
            mm=errors(value,z['true_pose']);mm['token_mse']=np.mean((token-z['observed_tokens'].astype(float))**2,-1)
            metrics[name]={k:v.tolist() for k,v in mm.items()}
        rows.append(dict(model_index=mi,condition=e['adaptation_condition'],recipe=e['score'],arm=e['arm'],metrics=metrics,file_sha256=sha(fp)))
    result=dict(status='PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS',index=g,rows=rows,seeds=common[1].tolist(),horizons=[5,10,15,20,25],report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol),scope='Independent NumPy decodes/metrics and saved-array matched endpoint, common truth, first-step and same-image checks. Neural and physical replay inherit bound extraction scope.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],g,flush=True)

if __name__=='__main__':main()
