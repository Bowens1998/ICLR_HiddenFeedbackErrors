"""Independent physical decoding, one-reset identity and free trajectory anchors."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_task_coordinate_matched_native import errors


def main():
    p=argparse.ArgumentParser()
    for k in ('run','plan','protocol','horizon','fibers'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());plan=json.loads(Path(a.plan).read_text());g=r['index'];hd=Path(a.horizon)/f'job_{g}';fd=Path(a.fibers)/f'job_{g}'
    assert r['status']=='MATCHED_FIBER_ROLLOUT_REQUIRES_NUMPY_ACCEPTANCE' and r['plan_sha256']==sha(a.plan) and r['protocol_sha256']==sha(a.protocol) and r['source_sha256']==sha(Path(__file__).with_name('run_matched_fiber_rollout.py'))
    assert r['anchor_checks']==r['readout_checks']==1536 and r['goal_checks']==384 and len(r['rows'])==3
    for name,h in r['sources'].items():assert sha(Path(__file__).with_name(name))==h
    hr=json.loads((hd/'report.json').read_text());fr=json.loads((fd/'report.json').read_text());fac=json.loads((fd/'acceptance.json').read_text());assert fac['constrained_valid']==1536 and fac['report_sha256']==sha(fd/'report.json')
    rows=[];common=None;maxerr=0
    for k,slot in enumerate((2,3,4)):
        row=r['rows'][k];mi=8*g+slot;e=plan['models'][mi];assert row['model_index']==mi and row['entry']==e and row['frozen_tensors_unchanged'];fp=d/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp);b=r['input_bindings'][k];hb=hr['rows'][slot];fb=next(x for x in fr['bindings'] if x.get('model_index')==mi)
        assert b['model_index']==mi and b['horizon_report_sha256']==sha(hd/'report.json') and b['horizon_acceptance_sha256']==sha(hd/'acceptance.json') and b['fiber_report_sha256']==sha(fd/'report.json') and b['fiber_acceptance_sha256']==sha(fd/'acceptance.json')
        assert b['horizon_file_sha256']==sha(hd/hb['file']) and b['fiber_file_sha256']==sha(fd/fb['output_file']);hz=np.load(hd/hb['file']);fz=np.load(fd/fb['output_file'])
        for name in ('true_pose','observed_tokens','observed_pose','horizons','seeds','reference_routes','goal_tokens','free_tokens'):np.testing.assert_array_equal(z[name],hz[name])
        np.testing.assert_array_equal(z['fiber_tokens'][:,:,0],fz['constrained'].reshape(4,128,192));np.testing.assert_array_equal(z['reset_tokens'][:,:,0],z['observed_tokens'][:,:,0])
        assert z['true_pose'].shape==(4,128,5,6);np.testing.assert_array_equal(z['horizons'],[5,10,15,20,25]);np.testing.assert_array_equal(z['reference_routes'],[16*g+j for j in (0,1,10,11)])
        if common is None:common=(z['true_pose'].copy(),z['seeds'].copy())
        else:
            for x,y in zip(common,(z['true_pose'],z['seeds'])):np.testing.assert_array_equal(x,y)
        assert sha(d/row['head_file'])==row['head_sha256']==e['endpoint_head']['sha256'];head=dict(np.load(d/row['head_file']));metrics={};normalized={}
        for path in ('free','fiber','reset','observed'):
            token=z[path+'_tokens'].astype(float);assert token.shape==(4,128,5,192) and np.isfinite(token).all();v=(token-head['mean'])/head['scale']
            for j in (0,2,4):
                v=v@head[f'{j}.weight'].astype(float).T+head[f'{j}.bias'].astype(float)
                if j!=4:v=np.maximum(v,0)
            normalized[path]=v;pose=v*head['target_scale']+head['target_mean'];np.testing.assert_allclose(pose,z[path+'_pose'],rtol=1e-10,atol=1e-8)
            mm=errors(pose,z['true_pose']);mm['token_mse']=np.mean((token-z['observed_tokens'].astype(float))**2,-1);metrics[path]={name:x.tolist() for name,x in mm.items()}
        err=float(np.max(abs(normalized['fiber'][:,:,0]-normalized['free'][:,:,0])));assert err<=1e-6;maxerr=max(maxerr,err)
        rows.append(dict(model_index=mi,condition=e['adaptation_condition'],recipe=e['score'],arm=e['arm'],metrics=metrics,file_sha256=sha(fp),max_initial_normalized_readout_error=err))
    result=dict(status='PASS_ALL3_MATCHED_FIBER_ROLLOUT_RECONSTRUCTIONS',index=g,rows=rows,seeds=common[1].tolist(),horizons=[5,10,15,20,25],max_initial_normalized_readout_error=maxerr,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol),scope='Independent NumPy decode/metrics and exact saved-array baseline, correction and reset identities. Full neural free trajectory anchored by extraction; modified neural trajectories are not independently reimplemented.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],g,flush=True)

if __name__=='__main__':main()
