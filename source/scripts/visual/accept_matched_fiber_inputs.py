"""Independent paired-distance, interpolation, nonlinear-head and region checks."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_readout_fiber_rollout_inputs import forward


def main():
    p=argparse.ArgumentParser()
    for k in ('run','horizon','full','protocol'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());g=r['index'];hd=Path(a.horizon)/f'job_{g}';fd=Path(a.full)/f'job_{g}'
    assert r['status']=='MATCHED1536_FIBER_INPUTS_REQUIRES_ACCEPTANCE' and r['source_sha256']==sha(Path(__file__).with_name('prepare_matched_fiber_inputs.py')) and r['protocol_sha256']==sha(a.protocol)
    assert r['full_report_sha256']==sha(fd/'report.json') and r['full_acceptance_sha256']==sha(fd/'acceptance.json')
    ac=json.loads((fd/'acceptance.json').read_text());assert ac['constrained_valid']==1536 and ac['report_sha256']==r['full_report_sha256']
    hr=json.loads((hd/'report.json').read_text());fr=json.loads((fd/'report.json').read_text());hb=r['bindings'][0]
    assert hb['report_sha256']==sha(hd/'report.json') and hb['acceptance_sha256']==sha(hd/'acceptance.json')
    allnorm=[];entries=[];head_hashes=[]
    for slot in (2,3,4):
        mi=8*g+slot;row=hr['rows'][slot];b=next(b for b in r['bindings'] if b.get('model_index')==mi);old=next(b for b in fr['bindings'] if b.get('model_index')==mi)
        assert b['output_sha256']==sha(d/b['output_file']) and b['full_file_sha256']==old['output_sha256']==sha(fd/b['full_file'])
        assert b['input_sha256']==row['sha256']==sha(hd/row['file']) and b['head_sha256']==row['head_sha256']==sha(hd/row['head_file']);head_hashes.append(b['head_sha256'])
        h=dict(np.load(hd/row['head_file']));s=dict(np.load(d/b['output_file']));f=dict(np.load(fd/b['full_file']));z=dict(np.load(hd/row['file']))
        for k in ('predicted','observed'):np.testing.assert_array_equal(s[k],f[k])
        np.testing.assert_array_equal(s['predicted'],z['free_tokens'][:,:,0].reshape(512,192));np.testing.assert_array_equal(s['observed'],z['observed_tokens'][:,:,0].reshape(512,192));np.testing.assert_array_equal(s['full'],f['constrained'])
        norm=np.sqrt(np.sum(((f['constrained'].astype(float)-f['predicted'])/h['scale'])**2,-1));allnorm.append(norm);entries.append((mi,h,s,norm))
    assert len(set(head_hashes))==1;common=np.minimum.reduce(allnorm);rows=[]
    for mi,h,s,norm in entries:
        alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0);np.testing.assert_array_equal(alpha,s['alpha']);np.testing.assert_array_equal(common,s['target_norm']);np.testing.assert_array_equal(norm,s['full_norm'])
        expected=(s['predicted'].astype(float)+alpha[:,None]*(s['full'].astype(float)-s['predicted'])).astype(np.float32);np.testing.assert_array_equal(expected,s['constrained']);assert expected.dtype==np.float32 and expected.shape==(512,192) and np.isfinite(expected).all()
        actual=np.sqrt(np.sum(((expected.astype(float)-s['predicted'])/h['scale'])**2,-1));np.testing.assert_allclose(actual,common,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(actual,s['actual_norm'])
        v0,a0,b0=forward(h,s['predicted']);v1,a1,b1=forward(h,expected);error=float(np.max(abs(v1-v0)));region=max(0.,float(np.max(-np.where(a0>=0,1.,-1.)*a1)),float(np.max(-np.where(b0>=0,1.,-1.)*b1)))
        assert error<=1e-6 and region<=1e-6
        rows.append(dict(model_index=mi,cases=512,max_normalized_readout_error=error,max_region_violation=region,max_norm_deviation=float(np.max(abs(actual-common))),mean_target_norm=float(common.mean()),mean_actual_norm=float(actual.mean()),mean_full_norm=float(norm.mean()),mean_alpha=float(alpha.mean())))
    result=dict(status='PASS1536_MATCHED_FIBER_INPUTS',index=g,constrained_valid=1536,rows=rows,max_constrained_readout_error=max(x['max_normalized_readout_error'] for x in rows),max_constrained_region_violation=max(x['max_region_violation'] for x in rows),report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol))
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],g,flush=True)

if __name__=='__main__':main()
