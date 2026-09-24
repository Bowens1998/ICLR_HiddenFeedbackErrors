"""Independent donor permutation, nonlinear-head and six-way norm verification."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_readout_fiber_rollout_inputs import forward


def main():
    p=argparse.ArgumentParser()
    for k in ('run','horizon','full','protocol'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());g=r['index'];hd=Path(a.horizon)/f'job_{g}';fd=Path(a.full)/f'job_{g}'
    assert r['status']=='SHUFFLED1536_PAIRED_FIBER_INPUTS_REQUIRES_ACCEPTANCE' and r['source_sha256']==sha(Path(__file__).with_name('prepare_shuffled_fiber_inputs.py')) and r['protocol_sha256']==sha(a.protocol) and r['projection_source_sha256']==sha(Path(__file__).with_name('readout_fiber.py'))
    assert r['full_report_sha256']==sha(fd/'report.json') and r['full_acceptance_sha256']==sha(fd/'acceptance.json');fac=json.loads((fd/'acceptance.json').read_text());assert fac['constrained_valid']==1536 and fac['report_sha256']==r['full_report_sha256']
    hb=r['bindings'][0];assert hb['report_sha256']==sha(hd/'report.json') and hb['acceptance_sha256']==sha(hd/'acceptance.json');hr=json.loads((hd/'report.json').read_text());fr=json.loads((fd/'report.json').read_text())
    rng=np.random.default_rng(1366001);permutation=rng.permutation(128)
    while np.any(permutation==np.arange(128)):permutation=rng.permutation(128)
    np.testing.assert_array_equal(permutation,r['permutation']);norms=[];entries=[];heads=[]
    for slot in (2,3,4):
        mi=8*g+slot;row=hr['rows'][slot];b=next(b for b in r['bindings'] if b.get('model_index')==mi);old=next(b for b in fr['bindings'] if b.get('model_index')==mi)
        assert b['output_sha256']==sha(d/b['output_file']) and b['full_file_sha256']==old['output_sha256']==sha(fd/b['full_file']) and b['input_sha256']==row['sha256']==sha(hd/row['file']) and b['head_sha256']==row['head_sha256']==sha(hd/row['head_file']);heads.append(b['head_sha256'])
        log=d/f'solver_attempts_{mi}.jsonl';assert sha(log)==b['solver_log_sha256'];solvers=[json.loads(line) for line in log.read_text().splitlines()];assert len(solvers)==512 and all(s['index']==i and s['status']=='solved' for i,s in enumerate(solvers))
        h=dict(np.load(hd/row['head_file']));s=dict(np.load(d/b['output_file']));f=dict(np.load(fd/b['full_file']));z=dict(np.load(hd/row['file']))
        for k in ('predicted','observed'):np.testing.assert_array_equal(s[k],f[k])
        np.testing.assert_array_equal(s['predicted'],z['free_tokens'][:,:,0].reshape(512,192));np.testing.assert_array_equal(s['observed'],z['observed_tokens'][:,:,0].reshape(512,192));np.testing.assert_array_equal(s['full'],f['constrained']);np.testing.assert_array_equal(s['permutation'],permutation);np.testing.assert_array_equal(s['donor'],s['observed'].reshape(4,128,192)[:,permutation].reshape(512,192))
        nn={}
        for name,full in [('constrained','full'),('shuffled','shuffled_full')]:nn[name]=np.sqrt(np.sum(((s[full].astype(float)-s['predicted'])/h['scale'])**2,-1));norms.append(nn[name])
        entries.append((mi,h,s,nn))
    assert len(set(heads))==1;common=np.minimum.reduce(norms);rows=[]
    for mi,h,s,nn in entries:
        v0,a0,b0=forward(h,s['predicted']);np.testing.assert_array_equal(s['target_norm'],common)
        for name,full in [('constrained','full'),('shuffled','shuffled_full')]:
            norm=nn[name];alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0);np.testing.assert_array_equal(alpha,s[name+'_alpha']);np.testing.assert_array_equal(norm,s[name+'_full_norm']);expected=(s['predicted'].astype(float)+alpha[:,None]*(s[full].astype(float)-s['predicted'])).astype(np.float32);np.testing.assert_array_equal(expected,s[name]);assert expected.shape==(512,192) and np.isfinite(expected).all()
            actual=np.sqrt(np.sum(((expected.astype(float)-s['predicted'])/h['scale'])**2,-1));np.testing.assert_allclose(actual,common,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(actual,s[name+'_actual_norm'])
            errors=[];regions=[]
            for value in (s[full],expected):
                v1,a1,b1=forward(h,value);error=float(np.max(abs(v1-v0)));region=max(0.,float(np.max(-np.where(a0>=0,1.,-1.)*a1)),float(np.max(-np.where(b0>=0,1.,-1.)*b1)));assert error<=1e-6 and region<=1e-6;errors.append(error);regions.append(region)
            rows.append(dict(model_index=mi,path=name,cases=512,max_scaled_readout_error=errors[1],max_full_readout_error=errors[0],max_region_violation=max(regions),max_norm_deviation=float(np.max(abs(actual-common))),mean_target_norm=float(common.mean()),mean_full_norm=float(norm.mean()),mean_alpha=float(alpha.mean())))
    result=dict(status='PASS1536_PAIRED_SHUFFLED_FIBER_INPUTS',index=g,constrained_valid=1536,shuffled_valid=1536,rows=rows,max_constrained_readout_error=max(x['max_scaled_readout_error'] for x in rows),max_region_violation=max(x['max_region_violation'] for x in rows),report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol))
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],g,flush=True)

if __name__=='__main__':main()
