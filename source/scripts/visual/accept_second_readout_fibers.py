"""Independent full-head, region, source and six-direction matching checks."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_readout_fiber_rollout_inputs import forward


def accept(root,protocol,check_source=True):
    d=Path(root);r=json.loads((d/'report.json').read_text());assert r['protocol_sha256']==sha(protocol)
    assert r['source_sha256']==sha(Path(__file__).with_name('prepare_second_readout_fibers.py'))
    assert r['projection_source_sha256']==sha(Path(__file__).with_name('readout_fiber.py'))
    assert r['head_sha256']==sha(d/'head.npz');h=dict(np.load(d/'head.npz'));per=np.random.default_rng(1368001).permutation(128)
    assert len(r['rows'])==3 and [x['objective'] for x in r['rows']]==['latent','decoded_teacher','physical_labels']
    entries=[];norms=[];rows=[]
    for row in r['rows']:
        assert sha(d/row['file'])==row['sha256'];z=dict(np.load(d/row['file']));np.testing.assert_array_equal(z['permutation'],per)
        assert z['free_baseline_tokens'].shape==(4,128,5,192)
        np.testing.assert_array_equal(z['predicted'],z['free_baseline_tokens'][:,:,0].reshape(512,192))
        np.testing.assert_array_equal(z['observed'],z['observed_tokens'][:,:,0].reshape(512,192))
        if check_source:
            for path,key in [(row['source_file'],'source_sha256'),(row['donor_file'],'donor_sha256'),(row['weights_path'],'weights_sha256')]:assert sha(path)==row[key]
            src=dict(np.load(row['source_file']));don=dict(np.load(row['donor_file']))
            for key in ['observed_tokens','true_pose','seeds','reference_routes']:np.testing.assert_array_equal(z[key],src[key])
            np.testing.assert_array_equal(z['free_baseline_tokens'],src['free_tokens'])
            assert not set(src['seeds'])&set(don['seeds'])
            np.testing.assert_array_equal(z['donor'],don['observed_tokens'][:,:,0][:,per].reshape(512,192))
        for b in ['actual','donor']:
            log=d/f"{row['objective']}_{b}_solver.jsonl";assert sha(log)==row['solver_sha256'][b]
            attempts=[json.loads(x) for x in log.read_text().splitlines()];assert len(attempts)==512 and all(x['index']==i and x['status']=='solved' for i,x in enumerate(attempts))
            norm=np.sqrt(np.square((z[b+'_full'].astype(float)-z['predicted'])/h['scale']).sum(-1));np.testing.assert_array_equal(norm,z[b+'_full_norm']);norms.append(norm)
        entries.append((row,z))
    common=np.minimum.reduce(norms)
    for row,z in entries:
        np.testing.assert_array_equal(common,z['target_norm']);v0,a0,b0=forward(h,z['predicted'])
        for b in ['actual','donor']:
            norm=z[b+'_full_norm'];alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0);np.testing.assert_array_equal(alpha,z[b+'_alpha'])
            expected=(z['predicted'].astype(float)+alpha[:,None]*(z[b+'_full'].astype(float)-z['predicted'])).astype(np.float32)
            np.testing.assert_array_equal(expected,z[b+'_matched']);actual=np.sqrt(np.square((expected.astype(float)-z['predicted'])/h['scale']).sum(-1));np.testing.assert_allclose(actual,common,rtol=1e-6,atol=1e-6)
            errors=[];regions=[]
            for key in [b+'_full',b+'_matched']:
                v,a1,b1=forward(h,z[key]);errors.append(float(abs(v-v0).max()));regions.append(float(max(0,np.max(-np.where(a0>=0,1,-1)*a1),np.max(-np.where(b0>=0,1,-1)*b1))))
            assert max(errors)<=1e-6 and max(regions)<=1e-6
            rows.append({'objective':row['objective'],'source':b,'full_readout_error':errors[0],'matched_readout_error':errors[1],'region_violation':max(regions),'norm_max_deviation':float(abs(actual-common).max())})
    result={'status':'PASS_SECOND_HEAD_3072_PROJECTIONS_AND_MATCHING','index':r['index'],'rows':rows,'report_sha256':sha(d/'report.json'),'source_sha256':sha(__file__),'protocol_sha256':sha(protocol),'source_files_checked':check_source,'mean_target_norm':float(common.mean()),'min_target_norm':float(common.min()),'max_target_norm':float(common.max())}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--protocol',required=True);a=p.parse_args();r=accept(a.run,a.protocol)
    (Path(a.run)/'acceptance.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['index'])
