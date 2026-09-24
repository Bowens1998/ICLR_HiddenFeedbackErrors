"""Independent NumPy reconstruction of PushT pose fitting artifacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--features',required=True);a=p.parse_args()
    out=Path(a.run);r=json.loads((out/'report.json').read_text());folder=Path(a.features)/f"job_{r['index']}"
    fr=json.loads((folder/'report.json').read_text());fa=json.loads((folder/'acceptance.json').read_text())
    assert r['protocol']=='pusht_nonlinear_pose_fit_v1' and fr['protocol']=='pusht_nonlinear_transfer_features_v1'
    assert not fr['engineering'] and not fa['engineering'] and fa['status']=='PASS'
    assert fa['report_sha256']==sha(folder/'report.json')==r['input_report_sha256']
    assert sha(folder/'acceptance.json')==r['input_acceptance_sha256']
    assert fa['verifier_sha256']==sha(Path(__file__).with_name('accept_pose_features.py'))
    assert (r['index'],r['replica'],r['arm'])==(fr['index'],fr['replica'],fr['arm'])
    assert r['input_files_sha256']==fr['files_sha256']
    for f,h in r['input_files_sha256'].items():assert sha(folder/f)==h
    for f,h in r['source_sha256'].items():assert sha(Path(__file__).with_name(f))==h
    data={split:np.load(folder/f'{split}_features.npz') for split in ['train','validation']}
    target=data['train']['target'];tm=np.mean(target,axis=0);ts=np.std(target,axis=0);ts=np.where(ts>1e-12,ts,1.)
    assert len(r['rows'])==2 and [x['role'] for x in r['rows']]==['encoded','predicted']
    initial_reference=None;rows=[]
    for row in r['rows']:
        role=row['role'];sub=out/role
        assert row['updates']==(20 if r['engineering'] else 2000) and row['batch_size']==256
        assert row['seed']==430001+r['index'] and np.isfinite([row['first_loss'],row['last_loss']]).all()
        assert set(row['files_sha256'])=={'weights.npz','initial.npz','validation_encoded.npz','validation_predicted.npz'}
        for f,h in row['files_sha256'].items():assert sha(sub/f)==h
        h=np.load(sub/'weights.npz');initial=np.load(sub/'initial.npz')
        x=data['train'][role].astype(np.float64);mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
        for key,value in [('mean',mean),('scale',scale),('target_mean',tm),('target_scale',ts)]:np.testing.assert_array_equal(h[key],value)
        keys=[]
        for i,nin,nout in [(0,192,256),(2,256,256),(4,256,6)]:
            for suffix,shape in [('weight',(nout,nin)),('bias',(nout,))]:
                key=f'{i}.{suffix}';keys.append(key)
                assert h[key].shape==initial[key].shape==shape and h[key].dtype==initial[key].dtype==np.float32
                assert np.isfinite(h[key]).all() and np.isfinite(initial[key]).all()
        assert set(initial.files)==set(keys) and set(h.files)==set(keys+['mean','scale','target_mean','target_scale'])
        if initial_reference is None:initial_reference={k:initial[k] for k in keys}
        else:
            for k in keys:np.testing.assert_array_equal(initial[k],initial_reference[k])
        for source in ['encoded','predicted']:
            v=np.load(sub/f'validation_{source}.npz');np.testing.assert_array_equal(v['identity'],data['validation']['identity'])
            y=(data['validation'][source].astype(np.float64)-mean)/scale
            for i in [0,2,4]:
                y=y@h[f'{i}.weight'].astype(np.float64).T+h[f'{i}.bias'].astype(np.float64)
                if i<4:y=np.maximum(y,0)
            pred=y*ts+tm
            assert pred.shape==data['validation']['target'].shape and np.isfinite(pred).all()
            np.testing.assert_allclose(v['prediction'],pred,rtol=1e-10,atol=1e-9)
            mse=float(np.square((pred-data['validation']['target'])/ts).mean())
            np.testing.assert_allclose(mse,row['validation_standardized_mse'][source],rtol=1e-10,atol=1e-12)
            rows.append(dict(role=role,source=source,examples=len(pred),standardized_mse=mse))
    result=dict(status='PASS',engineering=r['engineering'],report_sha256=sha(out/'report.json'),verifier_sha256=sha(__file__),rows=rows,
                scope='Source/input/file binding, train-only statistics, identical role initialization, all cross-input validation predictions and metrics reconstructed; does not replay training updates or prove planning benefit.')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
