"""Verify fixed-stage probe artifacts and recompute training predictions/norms."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser()
    for k in ['run','features','config']:p.add_argument('--'+k,required=True)
    a=p.parse_args();run=Path(a.run);f=Path(a.features)
    r=json.loads((run/'report.json').read_text());fr=json.loads((f/'report.json').read_text());fa=json.loads((f/'acceptance.json').read_text())
    assert r['status']=='FITTED_REQUIRES_ACCEPTANCE' and fa['status']=='PASS'
    assert r['index']==fr['index']==fa['index']
    assert r['source_sha256']==sha(Path(__file__).with_name('fit_representation_readouts.py'))
    assert r['feature_report_sha256']==fa['report_sha256']==sha(f/'report.json')
    assert r['feature_acceptance_sha256']==sha(f/'acceptance.json')
    assert fa['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_stages.py'))
    assert r['features_sha256']==fa['features_sha256']==sha(f/'features.npz')
    assert r['config_sha256']==fr['config_sha256']==sha(a.config)
    assert [v['stage'] for v in r['rows']]==['trained_cls','projected','initial_cls']
    z=dict(np.load(f/'features.npz'));reference=None;rows=[]
    for row in r['rows']:
        stage=row['stage'];sub=run/stage
        assert row['labels']==512 and row['updates']==2000 and row['seed']==430001+r['index']
        assert row['batch_size']==256 and row['learning_rate']==.001
        for name,expected in row['files_sha256'].items():assert sha(sub/name)==expected
        initial=dict(np.load(sub/'initial.npz'))
        if reference is None:reference=initial
        else:
            for k in initial:np.testing.assert_array_equal(initial[k],reference[k])
        h=dict(np.load(sub/'weights.npz'));x=z[stage].astype(np.float64)
        np.testing.assert_array_equal(h['mean'],x.mean(0))
        np.testing.assert_array_equal(h['scale'],np.maximum(x.std(0),1e-6))
        np.testing.assert_array_equal(h['target_mean'],[256.,256.,256.,256.,0.,0.])
        np.testing.assert_array_equal(h['target_scale'],[256.,256.,256.,256.,1.,1.])
        pred=numpy_pose(x,h);saved=np.load(sub/'train_prediction.npz')['prediction']
        np.testing.assert_allclose(pred,saved,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(np.mean(((pred-z['target'])/h['target_scale'])**2),row['train_standardized_mse'],rtol=1e-10,atol=1e-12)
        rows.append(dict(stage=stage,labels=512,prediction_max_difference=float(abs(pred-saved).max())))
    result=dict(status='PASS',index=r['index'],report_sha256=sha(run/'report.json'),verifier_sha256=sha(__file__),rows=rows,scope='All weights/artifact hashes, shared initialization, exact training-only normalizers, physical target scales and independently recomputed NumPy training predictions. Does not replay2000 optimizer updates or establish generalization.')
    with (run/'acceptance.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print('PASS_THREE_FIXED_PROBES',r['index'])


if __name__=='__main__':main()
