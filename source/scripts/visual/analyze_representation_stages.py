"""Complete fixed-layer contrast on512 shared independent static images."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha,metrics
from analyze_coverage_iid import summarize
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser()
    for k in ['evaluations','fits','bank','plan','config','holdout-config','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();bank=Path(a.bank);bm=json.loads((bank/'report.json').read_text());ba=json.loads((bank/'acceptance.json').read_text())
    assert ba['status']=='PASS' and ba['report_sha256']==sha(bank/'report.json')
    assert ba['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert bm['config_sha256']==sha(a.holdout_config) and bm['replica']==0
    row=next(v for v in bm['rows'] if v['arm']=='broad');assert row['count']==512
    for name,expected in row['files_sha256'].items():assert sha(bank/'broad'/name)==expected
    truth=np.load(bank/'broad/poses.npz');stages=['trained_cls','projected','initial_cls']
    indices=np.random.default_rng(1255901).integers(0,512,(10000,512))
    all_values={s:[] for s in stages};models=[];provenance=[]
    for i in range(6):
        run=Path(a.evaluations)/f'job_{i}';fit=Path(a.fits)/f'job_{i}'
        r=json.loads((run/'report.json').read_text());fr=json.loads((fit/'report.json').read_text());fa=json.loads((fit/'acceptance.json').read_text())
        assert r['status']=='COMPLETE_REPRESENTATION_EVALUATION' and r['index']==fr['index']==fa['index']==i
        assert fa['status']=='PASS' and fa['report_sha256']==r['fit_report_sha256']==sha(fit/'report.json')
        assert fa['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_readouts.py'))
        assert r['fit_acceptance_sha256']==sha(fit/'acceptance.json')
        for key,path in [('plan_sha256',a.plan),('config_sha256',a.config),('holdout_config_sha256',a.holdout_config),('bank_report_sha256',bank/'report.json'),('bank_acceptance_sha256',bank/'acceptance.json')]:assert r[key]==sha(path)
        assert r['source_sha256']==sha(Path(__file__).with_name('evaluate_representation_stages.py'))
        assert r['features_sha256']==sha(run/'features.npz')
        z=dict(np.load(run/'features.npz'))
        for k in ['target','source_index']:np.testing.assert_array_equal(z[k],truth[k])
        assert z['target'].shape==(512,6) and [v['stage'] for v in r['rows']]==stages
        values={}
        for row in r['rows']:
            stage=row['stage'];head=fit/stage/'weights.npz'
            fitrow=next(v for v in fr['rows'] if v['stage']==stage)
            assert sha(head)==row['head_sha256']==fitrow['files_sha256']['weights.npz']
            assert row['file']==f'{stage}.npz' and sha(run/row['file'])==row['file_sha256']
            pred=np.load(run/row['file'])['prediction'];assert pred.shape==(512,6) and np.isfinite(pred).all()
            np.testing.assert_allclose(pred,numpy_pose(z[stage],dict(np.load(head))),rtol=1e-10,atol=1e-9)
            values[stage]=metrics(pred,z['target']);all_values[stage].append(values[stage])
        models.append(dict(index=i,metrics={s:{k:summarize(v,indices) for k,v in values[s].items()} for s in stages},cls_minus_projected={k:summarize(values['trained_cls'][k]-values['projected'][k],indices) for k in values['trained_cls']}))
        provenance.append(dict(index=i,report_sha256=sha(run/'report.json')))
    pooled={s:{k:np.stack([v[k] for v in all_values[s]]).mean(0) for k in all_values[s][0]} for s in stages}
    contrasts={f'{left}_minus_{right}':{k:summarize(pooled[left][k]-pooled[right][k],indices) for k in pooled[left]} for left,right in [('trained_cls','projected'),('initial_cls','projected'),('trained_cls','initial_cls')]}
    result=dict(status='PASS_COMPLETE18_CONDITIONS',images=512,models=6,provenance=provenance,per_model=models,pooled_metrics={s:{k:summarize(v,indices) for k,v in pooled[s].items()} for s in stages},contrasts=contrasts,primary='equal-six trained_cls-minus-projected joint precision',bootstrap=dict(seed=1255901,draws=10000,unit='shared independent static image',interval='percentile95'),source_sha256=sha(__file__),bank_report_sha256=sha(bank/'report.json'),scope='Fixed512-label probe comparison; conditional on fixed models/pools. Shared-seed initial encoders are not independent pretrained replicas. Secondary intervals nominal. Does not establish information loss, a deployable method, or planning advantage. Image features inherit native evaluator; saved heads independently recomputed.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('PASS_COMPLETE18_CONDITIONS')


if __name__=='__main__':main()
