"""Exploratory deployment-observable support-distance baseline; no tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha,metrics
from nonlinear_pose_cost import numpy_pose


def auc(success,risk):
    # Higher risk should identify failures. Pairwise ties receive half credit.
    bad=np.asarray(risk)[~np.asarray(success,dtype=bool)]
    good=np.asarray(risk)[np.asarray(success,dtype=bool)]
    if not len(bad) or not len(good):return None
    return float(((bad[:,None]>good[None,:])+.5*(bad[:,None]==good[None,:])).mean())


def support_distance(query,train,mean,scale):
    t=(np.asarray(train,dtype=float)-mean)/scale
    q=(np.asarray(query,dtype=float)-mean)/scale
    out=[]
    for start in range(0,len(q),32):
        d=((q[start:start+32,None]-t[None])**2).mean(-1)
        out.extend(d.min(1))
    return np.array(out)


def main():
    p=argparse.ArgumentParser()
    for k in ['features','fits','evaluations','summary','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();s=json.loads(Path(a.summary).read_text());assert s['status']=='PASS_COMPLETE18_CONDITIONS'
    rows=[];arrays={};stages=['trained_cls','projected','initial_cls']
    for i in range(6):
        ev=Path(a.evaluations)/f'job_{i}';fit=Path(a.fits)/f'job_{i}';train=Path(a.features)/f'job_{i}'
        er=json.loads((ev/'report.json').read_text());fr=json.loads((fit/'report.json').read_text())
        assert sha(ev/'report.json')==s['provenance'][i]['report_sha256']
        assert sha(fit/'report.json')==er['fit_report_sha256']
        assert sha(train/'features.npz')==fr['features_sha256']
        assert sha(ev/'features.npz')==er['features_sha256']
        tr=dict(np.load(train/'features.npz'));te=dict(np.load(ev/'features.npz'))
        for stage in stages:
            hp=fit/stage/'weights.npz';row=next(v for v in er['rows'] if v['stage']==stage)
            assert sha(hp)==row['head_sha256'];head=dict(np.load(hp))
            assert sha(ev/row['file'])==row['file_sha256']
            pred=np.load(ev/row['file'])['prediction']
            np.testing.assert_allclose(pred,numpy_pose(te[stage],head),rtol=1e-10,atol=1e-9)
            risk=support_distance(te[stage],tr[stage],head['mean'],head['scale'])
            success=metrics(pred,te['target'])['joint_precision']
            assert len(risk)==512 and np.isfinite(risk).all()
            ix=np.argsort(risk,kind='stable');precision=float(success.mean())
            rows.append(dict(index=i,stage=stage,failure_auc=auc(success,risk),all_precision=precision,
                             lower_half_precision=float(success[ix[:256]].mean()),
                             lower_half_precision_minus_all=float(success[ix[:256]].mean()-precision),
                             success_count=int(success.sum()),risk_quantiles=np.quantile(risk,[0,.25,.5,.75,1]).tolist()))
            arrays[f'{i}_{stage}_risk']=risk;arrays[f'{i}_{stage}_success']=success
    # Selection operates separately within each model, then average fixed-model precision.
    pooled={stage:{k:float(np.mean([r[k] for r in rows if r['stage']==stage])) for k in ['all_precision','lower_half_precision','lower_half_precision_minus_all']} for stage in stages}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'scores.npz',**arrays)
    result=dict(status='COMPLETE_EXPLORATORY_SUPPORT_BASELINE',rows=rows,pooled=pooled,
                scores_sha256=sha(out/'scores.npz'),summary_sha256=sha(a.summary),source_sha256=sha(__file__),
                score='Minimum mean squared Euclidean distance to512 training feature vectors after frozen training-only normalization. No target state or predicted physical error enters risk score.',
                selection='Lowest256 of512 risk scores per model; stable input-order ties. Fixed50% coverage, no threshold/hyperparameter tuning. Relative batch ranking is not a calibrated online acceptance threshold.',
                scope='Consumed holdout, descriptive exploratory diagnostic only. No new-method novelty or confirmatory significance claim. Failure AUC undefined when a model has no successful images. Physical labels are used only to evaluate risk. This is localization risk, not dynamics or planning risk.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(pooled,indent=2))


if __name__=='__main__':main()
