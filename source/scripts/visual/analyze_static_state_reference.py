"""Descriptive complete matrix; no significance or matched-label claim."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha,metrics


def main():
    p=argparse.ArgumentParser()
    for k in ['runs','plan','reference-evaluations','reference-summary','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());rs=json.loads(Path(a.reference_summary).read_text());assert rs['status']=='PASS_COMPLETE18_CONDITIONS'
    rows=[]
    for i in range(6):
        run=Path(a.runs)/f'job_{i}';r=json.loads((run/'report.json').read_text())
        entry=plan['models'][plan['routes'][i*8+6]['model_index']]
        assert r['status']=='COMPLETE_STATIC_STATE_REFERENCE' and r['index']==i
        assert r['source_sha256']==sha(Path(__file__).with_name('evaluate_static_state_reference.py'))
        assert r['plan_sha256']==sha(a.plan) and r['weights_sha256']==entry['weights_sha256']
        assert r['file_sha256']==sha(run/'predictions.npz')
        ref=Path(a.reference_evaluations)/f'job_{i}';rr=json.loads((ref/'report.json').read_text())
        assert sha(ref/'report.json')==rs['provenance'][i]['report_sha256']
        assert r['bank_report_sha256']==rr['bank_report_sha256'] and r['bank_acceptance_sha256']==rr['bank_acceptance_sha256']
        assert sha(ref/'features.npz')==rr['features_sha256']
        z=dict(np.load(run/'predictions.npz'));rz=np.load(ref/'features.npz')
        for k in ['target','source_index']:np.testing.assert_array_equal(z[k],rz[k])
        assert z['prediction'].shape==(512,6) and np.isfinite(z['prediction']).all()
        norm=entry['target_normalization']
        np.testing.assert_array_equal(z['prediction'],z['encoded'].astype(float)*np.array(norm['std'])+np.array(norm['mean']))
        m={k:float(v.mean()) for k,v in metrics(z['prediction'],z['target']).items()}
        rows.append(dict(index=i,arm=entry['arm'],metrics=m,report_sha256=sha(run/'report.json')))
    pooled={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
    result=dict(status='COMPLETE6_STATE_REFERENCES',images=512,rows=rows,pooled=pooled,source_sha256=sha(__file__),reference_summary_sha256=sha(a.reference_summary),scope='Consumed common static bank, descriptive absolute metrics. Bound native outputs and independently checked denormalization; no independent image re-encoding. Supervised pretraining uses more state labels than512-label probes; no matched-budget or causal claim.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pooled))


if __name__=='__main__':main()
