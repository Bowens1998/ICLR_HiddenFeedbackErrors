"""Complete function-matched parameterization comparison; training only."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha,metrics


def main():
    p=argparse.ArgumentParser()
    for k in ['relu','absorbed','relu-config','absorbed-config','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();rc=json.loads(Path(a.relu_config).read_text());lc=json.loads(Path(a.absorbed_config).read_text())
    assert lc['parameterization']=='normalization_absorbed_first_layer'
    for key in ['labels','updates','batch_size','microbatch_size','encoder_learning_rate','head_learning_rate','weight_decay','engineering_updates']:assert rc[key]==lc[key]
    rows=[]
    for i in range(6):
        for stage in ['trained_cls','initial_cls']:
            results={};reports={};data={};provenance={}
            for mode,root,config,verifier,fit_source in [('relu',a.relu,a.relu_config,'accept_readout_adaptation.py','fit_readout_adaptation.py'),('absorbed',a.absorbed,a.absorbed_config,'accept_absorbed_adaptation.py','fit_absorbed_adaptation.py')]:
                folder=Path(root)/f'job_{i}'/stage;r=json.loads((folder/'report.json').read_text());ac=json.loads((folder/'acceptance.json').read_text())
                assert r['status']=='FIT_REQUIRES_ACCEPTANCE' and r['updates']==2000 and r['index']==ac['index']==i and r['stage']==ac['stage']==stage
                assert r['config_sha256']==sha(config) and r['source_sha256']==sha(Path(__file__).with_name(fit_source))
                assert ac['status']=='PASS' and ac['report_sha256']==sha(folder/'report.json') and ac['verifier_sha256']==sha(Path(__file__).with_name(verifier))
                assert r['weights_sha256']==sha(folder/'weights.pt')
                assert ac['train_predictions_sha256']==sha(folder/'accepted_train_predictions.npz')
                z=dict(np.load(folder/'accepted_train_predictions.npz'));assert z['prediction'].shape==z['target'].shape==(512,6)
                m={k:float(v.mean()) for k,v in metrics(z['prediction'],z['target']).items()}
                for k in m:np.testing.assert_allclose(m[k],ac['metrics'][k],rtol=1e-12,atol=1e-12)
                results[mode]=m;reports[mode]=r;data[mode]=z
                provenance[mode]=dict(report_sha256=sha(folder/'report.json'),acceptance_sha256=sha(folder/'acceptance.json'))
                if mode=='absorbed':
                    assert r['parameterization']=='normalization_absorbed_first_layer'
                    initial_gap=r['initial_standardized_output_max_difference']
            for k in ['feature_report_sha256','frozen_fit_report_sha256']:assert reports['relu'][k]==reports['absorbed'][k]
            for k in ['target','source_index']:np.testing.assert_array_equal(data['relu'][k],data['absorbed'][k])
            rows.append(dict(index=i,stage=stage,metrics=results,absorbed_minus_relu={k:results['absorbed'][k]-results['relu'][k] for k in results['relu']},initial_standardized_output_max_difference=initial_gap,provenance=provenance))
    pooled={stage:{mode:{k:float(np.mean([r['metrics'][mode][k] for r in rows if r['stage']==stage])) for k in rows[0]['metrics'][mode]} for mode in ['relu','absorbed']} for stage in ['trained_cls','initial_cls']}
    result=dict(status='COMPLETE12_MATCHED_TRAINING_COMPARISONS',rows=rows,pooled=pooled,source_sha256=sha(__file__),relu_config_sha256=sha(a.relu_config),absorbed_config_sha256=sha(a.absorbed_config),scope='Fixed initial-function-matched parameterization intervention, all12 pairs; descriptive resubstitution metrics only. Training observations are not independent test samples. No confidence intervals, heldout performance, parameterization selection, method novelty or planning claim. Both conditions use ReLU; Adam updates differ despite near-identical initial functions.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('COMPLETE12_MATCHED_TRAINING_COMPARISONS')


if __name__=='__main__':main()
