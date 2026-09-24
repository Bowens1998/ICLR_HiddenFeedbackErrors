"""Training-only fixed-normalizer drift and output-variance diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['adapted','features','frozen','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[]
    for i in range(6):
        features=Path(a.features)/f'job_{i}';f=dict(np.load(features/'features.npz'))
        for stage in ['trained_cls','initial_cls']:
            run=Path(a.adapted)/f'job_{i}'/stage;r=json.loads((run/'report.json').read_text());ac=json.loads((run/'acceptance.json').read_text())
            assert ac['status']=='PASS' and ac['report_sha256']==sha(run/'report.json')
            assert ac['train_predictions_sha256']==sha(run/'accepted_train_predictions.npz')
            assert ac['verifier_sha256']==sha(Path(__file__).with_name('accept_readout_adaptation.py'))
            assert r['feature_report_sha256']==sha(features/'report.json')
            fr=json.loads((features/'report.json').read_text());assert fr['features_sha256']==sha(features/'features.npz')
            frozen=Path(a.frozen)/f'job_{i}';assert r['frozen_fit_report_sha256']==sha(frozen/'report.json')
            fit=json.loads((frozen/'report.json').read_text());row=next(v for v in fit['rows'] if v['stage']==stage)
            hp=frozen/stage/'weights.npz';assert sha(hp)==row['files_sha256']['weights.npz'];h=dict(np.load(hp))
            z=dict(np.load(run/'accepted_train_predictions.npz'))
            np.testing.assert_array_equal(z['target'],f['target']);np.testing.assert_array_equal(z['source_index'],f['source_index'])
            x0=f[stage].astype(float);x1=z['tokens'].astype(float);assert x0.shape==x1.shape==(512,192)
            n0=(x0-h['mean'])/h['scale'];n1=(x1-h['mean'])/h['scale']
            target=z['target'];pred=z['prediction'];scale=h['target_scale']
            model_mse=float(np.mean(((pred-target)/scale)**2))
            constant_mse=float(np.mean(((target-target.mean(0))/scale)**2))
            delta=(x1-x0)/h['scale']
            rows.append(dict(index=i,stage=stage,report_sha256=sha(run/'report.json'),acceptance_sha256=sha(run/'acceptance.json'),
                initial_feature_std_quantiles=np.quantile(x0.std(0),[0,.5,1]).tolist(),
                final_feature_std_quantiles=np.quantile(x1.std(0),[0,.5,1]).tolist(),
                initial_normalized_variance=float(n0.var(0).mean()),final_normalized_variance=float(n1.var(0).mean()),
                normalized_mean_shift_rms=float(np.sqrt(np.mean((n1.mean(0)-n0.mean(0))**2))),
                normalized_feature_change_rms=float(np.sqrt(np.mean(delta**2))),
                prediction_std=pred.std(0).tolist(),target_std=target.std(0).tolist(),
                standardized_train_mse=model_mse,training_mean_predictor_mse=constant_mse,
                mse_ratio_to_training_mean=model_mse/constant_mse,
                training_joint_precision=ac['metrics']['joint_precision']))
    result=dict(status='COMPLETE12_TRAINING_DIAGNOSTICS',rows=rows,source_sha256=sha(__file__),scope='Training-only posthoc description of initial/final features, fixed normalizer, and accepted final predictions. No intermediate trajectory, causal attribution, alternative normalizer intervention, refit, or holdout use. Constant baseline is the training target mean, not a learned test result.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['index'],r['stage'],'variance',r['final_normalized_variance'],'mean_shift',r['normalized_mean_shift_rms'],'mse/constant',r['mse_ratio_to_training_mean'])


if __name__=='__main__':main()
