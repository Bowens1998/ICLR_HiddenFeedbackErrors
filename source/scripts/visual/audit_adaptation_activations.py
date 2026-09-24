"""Frozen final-head activation audit on training data only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from analyze_coverage_goals import sha
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser()
    for k in ['adapted','frozen','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[]
    for i in range(6):
        for stage in ['trained_cls','initial_cls']:
            run=Path(a.adapted)/f'job_{i}'/stage;r=json.loads((run/'report.json').read_text());ac=json.loads((run/'acceptance.json').read_text())
            assert ac['status']=='PASS' and ac['report_sha256']==sha(run/'report.json')
            assert sha(run/'weights.pt')==r['weights_sha256'] and sha(run/'accepted_train_predictions.npz')==ac['train_predictions_sha256']
            checkpoint=torch.load(run/'weights.pt',map_location='cpu',weights_only=False)
            h={k:v.numpy() for k,v in checkpoint['head'].items()};h.update(checkpoint['normalization'])
            z=dict(np.load(run/'accepted_train_predictions.npz'));x=(z['tokens'].astype(float)-h['mean'])/h['scale']
            layers=[]
            for k in (0,2):
                pre=x@h[f'{k}.weight'].T+h[f'{k}.bias'];x=np.maximum(pre,0)
                layers.append(dict(layer=k,active_fraction=float((pre>0).mean()),always_inactive_units=int((pre<=0).all(0).sum()),always_active_units=int((pre>0).all(0).sum()),units=pre.shape[1],mean_activation_variance=float(x.var(0).mean()),zero_activation_rows=int((x==0).all(1).sum())))
            pred=(x@h['4.weight'].T+h['4.bias'])*h['target_scale']+h['target_mean']
            np.testing.assert_allclose(pred,z['prediction'],rtol=1e-10,atol=1e-9)
            np.testing.assert_allclose(pred,numpy_pose(z['tokens'],h),rtol=1e-10,atol=1e-9)
            # Compare exact output spread with labels; no posthoc correction or refit.
            frozen=Path(a.frozen)/f'job_{i}';fr=json.loads((frozen/'report.json').read_text())
            assert sha(frozen/'report.json')==r['frozen_fit_report_sha256']
            old=next(v for v in fr['rows'] if v['stage']==stage)
            assert sha(frozen/stage/'train_prediction.npz')==old['files_sha256']['train_prediction.npz']
            oldpred=np.load(frozen/stage/'train_prediction.npz')['prediction']
            rows.append(dict(index=i,stage=stage,report_sha256=sha(run/'report.json'),acceptance_sha256=sha(run/'acceptance.json'),layers=layers,
                             final_prediction_std=pred.std(0).tolist(),frozen_prediction_std=oldpred.std(0).tolist(),target_std=z['target'].std(0).tolist(),
                             standardized_prediction_variance=float((pred/h['target_scale']).var(0).mean())))
    result=dict(status='COMPLETE12_FROZEN_ACTIVATION_AUDITS',rows=rows,source_sha256=sha(__file__),scope='Final saved parameters and accepted training features only. Exact inactive-unit counts apply to512 observed training rows, not all possible inputs. No perturbation, refit, temporal or causal inference. NumPy layer reconstruction agrees with accepted Torch predictions.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('PASS_ALL12_ACTIVATION_AUDITS')


if __name__=='__main__':main()
