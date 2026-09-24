"""Small validation-only numerical inputs; independently recheck new A fit metrics."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import sha, atomic_json
from factorial_model import make_model, state_features
from adaptation_freeze import configure_dynamics_only, verify_frozen
from train_task_coordinate_formal import temporal
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ['base','heads','A-runs','latent-reuse','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int);a=p.parse_args();g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';cache=r/f'formal_cache/job_{2*g}'
    cr=json.loads((cache/'report.json').read_text());e=cr['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    reuse=json.loads(Path(a.latent_reuse).read_text());assert reuse['status']=='PASS_SIX_LEGACY_LATENT_REUSE_BINDINGS'
    lr=next(x for x in reuse['rows'] if x['group']==g)
    hs=Path(a.heads)/f'group_{g}';s=json.loads((hs/'selection.json').read_text());ac=json.loads((hs/'acceptance.json').read_text())
    assert ac['selection_sha256']==sha(hs/'selection.json')
    heads={}
    for name in ['head_A','head_B']:
        assert sha(s[name]['path'])==s[name]['sha256'];heads[name]=dict(np.load(s[name]['path']))
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=False)
    for name,h in heads.items():np.savez_compressed(out/(name+'.npz'),**h)
    data={}
    for row in cr['rows']:
        path=cache/row['file'];assert sha(path)==row['sha256'];data[row['stream']]=dict(np.load(path))
    original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True)
    configure_evaluation_precision();torch.set_num_threads(2);predicted=[];rows=[]
    for q,obj in enumerate(['latent','decoded_teacher','physical_labels']):
        if q==0:
            cp=Path(lr['checkpoint']);expected=lr['checkpoint_sha256'];fit=None
        else:
            ar=Path(a.A_runs)/f'job_{2*g+q-1}';assert (ar/'DONE').exists();fit=json.loads((ar/'report.json').read_text())
            assert fit['head_A_sha256']==s['head_A']['sha256'];cp=ar/'last_weights.pt';expected=fit['weights_sha256']
        assert sha(cp)==expected;state=torch.load(cp,map_location='cpu',weights_only=True)
        model=make_model(base/'releases/visual-v1/official',base/'assets/pusht-v1/models/config.json',e['arm'],tr['seed'])
        model.load_state_dict(state,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        for key,value in boundary['frozen'].items():assert torch.equal(value.cpu(),original[key])
        assert tensor_digest(boundary['frozen'])==cr['frozen_tensor_sha256']
        metrics={}
        with torch.inference_mode():
            for stream,d in data.items():
                outputs=[]
                for i in range(0,len(d['observed']),128):
                    outputs.append(temporal(model,e['arm'],torch.as_tensor(d['observed'][i:i+128],device='cuda'),
                                    torch.as_tensor(d['normalized_actions'][i:i+128],device='cuda')).cpu().numpy())
                values=np.concatenate(outputs)
                if stream=='planner_validation':predicted.append(values[:8,-1])
                if q:
                    h=heads['head_A'];decoded=(numpy_pose(values,h)-h['target_mean'])/h['target_scale']
                    target=(numpy_pose(d['observed'][:,1:],h) if obj=='decoded_teacher' else
                            state_features(torch.as_tensor(d['raw_states'][:,1:],dtype=torch.float64)).numpy())
                    target=(target-h['target_mean'])/h['target_scale']
                    metrics[stream]=float(np.mean((decoded-target)**2))
                    np.testing.assert_allclose(metrics[stream],fit['fitted_objective_metrics'][stream],rtol=1e-9,atol=1e-8)
        verify_frozen(model,boundary)
        rows.append(dict(objective=obj,checkpoint=str(cp),checkpoint_sha256=sha(cp),
            independent_numpy_fit_metrics=metrics,frozen_tensors_equal_original=True))
        del model,boundary;torch.cuda.empty_cache()
    np.savez_compressed(out/'inputs.npz',predicted=np.stack(predicted),
        observed=data['planner_validation']['observed'][:8,-1],
        donor=data['expert_validation']['observed'][:8,-1])
    atomic_json(out/'report.json',dict(status='PASS_A_NUMPY_METRICS_AND_DEVELOPMENT_INPUTS',group=g,
        role='projection_numerical_development',rows=rows,source_sha256=sha(__file__),
        inputs_sha256=sha(out/'inputs.npz'),head_sha256={name:sha(out/(name+'.npz')) for name in heads},
        head_selection_sha256=sha(hs/'selection.json'),cache_report_sha256=sha(cache/'report.json'),
        source_policy='First eight planner-validation windows and first eight expert-validation windows, fixed before projection. Numerical QC only; not a donor efficacy comparison.',
        scope='Full four-stream A objective metrics independently rebuilt with NumPy; no corrected rollout or confirmation outcome.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
