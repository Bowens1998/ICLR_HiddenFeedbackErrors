"""A coordinate continuation: legacy data, exact update schedule and optimizer; new g_A."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only, verify_frozen, TRAINABLE_ROOTS
from adaptation_gradient_control import unit_global_gradient
from adaptation_task_losses import target_for, coordinates
from train_task_coordinate_formal import temporal
from nonlinear_pose_cost import NonlinearPoseCost
from evaluation_precision import configure_evaluation_precision
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ['base','heads','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int);a=p.parse_args();idx=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index
    g,q=divmod(idx,2);objective=['decoded_teacher','physical_labels'][q]
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';cache=r/f'formal_cache/job_{2*g}';freeze=r/'formal_stream_freeze'
    cr=json.loads((cache/'report.json').read_text());ca=json.loads((cache/'acceptance.json').read_text())
    assert ca['status']=='PASS_FULL_FRESH_PROCESS_REENCODING' and ca['cache_report_sha256']==sha(cache/'report.json')
    e=cr['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text());config=base/'assets/pusht-v1/models/config.json'
    assert sha(td/'last_weights.pt')==e['weights_sha256'] and sha(config)==tr['config_sha256']
    hs=Path(a.heads)/f'group_{g}';selection=json.loads((hs/'selection.json').read_text())
    if not (hs/'DONE').exists():raise ValueError('Head grid not complete')
    acceptance=json.loads((hs/'acceptance.json').read_text())
    assert acceptance['status']=='PASS_INDEPENDENT_SELECTED_HEAD_RECONSTRUCTION' and acceptance['selection_sha256']==sha(hs/'selection.json')
    hp=Path(selection['head_A']['path']);assert sha(hp)==selection['head_A']['sha256']
    fr=json.loads((freeze/'report.json').read_text());assert sha(freeze/'update_indices.npz')==fr['schedule_sha256']
    schedule=np.load(freeze/'update_indices.npz')['indices'];assert schedule.shape==(2100,128)
    for epoch in schedule.reshape(210,1280):np.testing.assert_array_equal(np.sort(epoch),np.arange(1280))
    precision=configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1292001)
    original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True)
    model=make_model(base/'releases/visual-v1/official',config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);model=model.cuda()
    boundary=configure_dynamics_only(model);assert tensor_digest(model.state_dict())==cr['initial_tensor_sha256']
    assert tensor_digest(boundary['frozen'])==cr['frozen_tensor_sha256']
    head=NonlinearPoseCost.prepare(dict(np.load(hp)),'cuda');data={}
    for row in cr['rows']:
        fp=cache/row['file'];assert sha(fp)==row['sha256']
        with np.load(fp) as z:data[row['stream']]={k:torch.as_tensor(z[k],device='cuda') for k in ['observed','normalized_actions','raw_states']}
        d=data[row['stream']];d['target']=target_for(objective,d['observed'],d['raw_states'],head)
    out=Path(a.output)/f'job_{idx}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    params=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-3)
    train=data['planner_train'];losses=[];gradient=[]
    for indices in schedule:
        ix=torch.as_tensor(indices.astype(np.int64),device='cuda');optim.zero_grad(set_to_none=True)
        pred=temporal(model,e['arm'],train['observed'][ix],train['normalized_actions'][ix])
        loss=(coordinates(pred,head)-train['target'][ix]).square().mean();assert torch.isfinite(loss)
        loss.backward();control=unit_global_gradient(params);optim.step();verify_frozen(model,boundary)
        losses.append(float(loss.detach()));gradient.append(control)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    changed=[k for k,v in final.items() if not torch.equal(v,original[k])]
    assert set(changed)<=set(boundary['trainable_names'])
    assert all(any(k.startswith(root+'.') for k in changed) for root in TRAINABLE_ROOTS)
    torch.save(final,out/'last_weights.pt');metrics={}
    with torch.inference_mode():
        for name,d in data.items():
            pred=torch.cat([temporal(model,e['arm'],d['observed'][i:i+128],d['normalized_actions'][i:i+128]) for i in range(0,len(d['observed']),128)])
            metrics[name]=float((coordinates(pred,head)-d['target']).square().mean())
    assert sha(hp)==selection['head_A']['sha256']
    atomic_json(out/'report.json',dict(status='PASS_FIXED_A_CONTINUATION',group=g,index=idx,objective=objective,
        original_checkpoint_sha256=e['weights_sha256'],weights_sha256=sha(out/'last_weights.pt'),
        head_A_sha256=sha(hp),head_selection_sha256=sha(hs/'selection.json'),quality_gate_pass=selection['quality_gate_pass'],
        schedule_sha256=sha(freeze/'update_indices.npz'),cache_report_sha256=sha(cache/'report.json'),
        frozen_before_sha256=cr['frozen_tensor_sha256'],frozen_after_sha256=tensor_digest({k:final[k] for k in boundary['frozen']}),
        trainable_names=boundary['trainable_names'],changed_names=changed,updates=2100,batch=128,
        optimizer={'name':'AdamW','lr':1e-5,'weight_decay':1e-3},losses=losses,gradient_control=gradient,
        fitted_objective_metrics=metrics,precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
        source_sha256=sha(__file__),helper_sha256={n:sha(ROOT/'scripts/visual'/n) for n in
          ['train_task_coordinate_formal.py','adaptation_task_losses.py','adaptation_gradient_control.py','adaptation_freeze.py']},
        scope='All groups retained, fixed final checkpoint. No confirmation evaluation.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
