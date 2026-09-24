"""C fixed-budget rolling training; development validation, no confirmation reader."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha, require_role, namespace_seed
from rolling_training import rolling_loss, rolling_predictions
from factorial_model import make_model, state_features
from adaptation_freeze import configure_dynamics_only, verify_frozen
from adaptation_gradient_control import unit_global_gradient
from nonlinear_pose_cost import NonlinearPoseCost
from evaluation_precision import configure_evaluation_precision
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ['base','heads','cache','A-runs','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--pool',type=int,choices=[0,1,2],required=True)
    p.add_argument('--objective',choices=['decoded_teacher','physical_labels'],required=True)
    p.add_argument('--condition',choices=['T0','T1','T2'],required=True)
    p.add_argument('--anchor-lambda',type=float,default=0.)
    p.add_argument('--profile-only',action='store_true');a=p.parse_args()
    if a.condition=='T2' and a.anchor_lambda not in [.1,1.]:raise ValueError('Prespecified lambda grid only')
    if a.condition!='T2' and a.anchor_lambda!=0:raise ValueError('Anchor belongs only to T2')
    g=2*a.pool;q=['decoded_teacher','physical_labels'].index(a.objective);base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs'
    ar=Path(a.A_runs)/f'job_{2*g+q}';hr=Path(a.heads)/f'group_{g}';cache=Path(a.cache)/f'group_{g}'
    init=json.loads((ar/'report.json').read_text());selection=json.loads((hr/'selection.json').read_text())
    assert (ar/'DONE').exists() and init['head_A_sha256']==selection['head_A']['sha256']
    hp=Path(selection['head_A']['path']);assert sha(hp)==init['head_A_sha256']
    plan=json.loads((r/'fiber_confirmation_plan.json').read_text());e=plan['models'][8*g];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1292001)
    model=make_model(base/'releases/visual-v1/official',base/'assets/pusht-v1/models/config.json',e['arm'],tr['seed'])
    initial=torch.load(ar/'last_weights.pt',map_location='cpu',weights_only=True);assert sha(ar/'last_weights.pt')==init['weights_sha256']
    model.load_state_dict(initial,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
    assert tensor_digest(boundary['frozen'])==init['frozen_after_sha256']
    head=NonlinearPoseCost.prepare(dict(np.load(hp)),'cuda');data={};bindings={}
    for split in ['train','validation']:
        meta=json.loads((cache/f'C_{split}.json').read_text());require_role(meta,{'continuation_'+split})
        fp=cache/f'C_{split}.npz';assert sha(fp)==meta['file_sha256'];bindings[split]=sha(cache/f'C_{split}.json')
        with np.load(fp) as z:data[split]={k:torch.as_tensor(z[k],device='cuda') for k in ['observed','normalized_actions','raw_states']}
    n=len(data['train']['observed']);assert n==768 and len(data['validation']['observed'])==192
    rng=np.random.default_rng(namespace_seed(20260919,f'C_schedule_pool_{a.pool}'))
    schedule=np.concatenate([rng.permutation(n) for _ in range(350)]).reshape(2100,128)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.save(out/'schedule.npy',schedule)
    params=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-3)
    steps=120 if a.profile_only else 2100;d=data['train'];losses=[];timing=[];start=time.monotonic()
    for i,ix in enumerate(schedule[:steps]):
        ix=torch.as_tensor(ix,device='cuda');optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize();t0=time.monotonic()
        total,parts=rolling_loss(model,d['observed'][ix],d['normalized_actions'][ix],d['raw_states'][ix],head,a.objective,a.condition,a.anchor_lambda)
        assert torch.isfinite(total)
        torch.cuda.synchronize();t1=time.monotonic();total.backward();torch.cuda.synchronize();t2=time.monotonic()
        control=unit_global_gradient(params);optim.step();verify_frozen(model,boundary);torch.cuda.synchronize();t3=time.monotonic()
        losses.append(dict(total=float(total.detach()),coordinate=float(parts['coordinate'].detach()),anchor=float(parts['anchor'].detach()),gradient=control))
        timing.append([t1-t0,t2-t1,t3-t2])
    final={k:v.detach().cpu() for k,v in model.state_dict().items()};verify_frozen(model,boundary)
    torch.save(final,out/'last_weights.pt');validation={}
    with torch.inference_mode():
        d=data['validation'];truth=state_features(d['raw_states'][:,3:].double())
        for mode in ['T0','T1']:
            predictions=torch.cat([rolling_predictions(model,d['observed'][i:i+128],d['normalized_actions'][i:i+128],mode) for i in range(0,len(truth),128)])
            pose=NonlinearPoseCost.forward(predictions,head)
            errors=(pose[...,2:4]-truth[...,2:4]).square().sum(-1)
            validation['observed_history' if mode=='T0' else 'free_running']=dict(endpoint_mse=float(errors[:,-1].mean()),per_step_mse=errors.mean(0).tolist())
    report=dict(status='PASS_C_PROFILE_ONLY' if a.profile_only else 'PASS_FIXED_C_CONTINUATION',pool=a.pool,group=g,
        objective=a.objective,condition=a.condition,anchor_lambda=a.anchor_lambda,updates=steps,
        initial_A_checkpoint_sha256=init['weights_sha256'],head_A_sha256=sha(hp),weights_sha256=sha(out/'last_weights.pt'),
        schedule_sha256=sha(out/'schedule.npy'),split_bindings=bindings,losses=losses,validation=validation,
        timing_seconds_per_step=np.mean(timing[20:],axis=0).tolist(),timing_columns=['forward','backward','gradient_normalize_optimizer_verify'],
        elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
        frozen_sha256=tensor_digest(boundary['frozen']),source_sha256=sha(__file__),
        rolling_helper_sha256=sha(ROOT/'strengthening/adapters/rolling_training.py'),
        selection_contract_sha256=sha(ROOT/'strengthening/configs/C_development_selection.json'),
        scope='Observed validation only; fixed final checkpoint; not a confirmation or intervention evaluation')
    atomic_json(out/'report.json',report);(out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
