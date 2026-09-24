"""A: fixed 72-fit maximum grid; validation-only head choice, all groups retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha, namespace_seed, require_role
from evaluation_precision import configure_evaluation_precision
from nonlinear_pose_cost import numpy_pose
from head_metrics import metrics, gate, nested_indices


def main():
    p=argparse.ArgumentParser()
    for key in ['base','cache','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--group',type=int);a=p.parse_args();g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';cache=Path(a.cache)/f'group_{g}'
    if not (cache/'DONE').exists():raise ValueError('Observed cache not accepted')
    spec=json.loads((ROOT/'strengthening/configs/development_contract.json').read_text());assert spec['phase']=='DEVELOPMENT_ONLY'
    data={}
    for split in ['train','validation']:
        meta=json.loads((cache/f'head_{split}.json').read_text());require_role(meta,{'head_'+split})
        assert meta['file_sha256']==sha(cache/f'head_{split}.npz')
        with np.load(cache/f'head_{split}.npz') as z:data[split]={k:z[k] for k in z.files}
    assert not(set(data['train']['pixel_sha256'])&set(data['validation']['pixel_sha256']))
    plan=json.loads((r/'fiber_confirmation_plan.json').read_text());hp=Path(plan['models'][8*g]['endpoint_head']['path'])
    assert sha(hp)==plan['models'][8*g]['endpoint_head']['sha256'];old=dict(np.load(hp))
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=False)
    precision=configure_evaluation_precision();torch.set_num_threads(2);start=time.monotonic()
    x=torch.as_tensor((data['train']['observed'].astype(float)-old['mean'])/old['scale'],device='cuda',dtype=torch.float32)
    y=torch.as_tensor((data['train']['labels']-old['target_mean'])/old['target_scale'],device='cuda',dtype=torch.float32)
    xv=torch.as_tensor((data['validation']['observed'].astype(float)-old['mean'])/old['scale'],device='cuda',dtype=torch.float64)
    vmasks={domain:data['validation']['domain']==domain for domain in ['expert','planner']}
    def evaluate(head):
        # Independent NumPy original head; new fitted weights evaluated in FP64 on GPU.
        state={k:torch.as_tensor(v,device='cuda',dtype=torch.float64) for k,v in head.items() if k[0].isdigit()}
        outputs=[]
        with torch.inference_mode():
            for j in range(0,len(xv),512):
                value=xv[j:j+512]
                for layer in [0,2,4]:
                    value=value@state[f'{layer}.weight'].T+state[f'{layer}.bias']
                    if layer!=4:value=value.relu()
                outputs.append(value.cpu().numpy()*old['target_scale']+old['target_mean'])
        prediction=np.concatenate(outputs)
        scores={dom:metrics(prediction[mask],data['validation']['labels'][mask],old['target_scale']) for dom,mask in vmasks.items()}
        return scores,.5*(scores['expert']['six_normalized_mse']+scores['planner']['six_normalized_mse']),prediction
    baseline,_,baseline_prediction=evaluate(old)
    oracle=numpy_pose(data['validation']['observed'][:32],old)
    for dom in vmasks:assert vmasks[dom].any()
    # Validate the numerical evaluation route against the original independent NumPy path.
    np.testing.assert_allclose(oracle,baseline_prediction[:32],rtol=1e-12,atol=1e-9)
    orders={domain:np.random.default_rng(namespace_seed(spec['root_seed'],f'head_frames_pool{g//2}_{domain}')).permutation(np.flatnonzero(data['train']['domain']==domain)) for domain in ['expert','planner']}
    sizes=[5000,20000,len(x)];fits=[]
    for size in sizes:
        indices=nested_indices(data['train']['domain'],size,orders)
        ix_by_domain={dom:indices[data['train']['domain'][indices]==dom] for dom in orders}
        for width in [256,512]:
            for seed_role in ['A','B']:
                name=f'n{len(indices)}_w{width}_seed{seed_role}';folder=out/name;folder.mkdir()
                seed=namespace_seed(spec['root_seed'],f'head_fit_g{g}_w{width}_n{len(indices)}_{seed_role}')
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=nn.Sequential(nn.Linear(192,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,6)).cuda()
                optim=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0.)
                best={};history=[];fit_start=time.monotonic()
                np.save(folder/'training_frame_indices.npy',indices)
                for step in range(1,3001):
                    ix=np.r_[rng.choice(ix_by_domain['expert'],128),rng.choice(ix_by_domain['planner'],128)]
                    optim.zero_grad(set_to_none=True);loss=(model(x[ix])-y[ix]).square().mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite head training loss')
                    loss.backward();optim.step()
                    if step%250==0:
                        weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
                        h={**weights,**{k:old[k] for k in ['mean','scale','target_mean','target_scale']}}
                        score,j,_=evaluate(h);qualification=gate(score,baseline)
                        record=dict(step=step,J_validation=j,quality_gate=qualification,metrics=score,loss=float(loss.detach()))
                        history.append(record)
                        for key,eligible in [('best_validation',True),('best_qualified',qualification['passed'])]:
                            if eligible and (key not in best or j<best[key]['J_validation']):
                                np.savez_compressed(folder/(key+'.npz'),**h)
                                best[key]={**record,'path':str(folder/(key+'.npz')),'sha256':sha(folder/(key+'.npz'))}
                entry=dict(name=name,size=len(indices),width=width,seed_role=seed_role,seed=seed,updates=3000,
                    indices_sha256=sha(folder/'training_frame_indices.npy'),domain_unique_counts={k:len(v) for k,v in ix_by_domain.items()},
                    best=best,history=history,seconds=time.monotonic()-fit_start)
                atomic_json(folder/'report.json',entry);fits.append(entry)
                print('HEAD_FIT',g,name,entry['seconds'],flush=True)
                del model,optim;torch.cuda.empty_cache()
    eligible=[(fit,fit['best']['best_qualified']) for fit in fits if fit['seed_role']=='A' and 'best_qualified' in fit['best']]
    qualified=bool(eligible)
    if not eligible:eligible=[(fit,fit['best']['best_validation']) for fit in fits if fit['seed_role']=='A']
    chosen,headA=min(eligible,key=lambda pair:(pair[1]['J_validation'],pair[0]['width'],pair[0]['size'],pair[1]['step']))
    fitB=next(f for f in fits if f['seed_role']=='B' and f['width']==chosen['width'] and f['size']==chosen['size']);headB=fitB['best']['best_validation']
    atomic_json(out/'selection.json',dict(group=g,head_A=headA,head_B=headB,quality_gate_pass=qualified,
        interpretation='qualified_readout' if qualified else 'refitted_head_sensitivity',
        selected_width=chosen['width'],selected_unique_frames=chosen['size'],head_B_quality_gate_pass=headB['quality_gate']['passed'],
        selection_rule='Validation only; gate-first A then J/width/size/earlier checkpoint; B uses same size/width and separate seed',
        old_head_sha256=sha(hp),head_cache_report_sha256=sha(cache/'report.json'),
        development_contract_sha256=sha(ROOT/'strengthening/configs/development_contract.json')))
    atomic_json(out/'report.json',dict(status='PASS_COMPLETE_HEAD_GRID',group=g,fit_count=len(fits),fits=fits,
        baseline=baseline,precision=precision,elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),
        source_sha256=sha(__file__),peak_allocated_bytes=torch.cuda.max_memory_allocated(),selection_sha256=sha(out/'selection.json')))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
