"""One predeclared visual-only native readout; no dynamics update or test selection."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import sha, atomic_json, require_role, namespace_seed
from head_metrics import metrics
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cache=Path(a.cache);assert (cache/'DONE').exists();cr=json.loads((cache/'report.json').read_text())
    specpath=ROOT/'strengthening/configs/B_development.json';spec=json.loads(specpath.read_text());assert cr['development_spec_sha256']==sha(specpath)
    h=dict(np.load(cache/'normalizers.npz'));assert sha(cache/'normalizers.npz')==cr['normalizers_sha256']
    configure_evaluation_precision();torch.set_num_threads(2);arrays={};bindings={};start=time.monotonic()
    for split in ['train','validation']:
        path=cache/f'head_{split}.json';meta=json.loads(path.read_text());require_role(meta,{'head_'+split});bindings[split]=sha(path)
        fp=Path(meta['file']);assert sha(fp)==meta['file_sha256'];assert fp.stat().st_size==int(np.prod(meta['shape']))*4
        ypath=cache/f'head_{split}_labels.npy';assert sha(ypath)==meta['labels_sha256'];truth=np.load(ypath)
        mm=np.memmap(fp,mode='r',dtype=np.float32,shape=tuple(meta['shape']));dtype=torch.float32 if split=='train' else torch.float64
        x=torch.empty(meta['shape'],device='cuda',dtype=dtype)
        for i in range(0,len(mm),128):x[i:i+128]=torch.as_tensor((mm[i:i+128].astype(float)-h['mean'])/h['scale'],device='cuda',dtype=dtype)
        y=torch.as_tensor((truth-h['target_mean'])/h['target_scale'],device='cuda',dtype=torch.float32)
        arrays[split]=dict(x=x,y=y,truth=truth,raw=mm)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed=namespace_seed(spec['root_seed'],spec['head']['seed_namespace'])
    torch.manual_seed(seed);rng=np.random.default_rng(seed);w=spec['head']['width']
    model=nn.Sequential(nn.Linear(75264,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,6)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=spec['head']['lr'],weight_decay=0.)
    best=None;history=[];d=arrays['train']
    for step in range(1,spec['head']['updates']+1):
        ix=rng.integers(0,len(d['x']),size=spec['head']['batch']);optimizer.zero_grad(set_to_none=True)
        loss=(model(d['x'][ix])-d['y'][ix]).square().mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
        if step%250==0:
            weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()};candidate={**h,**weights}
            layers={k:torch.as_tensor(v,device='cuda',dtype=torch.float64) for k,v in weights.items()};predictions=[]
            with torch.inference_mode():
                for i in range(0,len(arrays['validation']['x']),32):
                    val=arrays['validation']['x'][i:i+32]
                    for layer in [0,2,4]:
                        val=val@layers[f'{layer}.weight'].T+layers[f'{layer}.bias']
                        if layer<4:val=val.relu()
                    predictions.append(val.cpu().numpy()*h['target_scale']+h['target_mean'])
            pred=np.concatenate(predictions);scores=metrics(pred,arrays['validation']['truth'],h['target_scale'])
            np.testing.assert_allclose(pred[:16],numpy_pose(arrays['validation']['raw'][:16],candidate),rtol=1e-10,atol=1e-7)
            row=dict(step=step,validation=scores,loss=float(loss.detach()));history.append(row)
            if best is None or scores['six_normalized_mse']<best['validation']['six_normalized_mse']:
                np.savez_compressed(out/'weights.npz',**candidate);np.savez_compressed(out/'validation_predictions.npz',predicted=pred,truth=arrays['validation']['truth'])
                best=row
            print('NATIVE_HEAD',step,scores['six_normalized_mse'],flush=True)
    atomic_json(out/'report.json',dict(status='PASS_NATIVE_HEAD_FIXED_GRID_REQUIRES_FULL_CPU_ACCEPTANCE',
        best=best,history=history,seed=seed,updates=spec['head']['updates'],head_config=spec['head'],
        weights_sha256=sha(out/'weights.npz'),validation_predictions_sha256=sha(out/'validation_predictions.npz'),
        cache_report_sha256=sha(cache/'report.json'),normalizers_sha256=sha(cache/'normalizers.npz'),split_bindings=bindings,
        elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
        source_sha256=sha(__file__),development_spec_sha256=sha(specpath),
        scope='Visual patches only; frozen native dynamics; validation-only head selection; no confirmation or intervention effects.'))
    (out/'DONE').write_text('fit_complete_requires_cpu_acceptance\n')


if __name__=='__main__':main()
