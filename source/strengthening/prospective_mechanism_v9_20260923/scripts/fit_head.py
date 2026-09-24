"""Fixed C/D fit from disjoint observed fit/validation views; no qualification access."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from s1_common import (atomic_json, atomic_npz, checked_json, load_protocol, load_view,
    mixed_moments, namespace_seed, sha, source_hashes)
from s1_readout import build_model, numpy_forward, serialize_model, _V8_PATH


def main():
    p = argparse.ArgumentParser()
    for k in ['protocol','protocol-sha256','views','views-report-sha256','output']: p.add_argument('--'+k, required=True)
    p.add_argument('--group', type=int, choices=[0,1], required=True)
    p.add_argument('--head-role', choices=['C','D'], required=True)
    a = p.parse_args(); cfg = load_protocol(a.protocol, a.protocol_sha256)
    report = checked_json(Path(a.views) / 'report.json', a.views_report_sha256)
    if report.get('status') != 'PASS_S1_DISJOINT_PARENT_VIEWS' or report.get('protocol_sha256') != a.protocol_sha256:
        raise ValueError('Unaccepted or cross-protocol observed views')
    data = {r: {d: load_view(a.views, report, head_role=a.head_role, old_role=r, group=a.group,
            domain=d, protocol_sha=a.protocol_sha256, allowed_roles={'fit','validation'})
        for d in ['expert','planner']} for r in ['fit','validation']}
    # Complete input validation precedes importing Torch or constructing a head.
    import torch
    if not torch.cuda.is_available(): raise ValueError('Fitting requires an allocated GPU')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    settings = cfg['fit']; start = time.monotonic()
    ns = settings['seed_namespace'].format(role=a.head_role, group=a.group)
    seed = namespace_seed(cfg['root_seed'], ns)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    mean, scale = mixed_moments(data['fit']['expert']['observed'], data['fit']['planner']['observed'], settings['normalizer_std_floor'])
    tm, ts = mixed_moments(data['fit']['expert']['labels'], data['fit']['planner']['labels'], settings['normalizer_std_floor'])
    norm = dict(mean=mean, scale=scale, target_mean=tm, target_scale=ts)
    tensors = {}
    for role, domains in data.items():
        dtype = torch.float32 if role == 'fit' else torch.float64
        tensors[role] = {d: (torch.as_tensor((x['observed'].astype(np.float64)-mean)/scale, dtype=dtype, device='cuda'),
                             torch.as_tensor((x['labels'].astype(np.float64)-tm)/ts, dtype=dtype, device='cuda'))
                         for d,x in domains.items()}
    model = build_model(a.head_role).cuda().train()
    optimizer_spec = dict(lr=settings['learning_rate'], betas=(.9,.999), eps=1e-8,
        weight_decay=0., amsgrad=False, maximize=False, foreach=False, fused=False)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_spec)
    history=[]; best=None
    for step in range(1, settings['updates']+1):
        xs=[]; ys=[]
        for domain in ['expert','planner']:
            x,y=tensors['fit'][domain]; ix=rng.integers(0,len(x),size=128)
            xs.append(x[ix]); ys.append(y[ix])
        optimizer.zero_grad(set_to_none=True)
        loss=(model(torch.cat(xs))-torch.cat(ys)).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite fitting loss')
        loss.backward(); optimizer.step()
        if step % settings['validation_interval'] == 0:
            evaluation=copy.deepcopy(model).double().eval(); metrics={}
            with torch.inference_mode():
                for domain in ['expert','planner']:
                    x,y=tensors['validation'][domain]; total=0.
                    for begin in range(0,len(x),1024):
                        total+=float((evaluation(x[begin:begin+1024])-y[begin:begin+1024]).square().sum())
                    metrics[domain]=total/(len(x)*6)
            score=.5*(metrics['expert']+metrics['planner'])
            row=dict(step=step, balanced_six_normalized_validation_mse=score,
                     domain_validation=metrics, train_loss=float(loss.detach()))
            if not np.isfinite(score): raise ValueError('Nonfinite validation score')
            history.append(row)
            if best is None or score < best['balanced_six_normalized_validation_mse']:
                atomic_npz(out/'selected.npz', **serialize_model(model,norm))
                best={**row,'file':'selected.npz','sha256':sha(out/'selected.npz')}
            del evaluation
            print(json.dumps(dict(group=a.group,head_role=a.head_role,**row)),flush=True)
    if best is None: raise ValueError('No selected validation checkpoint')
    with np.load(out/'selected.npz',allow_pickle=False) as z: head=dict(z)
    verification_model=build_model(a.head_role).double().cuda().eval()
    verification_model.load_state_dict({k:torch.as_tensor(v,dtype=torch.float64,device='cuda')
        for k,v in head.items() if k.endswith(('.weight','.bias'))},strict=True)
    fixture=np.r_[data['validation']['expert']['observed'][:32],data['validation']['planner']['observed'][:32]]
    with torch.inference_mode():
        expected=verification_model(torch.as_tensor((fixture.astype(np.float64)-mean)/scale,dtype=torch.float64,device='cuda')).cpu().numpy()*ts+tm
    independent=numpy_forward(fixture,head,a.head_role)
    np.testing.assert_allclose(independent,expected,rtol=1e-12,atol=1e-9)
    atomic_npz(out/'forward_verification.npz',tokens=fixture,torch_prediction=expected,numpy_prediction=independent)
    atomic_json(out/'report.json',dict(status='PASS_S1_FIXED_HEAD_FIT',group=a.group,head_role=a.head_role,
        protocol_sha256=a.protocol_sha256,views=str(Path(a.views).resolve()),views_report_sha256=a.views_report_sha256,
        seed=seed,seed_namespace=ns,selected=best,history=history,updates=settings['updates'],
        optimizer={'name':'Adam',**optimizer_spec},architecture=cfg['head_design'][a.head_role],
        normalizers='Own fit-only 50/50 expert/planner population moments, FP64, floor1e-6',
        forward_fixture_sha256=sha(out/'forward_verification.npz'),
        independent_forward_max_abs=float(np.max(np.abs(expected-independent))),
        source_sha256=sha(__file__),adapter_source_sha256=source_hashes(),reused_gelu_source_sha256=sha(_V8_PATH),
        elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='New random initialization; fit/validation views only. No old learned evaluator weights, qualification arrays or intervention effects accessed.'))
    (out/'DONE').write_text('fit_complete_selected_checkpoint_frozen\n')


if __name__=='__main__': main()
