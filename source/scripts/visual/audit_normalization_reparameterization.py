"""Training-only function/gradient equivalence and one-step Adam non-equivalence."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from analyze_coverage_goals import sha


def absorb(mean,scale,weight,bias):
    w=weight/scale[None,:]
    return w,bias-w@mean


def main():
    p=argparse.ArgumentParser()
    for k in ['features','frozen','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();torch.set_num_threads(2);rows=[]
    for i in range(6):
        feature=Path(a.features)/f'job_{i}';fr=json.loads((feature/'report.json').read_text());assert sha(feature/'features.npz')==fr['features_sha256'];z=dict(np.load(feature/'features.npz'))
        frozen=Path(a.frozen)/f'job_{i}';r=json.loads((frozen/'report.json').read_text());ac=json.loads((frozen/'acceptance.json').read_text());assert ac['status']=='PASS' and ac['report_sha256']==sha(frozen/'report.json')
        for stage in ['trained_cls','initial_cls']:
            row=next(v for v in r['rows'] if v['stage']==stage)
            for name in ['weights.npz','initial.npz']:assert sha(frozen/stage/name)==row['files_sha256'][name]
            h=dict(np.load(frozen/stage/'weights.npz'));initial=dict(np.load(frozen/stage/'initial.npz'))
            mean=torch.tensor(h['mean']);scale=torch.tensor(h['scale'])
            def model():
                m=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)).double()
                m.load_state_dict({k:torch.tensor(v) for k,v in initial.items()});return m
            normalized=model();raw=model()
            with torch.no_grad():
                w,b=absorb(mean,scale,normalized[0].weight,normalized[0].bias)
                raw[0].weight.copy_(w);raw[0].bias.copy_(b)
            ix=np.random.default_rng(430001+i+11).integers(0,512,256)
            xn=torch.tensor(z[stage][ix],dtype=torch.float64,requires_grad=True);xr=xn.detach().clone().requires_grad_(True)
            target=torch.tensor((z['target'][ix]-h['target_mean'])/h['target_scale'],dtype=torch.float64)
            pn=normalized((xn-mean)/scale);pr=raw(xr)
            torch.testing.assert_close(pn,pr,rtol=1e-8,atol=1e-9)
            on=torch.optim.Adam(normalized.parameters(),lr=.001);orr=torch.optim.Adam(raw.parameters(),lr=.001)
            ((pn-target)**2).mean().backward();((pr-target)**2).mean().backward()
            torch.testing.assert_close(xn.grad,xr.grad,rtol=1e-7,atol=1e-9)
            output_delta=float((pn-pr).detach().abs().max());gradient_delta=float((xn.grad-xr.grad).abs().max())
            on.step();orr.step()
            with torch.no_grad():after_delta=float((normalized((xn-mean)/scale)-raw(xr)).abs().max())
            rows.append(dict(index=i,stage=stage,initial_output_max_difference=output_delta,initial_feature_gradient_max_difference=gradient_delta,output_max_difference_after_one_head_adam_step=after_delta,source_report_sha256=sha(frozen/'report.json')))
    result=dict(status='COMPLETE12_REPARAMETERIZATION_CHECKS',rows=rows,source_sha256=sha(__file__),identity='W_raw=W/scale; b_raw=b-W_raw@mean. Same forward function and derivative with respect to raw features before updates.',scope='Float64 training-feature/initial-head diagnostic, one fixed training minibatch per model. Head-only Adam step at.001; no encoder update, model refit or heldout outcome. Function-preserving reparameterization does not make ordinary Adam parameterization-invariant. Not a causal explanation of full-run failure.')
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['index'],r['stage'],r['initial_output_max_difference'],r['initial_feature_gradient_max_difference'],r['output_max_difference_after_one_head_adam_step'])


if __name__=='__main__':main()
