"""Fixed three-stage probes, same initialization and minibatch stream, training only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from analyze_coverage_goals import sha
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser()
    for name in ['features','config','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    folder=Path(a.features)/f'job_{a.index}';r=json.loads((folder/'report.json').read_text());ac=json.loads((folder/'acceptance.json').read_text())
    assert ac['status']=='PASS' and ac['index']==r['index']==a.index and ac['images_per_stage']==512
    assert ac['report_sha256']==sha(folder/'report.json') and ac['features_sha256']==r['features_sha256']==sha(folder/'features.npz')
    assert ac['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_stages.py'))
    cfg=json.loads(Path(a.config).read_text());assert r['config_sha256']==sha(a.config)
    assert cfg['fit_updates']==2000 and cfg['batch_size']==256 and cfg['learning_rate']==.001
    z=dict(np.load(folder/'features.npz'));target=z['target'];assert target.shape==(512,6)
    tm=np.array([256.,256.,256.,256.,0.,0.]);ts=np.array([256.,256.,256.,256.,1.,1.])
    yt=torch.tensor((target-tm)/ts,device='cuda',dtype=torch.float32)
    torch.set_num_threads(2);out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
    rows=[];seed=430001+a.index;initial_ref=None
    for stage in ['trained_cls','projected','initial_cls']:
        x=z[stage].astype(np.float64);assert x.shape==(512,192) and np.isfinite(x).all()
        mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
        xt=torch.tensor((x-mean)/scale,device='cuda',dtype=torch.float32)
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)).cuda()
        initial={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        if initial_ref is None:initial_ref=initial
        else:
            for k in initial:np.testing.assert_array_equal(initial[k],initial_ref[k])
        dest=out/stage;dest.mkdir();np.savez_compressed(dest/'initial.npz',**initial)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(seed+11);losses=[]
        for step in range(2000):
            ix=rng.integers(0,512,256);loss=(model(xt[ix])-yt[ix]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        weights.update(mean=mean,scale=scale,target_mean=tm,target_scale=ts)
        assert all(np.isfinite(v).all() for v in weights.values())
        np.savez_compressed(dest/'weights.npz',**weights)
        pred=numpy_pose(x,weights);np.savez_compressed(dest/'train_prediction.npz',prediction=pred)
        rows.append(dict(stage=stage,labels=512,updates=2000,seed=seed,batch_size=256,learning_rate=.001,first_loss=losses[0],last_loss=losses[-1],train_standardized_mse=float(np.mean(((pred-target)/ts)**2)),files_sha256={f.name:sha(f) for f in dest.iterdir()}))
    result=dict(status='FITTED_REQUIRES_ACCEPTANCE',index=a.index,rows=rows,feature_report_sha256=sha(folder/'report.json'),feature_acceptance_sha256=sha(folder/'acceptance.json'),features_sha256=sha(folder/'features.npz'),config_sha256=sha(a.config),source_sha256=sha(__file__),scope='Three fixed probes; own512 training-only population feature normalization, std floor1e-6, fixed physical target normalization, shared initial weights and minibatch stream; no holdout inputs.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('FITTED_THREE_STAGES',a.index)


if __name__=='__main__':main()
