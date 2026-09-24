"""Fixed-budget PushT pose heads on accepted, frozen trajectory features."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ['features','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True)
    p.add_argument('--engineering',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();folder=Path(a.features)/f'job_{a.index}'
    r=json.loads((folder/'report.json').read_text());acc=json.loads((folder/'acceptance.json').read_text())
    assert r['protocol']=='pusht_nonlinear_transfer_features_v1' and r['index']==a.index
    assert not r['engineering'] and not acc['engineering'] and acc['status']=='PASS'
    assert acc['report_sha256']==sha(folder/'report.json')
    assert acc['verifier_sha256']==sha(Path(__file__).with_name('accept_pose_features.py'))
    for f,h in r['files_sha256'].items():assert sha(folder/f)==h
    data={}
    for split in ['train','validation']:
        with np.load(folder/f'{split}_features.npz') as z:data[split]={k:z[k] for k in z.files}
        assert data[split]['target'].shape==(r['counts'][split],6)
        assert np.isfinite(data[split]['target']).all()
    n=r['counts']['train'];assert n>256
    target_mean=data['train']['target'].mean(0);target_scale=data['train']['target'].std(0)
    target_scale=np.where(target_scale>1e-12,target_scale,1.)
    yt=torch.tensor((data['train']['target']-target_mean)/target_scale,dtype=torch.float32,device=a.device)
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
    updates=20 if a.engineering else 2000;seed=430001+a.index;rows=[]
    for role in ['encoded','predicted']:
        x=data['train'][role].astype(np.float64);mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
        xt=torch.tensor((x-mean)/scale,dtype=torch.float32,device=a.device)
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)).to(a.device)
        initial={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(seed+11);losses=[]
        for step in range(updates):
            ix=rng.integers(0,n,256);prediction=model(xt[ix]);loss=(prediction-yt[ix]).square().mean()
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        assert all(np.isfinite(v).all() for v in weights.values())
        head=dict(weights,mean=mean,scale=scale,target_mean=target_mean,target_scale=target_scale)
        sub=out/role;sub.mkdir();np.savez_compressed(sub/'weights.npz',**head);np.savez_compressed(sub/'initial.npz',**initial)
        model=model.double().eval();metrics={}
        for source in ['encoded','predicted']:
            tokens=data['validation'][source].astype(np.float64);inp=(tokens-mean)/scale
            with torch.inference_mode():
                normalized=np.concatenate([model(torch.tensor(inp[j:j+256],device=a.device,dtype=torch.float64)).cpu().numpy() for j in range(0,len(inp),256)])
            physical=normalized*target_scale+target_mean
            np.testing.assert_allclose(physical,numpy_pose(tokens,head),rtol=1e-10,atol=1e-9)
            np.savez_compressed(sub/f'validation_{source}.npz',prediction=physical,identity=data['validation']['identity'])
            metrics[source]=float(np.square((physical-data['validation']['target'])/target_scale).mean())
        rows.append(dict(role=role,updates=updates,batch_size=256,seed=seed,first_loss=losses[0],last_loss=losses[-1],validation_standardized_mse=metrics,files_sha256={f.name:sha(f) for f in sub.glob('*.npz')}))
    report=dict(protocol='pusht_nonlinear_pose_fit_v1',index=a.index,replica=r['replica'],arm=r['arm'],engineering=a.engineering,rows=rows,
                input_report_sha256=sha(folder/'report.json'),input_acceptance_sha256=sha(folder/'acceptance.json'),input_files_sha256=r['files_sha256'],
                source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['fit_pose_readouts.py','nonlinear_pose_cost.py','evaluation_precision.py']},
                precision=precision,device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
                scope='Two fixed-last pose readouts; train-only input/target normalization, same initialization and minibatch stream across roles. No planning-based selection; independent acceptance required.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
