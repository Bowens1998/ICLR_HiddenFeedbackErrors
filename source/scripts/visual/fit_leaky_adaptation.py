"""Fixed activation intervention for adaptation training; no holdout inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['config','plan','data','features','frozen-fits','official','model-config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True)
    p.add_argument('--stage',choices=['trained_cls','initial_cls'],required=True)
    p.add_argument('--engineering',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());assert cfg['activation']=='leaky_relu' and cfg['negative_slope']==.01;plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']]
    features=Path(a.features)/f'job_{a.index}';fr=json.loads((features/'report.json').read_text());fa=json.loads((features/'acceptance.json').read_text())
    assert fa['status']=='PASS' and fa['report_sha256']==sha(features/'report.json') and fa['features_sha256']==sha(features/'features.npz')
    assert fa['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_stages.py'))
    assert fr['plan_sha256']==sha(a.plan) and fr['backbone_sha256']==entry['weights_sha256']
    folder=Path(a.data)/f'replica_{a.index//2}';assert sha(folder/'report.json')==fr['data_report_sha256'] and sha(folder/'acceptance.json')==fr['data_acceptance_sha256']
    dr=json.loads((folder/'report.json').read_text());row=next(v for v in dr['rows'] if v['arm']=='broad')
    for name,expected in row['files_sha256'].items():assert sha(folder/'broad'/name)==expected
    images=np.load(folder/'broad/pixels.npy');poses=np.load(folder/'broad/poses.npz');z=np.load(features/'features.npz')
    np.testing.assert_array_equal(poses['target'],z['target']);target=z['target'];assert target.shape==(512,6)
    frozen=Path(a.frozen_fits)/f'job_{a.index}';ar=json.loads((frozen/'acceptance.json').read_text());fitr=json.loads((frozen/'report.json').read_text())
    assert ar['status']=='PASS' and ar['report_sha256']==sha(frozen/'report.json') and ar['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_readouts.py'))
    fitrow=next(v for v in fitr['rows'] if v['stage']==a.stage)
    for name,expected in fitrow['files_sha256'].items():assert sha(frozen/a.stage/name)==expected
    h=dict(np.load(frozen/a.stage/'weights.npz'));initial_head=dict(np.load(frozen/a.stage/'initial.npz'))
    training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(a.official,a.model_config,entry['arm'],tr['seed'])
    if a.stage=='trained_cls':model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    encoder=model.encoder.cuda().eval();del model
    head=nn.Sequential(nn.Linear(192,256),nn.LeakyReLU(negative_slope=.01),nn.Linear(256,256),nn.LeakyReLU(negative_slope=.01),nn.Linear(256,6)).cuda()
    head.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in initial_head.items()},strict=True)
    before_encoder={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
    mean=torch.tensor(h['mean'],device='cuda',dtype=torch.float64);scale=torch.tensor(h['scale'],device='cuda',dtype=torch.float64)
    # Norms are computed in float64 from training-only features and stored unchanged.
    np.testing.assert_array_equal(h['mean'],z[a.stage].astype(float).mean(0))
    np.testing.assert_array_equal(h['scale'],np.maximum(z[a.stage].astype(float).std(0),1e-6))
    def pixels(ix):
        x=torch.tensor(images[ix],device='cuda').permute(0,3,1,2).float()/255
        return (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.inference_mode():
        observed=encoder(pixels(np.arange(32)),interpolate_pos_encoding=True).last_hidden_state[:,0].cpu().numpy()
    np.testing.assert_allclose(observed,z[a.stage][:32],rtol=2e-5,atol=2e-5)
    yt=torch.tensor((target-h['target_mean'])/h['target_scale'],device='cuda',dtype=torch.float32)
    opt=torch.optim.Adam([{'params':encoder.parameters(),'lr':cfg['encoder_learning_rate']},{'params':head.parameters(),'lr':cfg['head_learning_rate']}],weight_decay=0.)
    rng=np.random.default_rng(430001+a.index+11);updates=cfg['engineering_updates'] if a.engineering else cfg['updates'];losses=[];t=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        ix=rng.integers(0,512,256);opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,256,cfg['microbatch_size']):
            ids=ix[start:start+cfg['microbatch_size']]
            emb=encoder(pixels(ids),interpolate_pos_encoding=True).last_hidden_state[:,0]
            loss=(head(((emb.double()-mean)/scale).float())-yt[ids]).square().mean()*len(ids)/256
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        assert all(torch.isfinite(v.grad).all() for v in list(encoder.parameters())+list(head.parameters()) if v.grad is not None)
        opt.step();losses.append(total)
    torch.cuda.synchronize();elapsed=time.monotonic()-t
    encoder_delta=max(float((v.detach().cpu()-before_encoder[k]).abs().max()) for k,v in encoder.state_dict().items())
    head_delta=max(float((v.detach().cpu()-torch.tensor(initial_head[k])).abs().max()) for k,v in head.state_dict().items())
    assert encoder_delta>0 and head_delta>0
    out=Path(a.output)/f'job_{a.index}'/a.stage;out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(encoder={k:v.detach().cpu() for k,v in encoder.state_dict().items()},head={k:v.detach().cpu() for k,v in head.state_dict().items()},normalization={k:h[k] for k in ['mean','scale','target_mean','target_scale']}),out/'weights.pt')
    result=dict(activation='leaky_relu',negative_slope=.01,status='ENGINEERING_COMPLETE' if a.engineering else 'FIT_REQUIRES_ACCEPTANCE',index=a.index,stage=a.stage,updates=updates,elapsed_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),precision=precision,encoder_parameter_delta=encoder_delta,head_parameter_delta=head_delta,encoder_trainable_parameters=sum(v.numel() for v in encoder.parameters()),head_trainable_parameters=sum(v.numel() for v in head.parameters()),first_loss=losses[0],last_loss=losses[-1],weights_sha256=sha(out/'weights.pt'),config_sha256=sha(a.config),feature_report_sha256=sha(features/'report.json'),frozen_fit_report_sha256=sha(frozen/'report.json'),source_sha256=sha(__file__),scope='No heldout inputs; CLS first32 initial feature check; finite gradients and both modules change; independent fit acceptance still required.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
