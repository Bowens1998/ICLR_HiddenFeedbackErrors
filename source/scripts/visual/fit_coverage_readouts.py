"""Frozen-encoder equal-label goal heads; no evaluation-bank inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from nonlinear_pose_cost import numpy_pose


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['config','plan','data','official','model-config','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--index',type=int,choices=range(6),required=True);p.add_argument('--engineering',action='store_true');a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']]
 assert entry['score']=='pose_encoded' and entry['checkpoint']=='last';rep=a.index//2
 folder=Path(a.data)/f'replica_{rep}';dr=json.loads((folder/'report.json').read_text());da=json.loads((folder/'acceptance.json').read_text())
 assert dr['replica']==rep and not dr['engineering'] and da['status']=='PASS' and not da['engineering']
 assert da['report_sha256']==sha(folder/'report.json') and dr['config_sha256']==sha(a.config)
 assert da['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
 training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
 tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
 hp=Path(entry['goal_head']['path']);assert sha(hp)==entry['goal_head']['sha256']
 with np.load(hp) as h:mean=h['mean'];scale=h['scale']
 tm=np.array([256.,256.,256.,256.,0.,0.]);ts=np.array([256.,256.,256.,256.,1.,1.])
 precision=configure_evaluation_precision();torch.set_num_threads(2)
 backbone=make_model(a.official,a.model_config,entry['arm'],tr['seed']);backbone.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);backbone=backbone.cuda().eval()
 out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
 im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 rows=[];seed=430001+a.index;updates=20 if a.engineering else cfg['fit_updates']
 for arm in ['expert','broad']:
  row=next(v for v in dr['rows'] if v['arm']==arm);sub=folder/arm
  for f,h in row['files_sha256'].items():assert sha(sub/f)==h
  images=np.load(sub/'pixels.npy',mmap_mode='r');poses=np.load(sub/'poses.npz');target=poses['target'];assert len(images)==len(target)==cfg['labels_per_arm']==512
  encoded=[]
  with torch.inference_mode():
   for begin in range(0,512,64):
    x=(torch.tensor(np.array(images[begin:begin+64]),device='cuda').permute(0,3,1,2).float()/255-im)/sd
    encoded.append(backbone.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy())
  encoded=np.concatenate(encoded);assert encoded.shape==(512,192) and np.isfinite(encoded).all()
  dest=out/arm;dest.mkdir();np.savez_compressed(dest/'features.npz',encoded=encoded,target=target,source_index=poses['source_index'])
  xt=torch.tensor((encoded.astype(float)-mean)/scale,device='cuda',dtype=torch.float32);yt=torch.tensor((target-tm)/ts,device='cuda',dtype=torch.float32)
  torch.manual_seed(seed);model=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)).cuda()
  initial={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()};np.savez_compressed(dest/'initial.npz',**initial)
  opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.);rng=np.random.default_rng(seed+11);losses=[]
  for step in range(updates):
   ix=rng.integers(0,512,cfg['batch_size']);loss=(model(xt[ix])-yt[ix]).square().mean();assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
  weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()};weights.update(mean=mean,scale=scale,target_mean=tm,target_scale=ts)
  assert all(np.isfinite(v).all() for v in weights.values());np.savez_compressed(dest/'weights.npz',**weights)
  prediction=numpy_pose(encoded,weights);np.savez_compressed(dest/'train_prediction.npz',prediction=prediction)
  rows.append(dict(arm=arm,labels=512,seed=seed,updates=updates,batch_size=cfg['batch_size'],learning_rate=cfg['learning_rate'],first_loss=losses[0],last_loss=losses[-1],train_standardized_mse=float(np.square((prediction-target)/ts).mean()),files_sha256={f.name:sha(f) for f in dest.iterdir()}))
 report=dict(schema='pusht_coverage_goal_heads_v1',index=a.index,replica=rep,engineering=a.engineering,rows=rows,config_sha256=sha(a.config),plan_sha256=sha(a.plan),data_report_sha256=sha(folder/'report.json'),data_acceptance_sha256=sha(folder/'acceptance.json'),backbone_sha256=entry['weights_sha256'],normalization_reference_sha256=sha(hp),precision=precision,gpu=torch.cuda.get_device_name(),source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['fit_coverage_readouts.py','factorial_model.py','lewm_adapter.py','nonlinear_pose_cost.py','evaluation_precision.py']},scope='Fixed-backbone goal heads,512 labels each, shared expert input normalization and fixed physical target scaling, shared initialization and minibatch stream; no evaluation goals used. Independent fit/feature acceptance required.')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('FITTED',a.index)


if __name__=='__main__':main()
