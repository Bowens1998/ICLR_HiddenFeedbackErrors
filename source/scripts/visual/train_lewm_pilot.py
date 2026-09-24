"""From-scratch LeWM pipeline pilot on episode-disjoint PushT development data.

Eight epochs, official objective; not a full baseline comparison or paper result.
"""
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from lewm_adapter import build
class Clips(Dataset):
 def __init__(self,folder):
  self.folder=Path(folder);e=np.load(self.folder/'episodes.npz');self.starts=np.concatenate([int(o)+np.arange(int(n)-19) for o,n in zip(e['offsets'],e['lengths'])]);self.arrays=None
 def __len__(self):return len(self.starts)
 def __getitem__(self,i):
  if self.arrays is None:self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in ['pixels','action']}
  start=int(self.starts[i]);pixels=np.array(self.arrays['pixels'][start:start+20:5]);action=np.array(self.arrays['action'][start:start+20])
  return torch.from_numpy(pixels).permute(0,3,1,2),torch.from_numpy(action).reshape(4,10)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--official',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--max-steps',type=int,default=0);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(3072);np.random.seed(3072);torch.set_num_threads(4)
 source=out/'source';source.mkdir()
 for file in [Path(__file__),Path(__file__).with_name('lewm_adapter.py'),Path(a.official)/'jepa.py',Path(a.official)/'module.py',Path(a.config)]:
  (source/file.name).write_bytes(file.read_bytes())
 (source/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
 model=build(a.official,a.config).cuda();from module import SIGReg
 reg=SIGReg().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-3)
 # A short pilot keeps the initial learning rate fixed; this differs from the full official warmup/cosine schedule.
 norm=json.loads((Path(a.data)/'normalization.json').read_text())['action'];am=torch.tensor(norm['mean'],device='cuda').repeat(5);asd=torch.tensor(norm['std'],device='cuda').repeat(5)
 im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];isd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
 datasets={s:Clips(Path(a.data)/s) for s in ['train','validation']}
 loaders={s:DataLoader(d,batch_size=128,shuffle=s=='train',drop_last=s=='train',num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2,
  generator=torch.Generator().manual_seed(3072)) for s,d in datasets.items()}
 report={'purpose':'from_scratch_pipeline_pilot','epochs':[],'gpu':torch.cuda.get_device_name(),'job_id':os.environ.get('SLURM_JOB_ID'),'torch':torch.__version__,
  'seed':3072,'parameters':sum(p.numel() for p in model.parameters()),'clip_counts':{s:len(d) for s,d in datasets.items()},'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
  'deviations_from_full_recipe':['8 epochs instead of 100','fixed LR rather than warmup/cosine','episode-disjoint pilot subset','train-only normalization'],'normalization':norm}
 best=float('inf');started=time.perf_counter()
 for epoch in range(a.epochs):
  row={'epoch':epoch+1}
  for split,loader in loaders.items():
   training=split=='train';model.train(training);total=np.zeros(3);n=0;start=time.perf_counter();latents=[]
   for step,(pixels,actions) in enumerate(loader):
    if a.max_steps and step>=a.max_steps:break
    pixels=pixels.cuda(non_blocking=True).float().div_(255);pixels=(pixels-im)/isd
    actions=actions.cuda(non_blocking=True);actions=torch.nan_to_num((actions-am)/asd)
    if training:opt.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
     z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3])
     pred_loss=(pred-z['emb'][:,1:]).square().mean();reg_loss=reg(z['emb'].transpose(0,1));loss=pred_loss+.09*reg_loss
    assert torch.isfinite(loss)
    if training:
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    else:latents.append(z['emb'].detach().float().cpu().reshape(-1,192))
    b=len(pixels);total+=b*np.array([float(loss.detach()),float(pred_loss.detach()),float(reg_loss.detach())]);n+=b
   torch.cuda.synchronize();metrics=dict(zip(['loss','prediction_loss','sigreg_loss'],(total/n).tolist()));metrics.update(sequences=n,seconds=time.perf_counter()-start)
   if latents:metrics['mean_latent_std']=float(torch.cat(latents).std(0).mean())
   row[split]=metrics
  if row['validation']['prediction_loss']<best:
   best=row['validation']['prediction_loss'];torch.save(model.state_dict(),out/'best_weights.pt');report['best_epoch']=epoch+1
  report['epochs'].append(row);report['wall_seconds']=time.perf_counter()-started
  (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
 torch.save(model.state_dict(),out/'last_weights.pt');(out/'COMPLETE').write_text('pipeline pilot only\n')
if __name__=='__main__':main()
