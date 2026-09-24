"""Matched-data visual architecture/objective development study, not a full reproduction."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from factorial_model import ARMS,make_model,state_features

class Clips(Dataset):
    def __init__(self,folder):
        self.folder=Path(folder);e=np.load(self.folder/'episodes.npz');self.starts=np.concatenate([int(o)+np.arange(int(n)-19) for o,n in zip(e['offsets'],e['lengths'])]);self.arrays=None
    def __len__(self):return len(self.starts)
    def __getitem__(self,i):
        if self.arrays is None:self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in ['pixels','action','state']}
        start=int(self.starts[i]);pixels=np.array(self.arrays['pixels'][start:start+20:5]);action=np.array(self.arrays['action'][start:start+20]);state=np.array(self.arrays['state'][start:start+20:5])
        return torch.from_numpy(pixels).permute(0,3,1,2),torch.from_numpy(action).reshape(4,10),torch.from_numpy(state).float()

def tensor_hash(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def learning_rate(epoch,epochs,base=5e-5):
    warmup=min(10,max(1,epochs//10))
    if epoch<warmup:return base*(epoch+1)/warmup
    return base*.5*(1+math.cos(math.pi*(epoch-warmup)/max(1,epochs-warmup-1)))

def main():
    p=argparse.ArgumentParser()
    for key in ['data','official','config','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--seed',type=int,default=3072);a=p.parse_args()
    arm=ARMS[a.index];out=Path(a.output)/arm;out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    source=out/'source';source.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('lewm_adapter.py'),Path(a.official)/'jepa.py',Path(a.official)/'module.py',Path(a.config)]:
        (source/path.name).write_bytes(path.read_bytes())
    (source/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
    model=make_model(a.official,a.config,arm,a.seed);initial={'encoder':tensor_hash(model.encoder),'action_encoder':tensor_hash(model.action_encoder),
          'temporal_core':tensor_hash(model.predictor.core if arm.endswith('state') else model.predictor)}
    model=model.cuda();from module import SIGReg
    reg=SIGReg().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-3)
    norm=json.loads((Path(a.data)/'normalization.json').read_text())['action'];am=torch.tensor(norm['mean'],device='cuda').repeat(5);asd=torch.tensor(norm['std'],device='cuda').repeat(5)
    raw=np.load(Path(a.data)/'train/state.npy',mmap_mode='r');f=state_features(torch.from_numpy(np.array(raw)).double());assert torch.isfinite(f).all()
    target_norm={'mean':f.mean(0).tolist(),'std':f.std(0,unbiased=True).clamp_min(1e-6).tolist(),'scope':'training rows only; x_agent,y_agent,x_block,y_block,sin_angle,cos_angle'}
    sm=torch.tensor(target_norm['mean'],device='cuda');ssd=torch.tensor(target_norm['std'],device='cuda')
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];isd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    datasets={s:Clips(Path(a.data)/s) for s in ['train','validation']}
    loaders={s:DataLoader(d,batch_size=128,shuffle=s=='train',drop_last=s=='train',num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2,generator=torch.Generator().manual_seed(a.seed)) for s,d in datasets.items()}
    report={'arm':arm,'seed':a.seed,'epochs':[],'gpu':torch.cuda.get_device_name(),'job_id':os.environ.get('SLURM_JOB_ID'),'parameters':sum(p.numel() for p in model.parameters()),
            'initial_hashes':initial,'clip_counts':{s:len(d) for s,d in datasets.items()},'normalization':norm,'target_normalization':target_norm,
            'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
            'scope':'development; one dataset and one optimizer seed before expansion; direct state arms receive extra labels; matched steps not matched parameter count',
            'checkpoint_selection':'minimum validation total training objective; also retain fixed last checkpoint',
            'schedule':'explicit 10%-epoch linear warmup (up to 10 epochs) then cosine to zero; not an assertion of exact library scheduler parity','max_steps':a.max_steps,'expected_epochs':a.epochs}
    best=float('inf');started=time.perf_counter()
    for epoch in range(a.epochs):
        lr=learning_rate(epoch,a.epochs)
        for group in opt.param_groups:group['lr']=lr
        row={'epoch':epoch+1,'lr':lr}
        for split,loader in loaders.items():
            training=split=='train';model.train(training);total=np.zeros(4);n=0;start=time.perf_counter();latents=[];clipped=0
            for step,(pixels,actions,states) in enumerate(loader):
                if a.max_steps and step>=a.max_steps:break
                pixels=(pixels.cuda(non_blocking=True).float()/255-im)/isd;actions=(actions.cuda(non_blocking=True)-am)/asd
                targets=(state_features(states.cuda(non_blocking=True))-sm)/ssd
                assert torch.isfinite(actions).all() and torch.isfinite(targets).all()
                if training:opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3])
                    if arm.endswith('jepa'):
                        pred_loss=(pred-z['emb'][:,1:]).square().mean();obs_loss=pred_loss.new_zeros(())
                        if training:reg_loss=reg(z['emb'].transpose(0,1))
                        else:
                            # Fixed validation projections; do not consume training RNG state.
                            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                                torch.manual_seed(930001+step);reg_loss=reg(z['emb'].transpose(0,1))
                        loss=pred_loss+.09*reg_loss
                    else:
                        pred_loss=(pred-targets[:,1:]).square().mean();obs_loss=(z['emb']-targets).square().mean();reg_loss=pred_loss.new_zeros(());loss=.5*(pred_loss+obs_loss)
                assert torch.isfinite(loss)
                if training:
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);clipped+=int(gn>1);opt.step()
                else:latents.append(z['emb'].detach().float().cpu().flatten(0,1))
                b=len(pixels);total+=b*np.array([float(v.detach()) for v in [loss,pred_loss,obs_loss,reg_loss]]);n+=b
            assert n>0;torch.cuda.synchronize();metrics=dict(zip(['loss','prediction_loss','observation_loss','sigreg_loss'],(total/n).tolist()));metrics.update(sequences=n,seconds=time.perf_counter()-start,clipped_batches=clipped)
            if latents:metrics['mean_token_std']=float(torch.cat(latents).std(0).mean())
            row[split]=metrics
        if row['validation']['loss']<best:
            best=row['validation']['loss'];torch.save(model.state_dict(),out/'best_weights.pt');report['best_epoch']=epoch+1
        report['epochs'].append(row);report['wall_seconds']=time.perf_counter()-started
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
    torch.save(model.state_dict(),out/'last_weights.pt')
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    (out/'COMPLETE').write_text('developmental training only\n')
if __name__=='__main__':main()
