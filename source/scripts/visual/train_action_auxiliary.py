"""Fixed-update action-auxiliary adaptation; original experiments remain intact."""
import argparse,hashlib,json,math,os,time,itertools
from update_budget import update_lr,chunks,cycling
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from factorial_model import make_model
from action_auxiliary import ActionAuxiliary
ARMS=['transformer_jepa','gru_jepa']

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

def main():
    p=argparse.ArgumentParser()
    for key in ['data','official','config','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--updates',type=int,required=True);p.add_argument('--validation-every',type=int,default=210);p.add_argument('--seed',type=int,default=3072);p.add_argument('--mode',choices=['none','inverse','inverse_goal'],required=True);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    arm=ARMS[a.index];out=Path(a.output)/arm/a.mode;out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    source=out/'source';source.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('update_budget.py'),Path(__file__).with_name('action_auxiliary.py'),Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('lewm_adapter.py'),Path(a.official)/'jepa.py',Path(a.official)/'module.py',Path(a.config)]:
        (source/path.name).write_bytes(path.read_bytes())
    (source/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
    model=make_model(a.official,a.config,arm,a.seed);initial={'encoder':tensor_hash(model.encoder),'action_encoder':tensor_hash(model.action_encoder),
          'temporal_core':tensor_hash(model.predictor)}
    cuda_rng_before=torch.cuda.get_rng_state().clone()
    heads=ActionAuxiliary();initial['auxiliary']=tensor_hash(heads)
    assert torch.equal(cuda_rng_before,torch.cuda.get_rng_state()), 'Auxiliary initialization changed CUDA RNG'
    model=model.cuda();heads=heads.cuda();parameters=list(model.parameters())+list(heads.parameters());from module import SIGReg
    reg=SIGReg().cuda();opt=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=1e-3)
    norm=json.loads((Path(a.data)/'normalization.json').read_text())['action'];am=torch.tensor(norm['mean'],device='cuda').repeat(5);asd=torch.tensor(norm['std'],device='cuda').repeat(5)
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];isd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    datasets={s:Clips(Path(a.data)/s) for s in ['train','validation']}
    loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=s=='train',drop_last=s=='train',num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2,generator=torch.Generator().manual_seed(a.seed)) for s,d in datasets.items()}
    report={'arm':arm,'seed':a.seed,'evaluations':[],'gpu':torch.cuda.get_device_name(),'job_id':os.environ.get('SLURM_JOB_ID'),'parameters':sum(p.numel() for p in model.parameters()),
            'initial_hashes':initial,'clip_counts':{s:len(d) for s,d in datasets.items()},'normalization':norm,'mode':a.mode,'batch_size':a.batch_size,'auxiliary_parameters':sum(p.numel() for p in heads.parameters()),
            'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
            'scope':'action-auxiliary prior-art adaptation; state labels not used by objective; no scientific claim before matched planning acceptance',
            'checkpoint_selection':'minimum validation total training objective; also retain fixed last checkpoint',
            'schedule':'per-update 10% linear warmup then cosine to zero; fixed update budget independent of dataset size','expected_updates':a.updates,'validation_every':a.validation_every}
    best=float('inf');started=time.perf_counter()
    train_stream=cycling(loaders['train']);completed=0
    for evaluation,(begin,end) in enumerate(chunks(a.updates,a.validation_every)):
        assert completed==begin
        row={'evaluation':evaluation+1,'update_start':begin,'update_end':end,'lr_first':update_lr(begin,a.updates),'lr_last':update_lr(end-1,a.updates)}
        for split,loader in loaders.items():
            training=split=='train';model.train(training);heads.train(training);total=np.zeros(7);n=0;start=time.perf_counter();latents=[];clipped=0
            batches=itertools.islice(train_stream,end-begin) if training else loader
            for step,(pixels,actions,states) in enumerate(batches):
                if training:
                    for group in opt.param_groups:group['lr']=update_lr(completed,a.updates)
                pixels=(pixels.cuda(non_blocking=True).float()/255-im)/isd;actions=(actions.cuda(non_blocking=True)-am)/asd
                assert torch.isfinite(actions).all()
                if training:opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3])
                    pred_loss=(pred-z['emb'][:,1:]).square().mean()
                    if training:reg_loss=reg(z['emb'].transpose(0,1))
                    else:
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(930001+step);reg_loss=reg(z['emb'].transpose(0,1))
                    auxiliary,parts=heads.objective(z['emb'],actions[:,:3],a.mode)
                    weighted_reg=.09*reg_loss
                    loss=pred_loss+weighted_reg+auxiliary
                assert torch.isfinite(loss)
                if training:
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(gn);clipped+=int(gn>1);opt.step();completed+=1
                else:latents.append(z['emb'].detach().float().cpu().flatten(0,1))
                b=len(pixels);total+=b*np.array([float(v.detach()) for v in [loss,pred_loss,reg_loss,parts['inverse'],parts['goal'],weighted_reg,auxiliary]]);n+=b
            assert n>0;torch.cuda.synchronize();metrics=dict(zip(['loss','prediction_loss','sigreg_loss','inverse_loss','goal_loss','weighted_sigreg_loss','weighted_auxiliary_loss'],(total/n).tolist()));metrics.update(sequences=n,seconds=time.perf_counter()-start,clipped_batches=clipped)
            if latents:metrics['mean_token_std']=float(torch.cat(latents).std(0).mean())
            row[split]=metrics
        if row['validation']['loss']<best:
            best=row['validation']['loss'];torch.save(model.state_dict(),out/'best_weights.pt');torch.save(heads.state_dict(),out/'best_auxiliary.pt');report['best_evaluation']=evaluation+1;report['best_update']=completed
        assert completed==end;report['completed_updates']=completed
        report['evaluations'].append(row);report['wall_seconds']=time.perf_counter()-started
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
    assert completed==a.updates
    torch.save(model.state_dict(),out/'last_weights.pt');torch.save(heads.state_dict(),out/'last_auxiliary.pt')
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    (out/'COMPLETE').write_text('developmental training only\n')
if __name__=='__main__':main()
