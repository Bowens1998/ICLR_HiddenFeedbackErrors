"""Matched teacher-forced versus recursive five-step training intervention."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import state_features
from state_input_core import make_state_core
from evaluation_precision import configure_evaluation_precision
from train_visual_factorial import learning_rate,tensor_hash

def cache(data,split,mean,std,am,asd):
    folder=Path(data)/split;e=np.load(folder/'episodes.npz');starts=np.concatenate([int(o)+np.arange(int(n)-35) for o,n in zip(e['offsets'],e['lengths'])])
    state=np.load(folder/'state.npy',mmap_mode='r');act=np.load(folder/'action.npy',mmap_mode='r')
    s=torch.from_numpy(np.array(state[starts[:,None]+np.arange(0,36,5)[None]])).float().cuda()
    actions=torch.from_numpy(np.array(act[starts[:,None]+np.arange(35)[None]])).float().cuda().reshape(-1,7,10)
    return (state_features(s)-mean)/std,(actions-am)/asd

def forecast(model,states,actions,mode):
    history=states[:,:3];embedded=model.action_encoder(actions);predictions=[]
    for k in range(2,7):
        context=states[:,k-2:k+1] if mode=='teacher_forced' else history[:,-3:]
        predicted=model.predict(context,embedded[:,k-2:k+1])[:,-1:]
        predictions.append(predicted);history=torch.cat([history,predicted],1)
    return torch.cat(predictions,1)

def main():
    p=argparse.ArgumentParser()
    for key in ['data','reference','official','config','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--epochs',type=int,default=100);a=p.parse_args();lr=[5e-5,3e-4,1e-3][a.index%3];mode=['teacher_forced','recursive'][a.index//3]
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(3072);precision=configure_evaluation_precision()
    reference=json.loads((Path(a.reference)/'summary.json').read_text());assert hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest()==reference['data_manifest_sha256']
    model=make_state_core(a.official,a.config);assert tensor_hash(model.predictor.core)==reference['initial_hashes']['temporal_core'];assert tensor_hash(model.action_encoder)==reference['initial_hashes']['action_encoder'];model=model.cuda()
    norm=reference['target_normalization'];sn=lambda key:torch.tensor(norm[key],device='cuda');mean,std=sn('mean'),sn('std');an=reference['normalization'];am=torch.tensor(an['mean'],device='cuda').repeat(5);asd=torch.tensor(an['std'],device='cuda').repeat(5)
    datasets={split:cache(a.data,split,mean,std,am,asd) for split in ['train','validation']};assert all(torch.isfinite(v).all() for pair in datasets.values() for v in pair)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3);generator=torch.Generator(device='cuda').manual_seed(3072)
    report={'index':a.index,'mode':mode,'base_lr':lr,'seed':3072,'expected_epochs':a.epochs,'epochs':[],'normalization':an,'target_normalization':norm,'precision':precision,'data_manifest_sha256':reference['data_manifest_sha256'],
            'scope':'privileged-state diagnostic on same expert episodes; not an image-only deployed model or a loss-only causal intervention',
            'selection':'minimum recursive five-step validation MSE per LR, then per method; freeze before task scoring','clip_counts':{k:len(v[0]) for k,v in datasets.items()},'initial_hashes':{k:tensor_hash(getattr(model,k)) for k in ['predictor','action_encoder','pred_proj']},'parameters':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name()}
    source=out/'source';source.mkdir()
    for file in [Path(__file__),Path(__file__).with_name('state_input_core.py'),Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('evaluation_precision.py'),Path(__file__).with_name('train_visual_factorial.py'),Path(__file__).with_name('lewm_adapter.py'),Path(a.official)/'jepa.py',Path(a.official)/'module.py',Path(a.config)]:
        (source/file.name).write_bytes(file.read_bytes())
    (source/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
    best=float('inf');started=time.perf_counter()
    for epoch in range(a.epochs):
        rate=learning_rate(epoch,a.epochs,lr)
        for group in opt.param_groups:group['lr']=rate
        model.train();s,actions=datasets['train'];order=torch.randperm(len(s),device='cuda',generator=generator);total=0.;count=0;clipped=0
        for begin in range(0,len(s)-127,128):
            ids=order[begin:begin+128];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=forecast(model,s[ids],actions[ids],mode);loss=.5*(pred-s[ids,3:]).square().mean()
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);clipped+=int(gn>1);opt.step();total+=float(loss.detach())*128;count+=128
        model.eval();s,actions=datasets['validation'];predictions=[]
        with torch.inference_mode():
            for begin in range(0,len(s),128):predictions.append(forecast(model,s[begin:begin+128],actions[begin:begin+128],'recursive').float())
        pred=torch.cat(predictions);val=float((pred-s[:,3:]).square().mean());assert np.isfinite(val)
        if val<best:
            best=val;report['best_epoch']=epoch+1;report['best_validation_mse']=val;torch.save(model.state_dict(),out/'best_weights.pt')
            np.savez_compressed(out/'validation_best_predictions.npz',predicted=pred.cpu().numpy(),target=s[:,3:].cpu().numpy())
        row={'epoch':epoch+1,'lr':rate,'train_half_prediction_mse':total/count,'validation_prediction_mse':val,'clipped_batches':clipped,'training_sequences':count};report['epochs'].append(row);report['wall_seconds']=time.perf_counter()-started
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
    torch.save(model.state_dict(),out/'last_weights.pt');(out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n');(out/'COMPLETE').write_text('training only; requires outcome evaluation\n')
if __name__=='__main__':main()
