"""Matched-horizon expert validation audit; descriptive, no task-based reselection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from state_input_core import make_state_core
from factorial_model import state_features
from evaluation_precision import configure_evaluation_precision

def main():
 p=argparse.ArgumentParser()
 for k in ['data','runs','official','config','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);precision=configure_evaluation_precision()
 folder=Path(a.data)/'validation';e=np.load(folder/'episodes.npz');starts=[];episodes=[]
 for i,(o,n) in enumerate(zip(e['offsets'],e['lengths'])):
  indices=int(o)+np.arange(max(0,int(n)-35));starts.extend(indices.tolist());episodes.extend([i]*len(indices))
 starts=np.array(starts);episodes=np.array(episodes);raw=np.load(folder/'state.npy',mmap_mode='r');actions=np.load(folder/'action.npy',mmap_mode='r')
 states=np.array(raw[starts[:,None]+np.arange(0,36,5)[None]]);act=np.array(actions[starts[:,None]+np.arange(35)[None]]).reshape(-1,7,10)
 reports=[]
 for index in range(3):
  run=Path(a.runs)/f'job_{index}';r=json.loads((run/'summary.json').read_text());model=make_state_core(a.official,a.config,r['seed']).cuda().eval();weights=run/'best_weights.pt';model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True))
  mean=torch.tensor(r['target_normalization']['mean'],device='cuda');std=torch.tensor(r['target_normalization']['std'],device='cuda');am=torch.tensor(r['normalization']['mean'],device='cuda').repeat(5);asd=torch.tensor(r['normalization']['std'],device='cuda').repeat(5)
  modes={k:[] for k in ['recursive','teacher_forced','zero_future_action']};truth=[]
  with torch.inference_mode():
   for begin in range(0,len(starts),128):
    s=(state_features(torch.tensor(states[begin:begin+128],device='cuda').float())-mean)/std;ac=(torch.tensor(act[begin:begin+128],device='cuda')-am)/asd;truth.append(s[:,3:].cpu().numpy())
    for mode in modes:
     history=s[:,:3].clone();prediction=[];input_actions=ac.clone()
     if mode=='zero_future_action':input_actions[:,2:]=-am/asd
     encoded=model.action_encoder(input_actions)
     for k in range(2,7):
      context=s[:,k-2:k+1] if mode=='teacher_forced' else history[:,-3:]
      pred=model.predict(context,encoded[:,k-2:k+1])[:,-1:];prediction.append(pred);history=torch.cat([history,pred],1)
     modes[mode].append(torch.cat(prediction,1).cpu().numpy())
  truth=np.concatenate(truth);preds={k:np.concatenate(v) for k,v in modes.items()};norm=r['target_normalization'];mn=np.array(norm['mean']);sd=np.array(norm['std'])
  def feat(x):return np.concatenate([x[...,:4],np.sin(x[...,4:5]),np.cos(x[...,4:5])],-1)
  persist=np.repeat(states[:,2:3],5,1);linear=persist.copy();delta=states[:,2]-states[:,1];delta[:,4]=(delta[:,4]+np.pi)%(2*np.pi)-np.pi
  linear[...,:5]+=np.arange(1,6)[None,:,None]*delta[:,None,:5]
  preds['persistence']=(feat(persist)-mn)/sd;preds['linear_extrapolation']=(feat(linear)-mn)/sd
  metrics={}
  for mode,pred in preds.items():
   decoded=pred*sd+mn;actual=states[:,3:];angle=np.arctan2(decoded[...,4],decoded[...,5])-actual[...,4];angle=(angle+np.pi)%(2*np.pi)-np.pi
   errors={'normalized_mse':np.mean((pred-truth)**2,-1),'block_pose_error':np.sum((decoded[...,2:4]-actual[...,2:4])**2,-1)+900*angle**2,'agent_xy_error':np.sum((decoded[...,:2]-actual[...,:2])**2,-1)}
   metrics[mode]={}
   for name,values in errors.items():
    by_episode=np.stack([values[episodes==i].mean(0) for i in np.unique(episodes)]);metrics[mode][name]={'clip_mean':values.mean(0).tolist(),'episode_mean':by_episode.mean(0).tolist()}
  np.savez_compressed(out/f'job_{index}_predictions.npz',truth=truth,states=states,episodes=episodes,starts=starts,**preds)
  reports.append({'index':index,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'metrics':metrics,'normalization':norm})
 report={'precision':precision,'scope':'existing expert validation episodes; matched starts/horizons; descriptive audit only, no model reselection','horizons_primitive_steps':[5,10,15,20,25],'clips':len(starts),'episodes':len(np.unique(episodes)),'runs':reports,'data_manifest_sha256':hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest()}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
