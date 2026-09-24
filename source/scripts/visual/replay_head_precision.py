"""Matched-arithmetic head replay; diagnostic only, never overrides acceptance."""
import argparse,hashlib,json,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from state_head_numpy import decode_head
from evaluation_precision import configure_evaluation_precision
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--case',type=int,default=105);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);r=json.loads((root/'summary.json').read_text());path=root/f'case_{a.case:03d}_predictions.npz';z=np.load(path,allow_pickle=False);weights=np.load(root/'head_weights.npz')
precision=configure_evaluation_precision();torch.set_num_threads(2)
head=nn.Sequential(nn.Linear(192,256),nn.GELU(),nn.Linear(256,6));head.load_state_dict({k:torch.from_numpy(weights[k]) for k in weights.files},strict=True);head=head.cuda().eval();double_head=copy.deepcopy(head).double()
mn=np.array(r['target_normalization']['mean']);sd=np.array(r['target_normalization']['std'])

def physical_score(state,goal):
    angle=np.arctan2(state[...,4],state[...,5])-np.arctan2(goal[4],goal[5]);angle=(angle+np.pi)%(2*np.pi)-np.pi
    return np.square(state[...,2:4]-goal[2:4]).sum(-1)+900*angle**2

records={};arrays={}
with torch.inference_mode():
    for mode in ['float32_contiguous','float32_strided','float64']:
        dtype=torch.float64 if mode=='float64' else torch.float32;network=double_head if mode=='float64' else head
        mean=torch.tensor(mn,device='cuda',dtype=dtype);std=torch.tensor(sd,device='cuda',dtype=dtype)
        goal_raw=network(torch.tensor(z['goal_tokens'],device='cuda',dtype=dtype));goal=goal_raw*std+mean
        states=[];scores=[];raws=[]
        for t in range(30):
            tokens=torch.tensor(z['tokens'][t],device='cuda',dtype=dtype)
            if mode=='float32_strided':
                storage=torch.zeros(300,8,192,device='cuda');storage[:,-1]=tokens;tokens=storage[:,-1]
                assert tokens.stride()==(1536,1)
            raw=network(tokens);state=raw*std+mean
            angle=torch.atan2(state[:,4],state[:,5])-torch.atan2(goal[4],goal[5]);angle=torch.atan2(angle.sin(),angle.cos())
            score=(state[:,2:4]-goal[2:4]).square().sum(1)+900*angle.square()
            raws.append(raw.cpu().numpy());states.append(state.cpu().numpy());scores.append(score.cpu().numpy())
        raw=np.stack(raws);state=np.stack(states);score=np.stack(scores);goal_np=goal.cpu().numpy()
        staged=physical_score(state.astype(np.float64),goal_np.astype(np.float64));reference=decode_head(z['tokens'],weights)
        difference=np.abs(score-z['costs']);staged_difference=np.abs(staged-z['costs'])
        records[mode]={'exact_score_match':bool(np.array_equal(score,z['costs'])),'max_score_difference':float(difference.max()),
                       'violations_original_score_tolerance':int((difference>2e-3+5e-5*np.abs(z['costs'])).sum()),
                       'staged_float64_metric_max_difference':float(staged_difference.max()),'staged_violations_original_tolerance':int((staged_difference>2e-3+5e-5*np.abs(z['costs'])).sum()),
                       'max_raw_head_difference_from_numpy_float64':float(np.abs(raw-reference).max()),
                       'candidate_7_2':{'score':float(score[7,2]),'physical_features':state[7,2].tolist(),'raw_features':raw[7,2].tolist(),
                                          'radius':float(np.hypot(*state[7,2,4:6]))}}
        arrays[mode+'_raw_head']=raw;arrays[mode+'_state']=state;arrays[mode+'_scores']=score;arrays[mode+'_goal_state']=goal_np
arrays['stored_scores']=z['costs'];np.savez_compressed(out/'head_replay_predictions.npz',**arrays)
result={'case':a.case,'precision':precision,'gpu':torch.cuda.get_device_name(),'records':records,
        'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'head_weights_sha256':hashlib.sha256((root/'head_weights.npz').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('state_head_numpy.py'),Path(__file__).with_name('evaluation_precision.py')]},
        'archive_sha256':hashlib.sha256((out/'head_replay_predictions.npz').read_bytes()).hexdigest(),
        'scope':'matched precision/layout diagnostic on one saved case; no acceptance override or counterfactual planner performance'}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'COMPLETE').write_text('head replay diagnostic finished\n');print(json.dumps(result),flush=True)
