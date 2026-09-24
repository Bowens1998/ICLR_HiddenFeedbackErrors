"""Native LeWM inference and independently reproducible task ranking diagnostics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

def average_rank(x):
    _,inv,counts=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-.5*(counts+1))[inv]

def rank_correlation(x,y):
    rx,ry=average_rank(x),average_rank(y)
    if np.std(rx)==0 or np.std(ry)==0:return None
    return float(np.corrcoef(rx,ry)[0,1])

def task_cost(states,goal,include_agent=False):
    angle=np.arctan2(np.sin(states[:,4]-goal[4]),np.cos(states[:,4]-goal[4]))
    cost=np.square(states[:,2:4]-goal[2:4]).sum(1)+900*np.square(angle)
    if include_agent:cost+=np.square(states[:,:2]-goal[:2]).sum(1)
    return cost

def score_metrics(score,cost):
    chosen=np.flatnonzero(score==score.min());selected=float(cost[chosen].mean());random=float(cost.mean());best=float(cost.min())
    return {'spearman':rank_correlation(score,cost),'selected_cost':selected,'random_cost':random,'candidate_best_cost':best,
            'regret':selected-best,'gain_vs_random':random-selected,'selected_ties':chosen.tolist()}

def main():
    from lewm_adapter import build
    p=argparse.ArgumentParser()
    for key in ['data','official','config','weights','normalization','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--label',default='single eight-epoch model')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model=build(a.official,a.config,a.weights).cuda().eval()
    norm=json.loads(Path(a.normalization).read_text())['action'];am=torch.tensor(norm['mean'],device='cuda').repeat(5);sd=torch.tensor(norm['std'],device='cuda').repeat(5)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def pixels(x):return (torch.from_numpy(x.copy()).cuda().permute(0,3,1,2).float()/255-mean)/std
    def encode(x):return model.encode({'pixels':pixels(x)[:,None]})['emb'][:,0]
    manifest=json.loads((Path(a.data)/'manifest.json').read_text());rows=[]
    with torch.inference_mode():
        for item in manifest['cases']:
            path=Path(a.data)/f"case_{item['index']:03d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            z=np.load(path);n=len(z['actions'])
            actions=np.concatenate([np.broadcast_to(z['prefix'].reshape(1,2,10),(n,2,10)),z['actions'].reshape(n,5,10)],axis=1)
            act=(torch.from_numpy(actions).cuda()-am)/sd
            history=pixels(z['history_pixels'])[None,None].expand(1,n,-1,-1,-1,-1)
            native=model.rollout({'pixels':history},act[None])['predicted_emb'][0]
            # Explicit indexing: prefix groups 0/1 are already executed; predict from group 2 onward.
            emb=encode(z['history_pixels'])[None].expand(n,-1,-1).clone()
            for k in range(2,7):
                context=emb[:,-3:];ae=model.action_encoder(act[:,k-2:k+1]);pred=model.predict(context,ae)[:,-1:];emb=torch.cat([emb,pred],1)
            torch.testing.assert_close(native,emb,rtol=2e-5,atol=2e-5)
            goal=encode(z['goal_pixels'][None])[0];realized=encode(z['terminal_pixels']);predicted=native[:,-1]
            scores={'native_latent':(predicted-goal).square().sum(1).cpu().numpy(),
                    'oracle_realized_latent':(realized-goal).square().sum(1).cpu().numpy(),
                    'oracle_realized_pixel':np.square((z['terminal_pixels'].astype('float32')-z['goal_pixels'].astype('float32'))/255).mean((1,2,3)),
                    'uniform_persistence':np.zeros(n)}
            costs={name:task_cost(z['terminal_states'],z['goal_state'],agent) for name,agent in [('block_pose',False),('agent_and_block_pose',True)]}
            row={'seed':int(z['seed']),'index':item['index'],'alignment_verified':True,'tasks':{name:{method:score_metrics(score,cost) for method,score in scores.items()} for name,cost in costs.items()}}
            np.savez_compressed(out/f"case_{item['index']:03d}_predictions.npz",**scores,**{'cost_'+k:v for k,v in costs.items()},predicted_embedding=predicted.cpu().numpy(),realized_embedding=realized.cpu().numpy(),goal_embedding=goal.cpu().numpy())
            rows.append(row);print(json.dumps(row),flush=True)
    summary={'scope':a.label+' developmental diagnostic; no baseline superiority claim','gpu':torch.cuda.get_device_name(),'cases':rows,
             'hashes':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in {'weights':a.weights,'normalization':a.normalization,'config':a.config,'bank_manifest':str(Path(a.data)/'manifest.json'),'script':__file__}.items()}}
    summary['aggregate']={task:{method:{key:float(np.mean([r['tasks'][task][method][key] for r in rows if r['tasks'][task][method][key] is not None])) if any(r['tasks'][task][method][key] is not None for r in rows) else None for key in ['spearman','selected_cost','random_cost','candidate_best_cost','regret','gain_vs_random']} for method in scores} for task in costs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    (out/'COMPLETE').write_text('developmental native ranking only\n')
if __name__=='__main__':main()
