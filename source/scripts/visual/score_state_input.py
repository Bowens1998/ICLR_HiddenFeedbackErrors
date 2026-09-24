"""Privileged true-state candidate scoring; not a deployed visual method."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import state_features
from state_input_core import make_state_core
from factorial_task_scores import token_scores
from score_action_rank import task_cost,score_metrics
from evaluation_precision import configure_evaluation_precision

def main():
    p=argparse.ArgumentParser()
    for key in ['run','bank','official','config','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--checkpoint',choices=['best','last'],default='best');a=p.parse_args();run=Path(a.run);assert (run/'COMPLETE').exists()
    r=json.loads((run/'summary.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    precision=configure_evaluation_precision();model=make_state_core(a.official,a.config,r['seed']);weights=run/f'{a.checkpoint}_weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
    norm=r['normalization'];am=torch.tensor(norm['mean'],device='cuda').repeat(5);sd=torch.tensor(norm['std'],device='cuda').repeat(5)
    sm=torch.tensor(r['target_normalization']['mean'],device='cuda');ss=torch.tensor(r['target_normalization']['std'],device='cuda')
    def encode(x):return (state_features(torch.from_numpy(x.copy()).cuda().float())-sm)/ss
    manifest=json.loads((Path(a.bank)/'manifest.json').read_text());rows=[]
    with torch.inference_mode():
        for item in manifest['cases']:
            path=Path(a.bank)/f"case_{item['index']:03d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);n=len(z['actions'])
            act=np.concatenate([np.broadcast_to(z['prefix'].reshape(1,2,10),(n,2,10)),z['actions'].reshape(n,5,10)],1)
            act=(torch.from_numpy(act).cuda()-am)/sd
            initial=encode(z['history_states'])[None].expand(n,-1,-1).clone()
            # Full-prefix and cached rolling-window paths must agree at the original tolerance.
            native=initial.clone();act_emb=model.action_encoder(act)
            for k in range(2,7):native=torch.cat([native,model.predict(native[:,-3:],act_emb[:,k-2:k+1])[:,-1:]],1)
            emb=initial.clone()
            for k in range(2,7):emb=torch.cat([emb,model.predict(emb[:,-3:],model.action_encoder(act[:,k-2:k+1]))[:,-1:]],1)
            torch.testing.assert_close(native,emb,rtol=2e-5,atol=2e-5)
            goal=encode(z['goal_state'][None])[0].cpu().numpy();predicted=native[:,-1].cpu().numpy();realized=encode(z['terminal_states']).cpu().numpy()
            scores={'native':token_scores(predicted,goal,'privileged_gru_state',r['target_normalization']),
                    'oracle_realized':token_scores(realized,goal,'privileged_gru_state',r['target_normalization'])}
            costs={task:task_cost(z['terminal_states'],z['goal_state'],agent) for task,agent in [('block_pose',False),('agent_and_block_pose',True)]}
            zero=np.ones(n);zero[0]=0;scores['zero_action']={k:zero for k in costs};scores['uniform']={k:np.zeros(n) for k in costs}
            row={'index':item['index'],'seed':int(z['seed']),'alignment_verified':True,
                 'tasks':{task:{method:score_metrics(score[task],cost) for method,score in scores.items()} for task,cost in costs.items()}}
            np.savez_compressed(out/f"case_{item['index']:03d}_predictions.npz",predicted_tokens=predicted,realized_tokens=realized,goal_tokens=goal,
                                **{'score_'+method+'_'+task:value for method,s in scores.items() for task,value in s.items()},**{'cost_'+k:v for k,v in costs.items()})
            rows.append(row);print(json.dumps(row),flush=True)
    summary={'arm':'privileged_gru_state','checkpoint':a.checkpoint,'training_seed':r['seed'],'gpu':torch.cuda.get_device_name(),'precision':precision,'cases':rows,'target_normalization':r['target_normalization'],
             'scope':'privileged diagnostic: true state histories and goal state at inference; not a deployable visual method',
             'hashes':{'weights':hashlib.sha256(weights.read_bytes()).hexdigest(),'training_summary':hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
                       'bank_manifest':hashlib.sha256((Path(a.bank)/'manifest.json').read_bytes()).hexdigest()}}
    summary['aggregate']={task:{method:{key:float(np.mean([v['tasks'][task][method][key] for v in rows if v['tasks'][task][method][key] is not None])) if any(v['tasks'][task][method][key] is not None for v in rows) else None
                                      for key in ['spearman','selected_cost','random_cost','candidate_best_cost','regret','gain_vs_random']} for method in scores} for task in costs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    (out/'COMPLETE').write_text('requires independent CPU rescoring\n')
if __name__=='__main__':main()
