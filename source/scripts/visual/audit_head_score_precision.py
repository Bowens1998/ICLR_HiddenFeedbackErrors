"""Diagnose failed score reconstruction without accepting or changing traces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from state_head_numpy import decode_head
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.run)
r=json.loads((root/'summary.json').read_text());head=np.load(root/'head_weights.npz');mn=np.array(r['target_normalization']['mean']);sd=np.array(r['target_normalization']['std']);bad=[];maximum=0
for row in r['cases']:
    path=root/f"case_{row['index']:03d}_predictions.npz"
    with np.load(path,allow_pickle=False) as z:
        tokens=z['tokens'];costs=z['costs'];goal=decode_head(z['goal_tokens'],head)*sd+mn;state=decode_head(tokens,head)*sd+mn
        angle=np.arctan2(state[...,4],state[...,5])-np.arctan2(goal[4],goal[5]);angle=(angle+np.pi)%(2*np.pi)-np.pi
        rebuilt=np.square(state[...,2:4]-goal[2:4]).sum(-1)+900*angle**2
        delta=np.abs(rebuilt-costs);maximum=max(maximum,float(delta.max()));where=np.argwhere(delta>2e-3+5e-5*np.abs(costs))
        if len(where):
            details=[]
            for t,c in where:
                details.append({'iteration':int(t),'candidate':int(c),'stored_score':float(costs[t,c]),'float64_score':float(rebuilt[t,c]),'absolute_difference':float(delta[t,c]),'allowed_difference':float(2e-3+5e-5*abs(costs[t,c])),
                                'predicted_sin_cos':state[t,c,4:6].tolist(),'predicted_radius':float(np.hypot(*state[t,c,4:6])),'goal_sin_cos':goal[4:6].tolist(),'goal_radius':float(np.hypot(*goal[4:6]))})
            elite_changes=[len(set(np.argsort(costs[t],kind='stable')[:30])-set(np.argsort(rebuilt[t],kind='stable')[:30])) for t in range(30)]
            entry={'case':row['index'],'seed':row['seed'],'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'violations':details,'selected_stored':list(map(int,np.unravel_index(np.argmin(costs),costs.shape))),
                   'minimum_float64_on_saved_candidates':list(map(int,np.unravel_index(np.argmin(rebuilt),rebuilt.shape))),'elite_membership_changes_per_round':elite_changes,
                   'scope':'fixed saved populations only; does not simulate a counterfactual CEM trajectory'}
            bad.append(entry);print(json.dumps(entry),flush=True)
result={'cases_scanned':len(r['cases']),'violating_cases':bad,'max_absolute_difference':maximum,'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'scope':'diagnostic only; original tolerance unchanged and failed route remains unaccepted'}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
