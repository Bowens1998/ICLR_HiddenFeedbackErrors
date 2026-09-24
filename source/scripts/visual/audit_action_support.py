"""Descriptive expert/planner action-distribution audit; not an overlap theorem."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('data/pusht_expert_state_audit');source=json.loads(Path('data/pusht_pilot_metadata/manifest.json').read_text());sets={};hashes={}
for split in ['train','validation']:
 folder=root/split
 for key in ['state.npy','action.npy','episodes.npz']:
  h=hashlib.sha256((folder/key).read_bytes()).hexdigest();assert h==source['splits'][split]['files'][key];hashes[str(folder/key)]=h
 e=np.load(folder/'episodes.npz');starts=np.concatenate([int(o)+np.arange(max(0,int(n)-35)) for o,n in zip(e['offsets'],e['lengths'])]);action=np.load(folder/'action.npy');state=np.load(folder/'state.npy');sets['expert_'+split]=(action[starts[:,None]+np.arange(10,35)],state[starts+10])
for label,path in [('development_inview','data/pusht_inview_rank_v1'),('development_stress','data/pusht_action_rank_v2'),('prospective_validation','data/pusht_prospective_v1/validation')]:
 bank=Path(path);r=json.loads((bank/'manifest.json').read_text());actions=[];states=[]
 for row in r['cases']:
  p=bank/f"case_{row['index']:03d}.npz";assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];z=np.load(p);actions.append(z['actions'][1:]);states.append(np.repeat(z['history_states'][-1:,:],len(z['actions'])-1,axis=0))
 sets[label]=(np.concatenate(actions),np.concatenate(states));hashes[str(bank/'manifest.json')]=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
rows={};q=[0,.1,.5,.9,.99,1]
for label,(action,state) in sets.items():
 groups=action.astype(float).reshape(-1,5,5,2);variation=np.sqrt(np.mean((groups-groups.mean(2,keepdims=True))**2,(1,2,3)));magnitude=np.linalg.norm(action,axis=-1).mean(1);distance=np.linalg.norm(state[:,:2]-state[:,2:4],axis=1);toward=state[:,2:4]-state[:,:2];first=groups[:,0].mean(1);cosine=np.sum(first*toward,1)/np.maximum(np.linalg.norm(first,axis=1)*np.linalg.norm(toward,axis=1),1e-12)
 rows[label]={'sequences':len(action),'quantiles':q,'within_group_action_rms_variation_quantiles':np.quantile(variation,q).tolist(),'fraction_exactly_constant_within_all_groups':float(np.mean(variation<1e-12)),'mean_action_norm_quantiles':np.quantile(magnitude,q).tolist(),'agent_block_distance_quantiles':np.quantile(distance,q).tolist(),'first_group_toward_block_cosine_mean':float(cosine.mean()),'action_component_mean':action.mean((0,1)).tolist(),'action_component_std':action.std((0,1)).tolist()}
report={'scope':'descriptive distributions only; overlapping expert clips and grouped planner candidates are not independent samples; no causal support claim; zero candidate excluded consistently','rows':rows,'hashes':hashes};out=Path('outputs/maintrack/action_support_audit');out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');lines=['# Expert/planner action audit','','Zero candidate excluded; descriptive overlapping-clip statistics, not causal evidence.','','| Source | Sequences | Constant groups, all five | Median action norm | Median within-group RMS variation | Median agent/block distance |','|---|---:|---:|---:|---:|---:|']
for name,r in rows.items():lines.append(f"| {name} | {r['sequences']} | {r['fraction_exactly_constant_within_all_groups']:.1%} | {r['mean_action_norm_quantiles'][2]:.4f} | {r['within_group_action_rms_variation_quantiles'][2]:.4f} | {r['agent_block_distance_quantiles'][2]:.1f} |")
out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
