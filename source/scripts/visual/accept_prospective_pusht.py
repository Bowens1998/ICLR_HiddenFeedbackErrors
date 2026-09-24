"""Data-only QA for prospectively generated banks; no model evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);sets=[];reports={}
for split,count,start in [('validation',32,890001),('heldout',128,900001)]:
 bank=root/split;r=json.loads((bank/'manifest.json').read_text());assert (bank/'COMPLETE').exists() and len(r['cases'])==count and r['seed_start']==start and r['minimum_goal_block_displacement']==40;seeds=[];distances=[]
 for item in r['cases']:
  path=bank/f"case_{item['index']:03d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);assert int(z['seed'])==item['seed'];seeds.append(item['seed']);d=float(np.linalg.norm(z['goal_state'][2:4]-z['history_states'][-1,2:4]));assert d>=40;np.testing.assert_allclose(d,item['initial_goal_block_distance'],rtol=1e-12);distances.append(d)
  assert z['actions'].shape==(32,25,2) and not z['actions'][0].any();assert np.max(abs(z['actions']))<=.35 and np.max(abs(z['goal_actions']))<=.35
  assert item['all_shapes_visible_every_step'];assert sum(item['goal_rejections'].values())==item['goal_attempts']-1;assert 1<=item['goal_attempts']<=128
 assert len(set(seeds))==count and all(start<=s<start+r['max_seeds'] for s in seeds);sets.append(set(seeds));reports[split]={'cases':count,'rejected_resets':len(r['rejected_seeds']),'minimum_initial_goal_distance':min(distances),'maximum_initial_goal_distance':max(distances),'manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()}
assert sets[0].isdisjoint(sets[1]);result={'data_hashes_shapes_actions_distances_seed_ranges':'accepted','scope':'data QA only; no model scoring; visibility/replay checked by generator','banks':reports};(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
