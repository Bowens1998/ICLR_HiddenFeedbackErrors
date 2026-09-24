"""Rebuild proposals, geometric decisions, labels and rendered coverage images."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def circle_polygon_overlap(center,radius,vertices):
    v=np.asarray(vertices);w=np.roll(v,-1,axis=0);edges=w-v;rel=center-v
    cross=edges[:,0]*rel[:,1]-edges[:,1]*rel[:,0]
    inside=np.all(cross>=0) or np.all(cross<=0)
    t=np.clip(np.sum(rel*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
    distance=np.min(np.linalg.norm(center-(v+t[:,None]*edges),axis=1))
    return bool(inside or distance<=radius)


def main():
 p=argparse.ArgumentParser()
 for k in ['run','features','config','simulator']:p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.run);r=json.loads((out/'report.json').read_text());cfg=json.loads(Path(a.config).read_text());rep=r['replica']
 assert r['config_sha256']==sha(a.config) and r['schema']==cfg['schema']=='pusht_coverage_intervention_v1'
 for f,h in r['source_sha256'].items():assert sha(Path(__file__).with_name(f))==h
 assert sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py')==r['simulator_sha256']
 f=Path(a.features)/f'job_{2*rep}';fr=json.loads((f/'report.json').read_text());assert sha(f/'report.json')==r['input_report_sha256'] and sha(f/'acceptance.json')==r['input_acceptance_sha256']
 assert sha(f/'train_features.npz')==fr['files_sha256']['train_features.npz'];features=np.load(f/'train_features.npz')
 assert sha(out/'attempts.json')==r['attempts_sha256'];attempts=json.loads((out/'attempts.json').read_text());count=8 if r['engineering'] else 512
 chosen=np.random.default_rng(cfg['expert_selection_seeds'][rep]).choice(len(features['target']),512,replace=False)[:count]
 sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
 env=PushT(resolution=224);env.reset(seed=0)
 def restore(s):
  env.block.angle=float(s[4]);env.block.position=tuple(s[2:4]);env.agent.position=tuple(s[:2]);env.agent.velocity=(0.,0.)
  env.space.reindex_shapes_for_body(env.block);env.space.reindex_shapes_for_body(env.agent)
 def geometry():
  agent=next(iter(env.agent.shapes));center=np.array(env.agent.local_to_world(agent.offset));radius=agent.radius
  polygons=[(np.array([env.block.local_to_world(v) for v in shape.get_vertices()]),shape.radius) for shape in env.block.shapes]
  if np.any(center-radius<8) or np.any(center+radius>504):return 'not_fully_visible'
  for v,bevel in polygons:
   if np.any(v.min(0)-bevel<8) or np.any(v.max(0)+bevel>504):return 'not_fully_visible'
  if any(circle_polygon_overlap(center,radius+bevel,v) for v,bevel in polygons):return 'agent_block_overlap'
  return 'accepted'
 accepted=[];rng=np.random.default_rng(cfg['broad_sampling_seeds'][rep])
 assert len(attempts)==r['broad_attempts']<=cfg['max_broad_attempts']
 for index,row in enumerate(attempts):
  s=np.r_[rng.uniform(*cfg['broad_position_bounds'],size=4),rng.uniform(0,2*np.pi),0.,0.]
  assert row['index']==index;np.testing.assert_array_equal(row['state'],s);restore(s);assert geometry()==row['reason']
  if row['reason']=='accepted':accepted.append(index)
 assert len(accepted)==count and accepted[-1]==len(attempts)-1
 rows=[]
 for row in r['rows']:
  arm=row['arm'];folder=out/arm
  for name,h in row['files_sha256'].items():assert sha(folder/name)==h
  z=np.load(folder/'poses.npz');pixels=np.load(folder/'pixels.npy',mmap_mode='r');assert row['count']==len(z['states'])==len(pixels)==count
  if arm=='expert':
   np.testing.assert_array_equal(z['source_index'],chosen);np.testing.assert_array_equal(z['expert_identity'],features['identity'][chosen]);t=features['target'][chosen];expected=np.column_stack([t[:,:4],np.arctan2(t[:,4],t[:,5]),np.zeros((count,2))])
  else:
   assert arm=='broad';np.testing.assert_array_equal(z['source_index'],accepted);expected=np.array([attempts[i]['state'] for i in accepted]);assert z['expert_identity'].shape==(0,2)
  np.testing.assert_array_equal(z['states'],expected)
  targets=np.column_stack([expected[:,:4],np.sin(expected[:,4]),np.cos(expected[:,4])]);np.testing.assert_array_equal(z['target'],targets)
  for i,s in enumerate(expected):restore(s);np.testing.assert_array_equal(env.render(),pixels[i])
  rows.append(dict(arm=arm,labels=count,exact_images=count))
 env.close();assert [v['arm'] for v in rows]==['expert','broad']
 result=dict(status='PASS',engineering=r['engineering'],report_sha256=sha(out/'report.json'),verifier_sha256=sha(__file__),rows=rows,all_attempts_reconstructed=len(attempts),scope='Independent RNG/labels, analytic circle-polygon/visibility decisions and exact same-state pixel reconstruction. No model or task outcome claim.')
 (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
