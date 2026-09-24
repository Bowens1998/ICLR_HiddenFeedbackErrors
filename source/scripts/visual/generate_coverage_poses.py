"""Equal-label expert and broad static-pose images, same pinned renderer."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from coverage_pose_sampling import expert_indices,broad_proposals


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['config','features','simulator','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--replica',type=int,choices=range(3),required=True);p.add_argument('--engineering',action='store_true');a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());assert cfg['schema']=='pusht_coverage_intervention_v1' and cfg['labels_per_arm']==512
 folder=Path(a.features)/f'job_{2*a.replica}';fr=json.loads((folder/'report.json').read_text());fa=json.loads((folder/'acceptance.json').read_text())
 assert not fr['engineering'] and fa['status']=='PASS' and fa['report_sha256']==sha(folder/'report.json')
 assert sha(folder/'train_features.npz')==fr['files_sha256']['train_features.npz']
 with np.load(folder/'train_features.npz') as z:target=z['target'];identity=z['identity']
 ix=expert_indices(len(target),512,cfg['expert_selection_seeds'][a.replica]);count=8 if a.engineering else 512;ix=ix[:count]
 sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
 env=PushT(resolution=224);env.reset(seed=0)
 def restore(state):
  env.agent.position=state[:2].tolist();env.agent.velocity=(0.,0.);env.block.angle=float(state[4]);env.block.position=state[2:4].tolist()
  env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
  obs=env._get_obs();np.testing.assert_allclose(obs[:4],state[:4],rtol=0,atol=1e-10)
  assert abs(np.angle(np.exp(1j*(obs[4]-state[4]))))<1e-10
 def reason():
  for body in [env.agent,env.block]:
   for shape in body.shapes:
    bb=shape.cache_bb()
    if min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504:return 'not_fully_visible'
  block_shapes=set(env.block.shapes)
  for shape in env.agent.shapes:
   for contact in env.space.shape_query(shape):
    if contact.shape in block_shapes and len(contact.contact_point_set.points):return 'agent_block_overlap'
  return 'accepted'
 out=Path(a.output)/f'replica_{a.replica}';out.mkdir(parents=True,exist_ok=False)
 rows=[];attempts=[]
 for arm in ['expert','broad']:
  states=[];images=[];ids=[]
  if arm=='expert':
   for index in ix:
    t=target[index];s=np.r_[t[:4],np.arctan2(t[4],t[5]),0.,0.];restore(s)
    states.append(s);images.append(env.render().copy());ids.append(int(index))
  else:
   for index,s in enumerate(broad_proposals(cfg['broad_sampling_seeds'][a.replica],cfg['max_broad_attempts'],cfg['broad_position_bounds'])):
    restore(s);why=reason();attempts.append(dict(index=index,state=s.tolist(),reason=why))
    if why!='accepted':continue
    states.append(s);images.append(env.render().copy());ids.append(index)
    if len(states)==count:break
  assert len(states)==count,'Do not reduce quota or relax filters'
  states=np.array(states);labels=np.column_stack([states[:,:4],np.sin(states[:,4]),np.cos(states[:,4])]);images=np.array(images)
  assert images.shape==(count,224,224,3) and images.dtype==np.uint8
  sub=out/arm;sub.mkdir();np.save(sub/'pixels.npy',images);np.savez_compressed(sub/'poses.npz',states=states,target=labels,source_index=np.array(ids),expert_identity=identity[ix] if arm=='expert' else np.empty((0,2),dtype=np.int64))
  rows.append(dict(arm=arm,count=count,files_sha256={f.name:sha(f) for f in sub.iterdir()}))
 env.close();(out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
 report=dict(schema=cfg['schema'],replica=a.replica,engineering=a.engineering,rows=rows,config_sha256=sha(a.config),input_report_sha256=sha(folder/'report.json'),input_acceptance_sha256=sha(folder/'acceptance.json'),attempts_sha256=sha(out/'attempts.json'),broad_attempts=len(attempts),source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['generate_coverage_poses.py','coverage_pose_sampling.py']},simulator_sha256=sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py'),scope='Same renderer, fixed expert labels versus sampled static-pose labels. No dynamics step or model filtering; broad labels are additional offline supervision. Independent acceptance required.')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
