"""Render stored poses without stepping physics; audit image/state alignment."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['simulator','validation','bank','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 folder=Path(a.validation);e=np.load(folder/'episodes.npz');pixels=np.load(folder/'pixels.npy',mmap_mode='r');states=np.load(folder/'state.npy',mmap_mode='r')
 bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());rows=[];saved={}
 samples=[]
 # First/middle/last episode, fixed row35; no choice based on image/model error.
 for ep in [0,len(e['offsets'])//2,len(e['offsets'])-1]:
  index=int(e['offsets'][ep])+35;assert e['lengths'][ep]>35
  samples.append(('validation',index,states[index],pixels[index],0))
 for item in bm['cases'][:3]:
  f=bank/f"case_{item['index']:03d}.npz";assert sha(f)==item['sha256']
  with np.load(f) as z:samples.append(('planning_goal',item['index'],z['goal_state'].copy(),z['goal_pixels'].copy(),item['seed']))
 for domain,index,state,original,seed in samples:
  env=PushT(resolution=224);env.reset(seed=int(seed))
  # _set_state advances physics; direct assignments preserve the stored pose.
  env.agent.position=state[:2].tolist();env.agent.velocity=state[-2:].tolist()
  env.block.angle=float(state[4]);env.block.position=state[2:4].tolist()
  env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
  restored=env._get_obs();np.testing.assert_allclose(restored[:4],state[:4],rtol=0,atol=1e-10)
  assert abs(np.angle(np.exp(1j*(restored[4]-state[4]))))<1e-10
  rebuilt=env.render().copy();env.close();assert rebuilt.shape==original.shape==(224,224,3)
  difference=np.abs(rebuilt.astype(float)-original.astype(float));key=f'{domain}_{index}'
  saved[key+'_original']=original;saved[key+'_restored']=rebuilt;saved[key+'_state']=state
  rows.append(dict(domain=domain,index=index,exact_equal=bool(np.array_equal(original,rebuilt)),mean_absolute_pixel_difference=float(difference.mean()),changed_pixel_fraction=float(np.any(difference>0,axis=-1).mean()),max_absolute_pixel_difference=float(difference.max())))
 np.savez_compressed(out/'images.npz',**saved)
 r=dict(status='COMPLETE_SAMPLED_AUDIT',rows=rows,source_sha256=sha(__file__),simulator_sha256=sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py'),episodes_sha256=sha(folder/'episodes.npz'),bank_manifest_sha256=sha(bank/'manifest.json'),images_sha256=sha(out/'images.npz'),scope='Three fixed validation rows and first three planning goals; exact pose restoration without a physics step. Does not prove full rendering equivalence or dynamics equivalence. Pixel differences require interpretation.')
 (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
