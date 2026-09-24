"""Independently replay every fresh goal witness and zero-action reference."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['bank','config','simulator','seed-audit']:p.add_argument('--'+k,required=True)
 a=p.parse_args();bank=Path(a.bank);cfg=json.loads(Path(a.config).read_text());m=json.loads((bank/'manifest.json').read_text());audit=json.loads(Path(a.seed_audit).read_text())
 assert audit['status']=='PASS' and audit['config_sha256']==sha(a.config)
 assert (bank/'COMPLETE').exists() and len(m['cases'])==cfg['evaluation_goals']==256
 assert m['seed_start']==cfg['evaluation_seed_start'] and m['max_seeds']==cfg['evaluation_max_seeds']
 assert m['script_sha256']==sha(Path(__file__).with_name('prepare_prospective_pusht.py'))
 assert m['simulator_env_sha256']==sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py')
 seeds=[v['seed'] for v in m['cases']];assert len(set(seeds))==256 and all(m['seed_start']<=s<m['seed_start']+m['max_seeds'] for s in seeds)
 sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
 rows=[]
 def visible(env):
  for body in [env.agent,env.block]:
   for shape in body.shapes:
    bb=shape.cache_bb()
    assert min(bb.left,bb.bottom)>=8 and max(bb.right,bb.top)<=504
 for i,item in enumerate(m['cases']):
  assert item['index']==i;f=bank/f'case_{i:03d}.npz';assert sha(f)==item['sha256'];z=np.load(f)
  assert int(z['seed'])==item['seed'] and z['prefix'].shape==(10,2) and z['actions'].shape==(32,25,2)
  assert not z['actions'][0].any() and np.isfinite(z['actions']).all() and np.max(abs(z['actions']))<=np.float32(.35)
  assert z['goal_actions'].shape==(25,2) and np.isfinite(z['goal_actions']).all() and np.max(abs(z['goal_actions']))<=np.float32(.35)
  distance=np.linalg.norm(z['goal_state'][2:4]-z['history_states'][-1,2:4]);assert distance>=40;np.testing.assert_allclose(distance,item['initial_goal_block_distance'],rtol=1e-12)
  for kind,actions in [('goal',z['goal_actions']),('zero',z['actions'][0])]:
   env=PushT(resolution=224);obs,_=env.reset(seed=item['seed']);visible(env)
   np.testing.assert_array_equal(obs['state'],z['history_states'][0]);np.testing.assert_array_equal(env.render(),z['history_pixels'][0])
   for t,action in enumerate(z['prefix']):
    obs,*_=env.step(action);visible(env)
    if (t+1)%5==0:
     np.testing.assert_array_equal(obs['state'],z['history_states'][(t+1)//5]);np.testing.assert_array_equal(env.render(),z['history_pixels'][(t+1)//5])
   for action in actions:obs,*_=env.step(action);visible(env)
   expected_state=z['goal_state'] if kind=='goal' else z['terminal_states'][0]
   expected_pixels=z['goal_pixels'] if kind=='goal' else z['terminal_pixels'][0]
   np.testing.assert_array_equal(obs['state'],expected_state);np.testing.assert_array_equal(env.render(),expected_pixels);env.close()
  rows.append(dict(index=i,seed=item['seed'],file_sha256=item['sha256']))
 result=dict(status='PASS',cases=256,exact_witness_and_zero_replays=512,manifest_sha256=sha(bank/'manifest.json'),config_sha256=sha(a.config),seed_audit_sha256=sha(a.seed_audit),verifier_sha256=sha(__file__),rows=rows,scope='All goal witnesses and zero references independently replayed with history pixels/states and every-step visibility. Other31 stored candidate branches inherit generator checks; no model outcomes.')
 (bank/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS256_GOALS_512_REPLAYS')


if __name__=='__main__':main()
