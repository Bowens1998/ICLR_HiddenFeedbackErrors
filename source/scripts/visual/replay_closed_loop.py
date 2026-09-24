"""Replay saved controls in a fresh simulator without loading a learned model."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    for key in ['run','bank','source']:p.add_argument('--'+key,required=True)
    a=p.parse_args();sys.path.insert(0,a.source);from stable_worldmodel.envs.pusht.env import PushT
    run=Path(a.run);assert (run/'numeric_acceptance.json').exists();summary=json.loads((run/'summary.json').read_text());rows=[]
    for row in summary['cases']:
        i=row['index'];z=np.load(run/f'case_{i:03d}_predictions.npz');bank=np.load(Path(a.bank)/f'case_{i:03d}.npz');e=PushT(resolution=224);obs,_=e.reset(seed=row['seed']);frames=[e.render().copy()]
        for j,act in enumerate(bank['prefix']):
            obs,*_=e.step(act)
            if (j+1)%5==0:frames.append(e.render().copy())
        states=[obs['state'].copy()];outside=[]
        for j,act in enumerate(z['actions']):
            obs,*_=e.step(act);states.append(obs['state'].copy())
            boxes=[shape.cache_bb() for body in [e.agent,e.block] for shape in body.shapes]
            outside.append(any(bb.left<0 or bb.bottom<0 or bb.right>512 or bb.top>512 for bb in boxes))
            if (j+1)%5==0:frames.append(e.render().copy())
        e.close();np.testing.assert_allclose(states,z['states'],rtol=0,atol=1e-7);np.testing.assert_array_equal(frames,z['frames']);np.testing.assert_array_equal(outside,z['boundary'])
        rows.append({'index':i,'max_state_abs_difference':float(np.max(np.abs(np.array(states)-z['states']))),'frames_exact':True,'boundary_flags_exact':True})
    result={'accepted_cases':len(rows),'state_tolerance':1e-7,'cases':rows,'scope':'fresh simulator replay of every saved primitive action; no learned model loaded'}
    (run/'replay_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'accepted_cases':len(rows),'largest_state_difference':max(r['max_state_abs_difference'] for r in rows)},indent=2))
if __name__=='__main__':main()
