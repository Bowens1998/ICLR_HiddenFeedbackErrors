"""Dispatch frozen observation-only comparisons through unchanged planners."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for k in ['plan','base','output']: p.add_argument('--'+k,required=True)
    p.add_argument('--phase',choices=['engineering','scientific'],required=True)
    p.add_argument('--index',type=int,required=True)
    a=p.parse_args(); plan=Path(a.plan); entries=json.loads(plan.read_text())[a.phase]
    entry=entries[a.index]; assert entry['index']==a.index
    manifest=plan.parent/entry['manifest']; assert sha(manifest)==entry['manifest_sha256']
    frozen=json.loads(manifest.read_text()); bank=Path(entry['bank'])
    assert sha(bank/'manifest.json')==frozen['bank_manifest_sha256']
    if entry['sigma']:
        accepted=json.loads((bank/'acceptance.json').read_text())
        assert accepted['manifest_sha256']==sha(bank/'manifest.json') and accepted['cases']==128
        assert accepted['verifier_sha256']==sha(Path(__file__).with_name('accept_observation_noise.py'))
    base=Path(a.base); out=Path(a.output)/f'job_{a.index}'; cases=2 if a.phase=='engineering' else 128
    scripts=Path(__file__).parent; sim=base/'releases/visual-v1/stable-worldmodel'
    runner='run_action_manifest_planner.py' if entry['family']=='action' else 'run_manifest_planner.py'
    common=['--bank',str(bank),'--model-manifest',str(manifest),'--simulator',str(sim)]
    subprocess.run([sys.executable,str(scripts/runner),*common,'--training',entry['training'],'--official',str(base/'releases/visual-v1/official'),'--config',str(base/'assets/pusht-v1/models/config.json'),'--output',str(out),'--index',str(entry['route_index']),'--cases',str(cases)],check=True)
    verifier='accept_action_manifest_planner.py' if entry['family']=='action' else 'accept_state_staged.py'
    cmd=[sys.executable,str(scripts/verifier),*common,'--run',str(out),'--expected-cases',str(cases)]
    if entry['family']=='state': cmd+=['--output',str(out/'staged')]
    subprocess.run(cmd,check=True)
    identity=[]
    if entry['sigma']==0:
        original=Path(entry['original_run']); hashes=json.loads((original/'artifact_manifest.json').read_text())
        for i in range(cases):
            name=f'case_{i:03d}_predictions.npz'; assert sha(original/name)==hashes[name]
            with np.load(original/name) as old,np.load(out/name) as new:
                assert set(old.files)==set(new.files)
                for key in old.files: np.testing.assert_array_equal(new[key],old[key],err_msg=f'clean gate {i} {key}')
            identity.append(dict(case=i,original_sha256=sha(original/name),fresh_sha256=sha(out/name)))
    report=dict(entry=entry,phase=a.phase,cases=cases,plan_sha256=sha(plan),dispatcher_sha256=sha(Path(__file__)),zero_noise_bitwise_comparisons=identity,summary_sha256=sha(out/'summary.json'))
    (out/'noise_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NOISE ROUTE ACCEPTED',a.phase,a.index,flush=True)
if __name__=='__main__': main()
