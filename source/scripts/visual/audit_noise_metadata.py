"""Rebind downloaded metadata; remote raw-trace acceptance remains explicit."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for k in ['plan','runs','output']:p.add_argument('--'+k,required=True)
p.add_argument('--phase',choices=['engineering','scientific'],required=True)
a=p.parse_args();plan=Path(a.plan);entries=json.loads(plan.read_text())[a.phase];root=Path(a.runs);rows=[];cases=2 if a.phase=='engineering' else 128
for e in entries:
    folder=root/f"job_{e['index']}";nf=folder/'noise_acceptance.json'
    if not nf.exists():continue
    n=json.loads(nf.read_text());sf=folder/'summary.json';s=json.loads(sf.read_text());manifest=plan.parent/e['manifest'];f=json.loads(manifest.read_text())
    assert n['phase']==a.phase and n['cases']==cases and n['entry']==e
    assert n['plan_sha256']==sha(plan) and n['dispatcher_sha256']==sha(Path(__file__).with_name('run_observation_noise.py'))
    assert n['summary_sha256']==sha(sf)==json.loads((folder/'artifact_manifest.json').read_text())['summary.json']
    assert sha(manifest)==e['manifest_sha256']==s['hashes']['model_manifest']
    route=f['routes'][e['route_index']];m=f['models'][route['model_index']]
    assert s['route_index']==e['route_index'] and s['model_index']==route['model_index']
    assert s['hashes']['weights']==m['weights_sha256'] and s['hashes']['bank_manifest']==f['bank_manifest_sha256']
    assert s['arm']==m['arm']==e['arm'] and s['checkpoint']=='last'
    assert s['algorithm']==route['algorithm']==e['algorithm'] and len(s['cases'])==cases
    assert s['training_path']==m['training_path'] and s['parameterization']==route['parameterization']
    for name,h in json.loads((folder/'source/manifest.json').read_text()).items():assert sha(folder/'source'/name)==h
    af=folder/('acceptance.json' if e['family']=='action' else 'staged/staged_acceptance.json');acc=json.loads(af.read_text())
    assert acc['cases']==cases and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
    verifier='accept_manifest_planner.py' if e['family']=='action' else 'accept_state_staged.py'
    assert acc['verifier_sha256']==sha(Path(__file__).with_name(verifier))
    if e['family']=='action':
        assert acc['mode_binding_verifier_sha256']==sha(Path(__file__).with_name('accept_action_manifest_planner.py'))
        assert acc['mode']==s['mode']==e['mode']
    else:assert acc['original_summary_sha256']==sha(sf)
    assert len(n['zero_noise_bitwise_comparisons'])==(cases if e['sigma']==0 else 0)
    rows.append(dict(index=e['index'],sigma=e['sigma'],summary_sha256=sha(sf),noise_acceptance_sha256=sha(nf),physical_acceptance_sha256=sha(af)))
result=dict(phase=a.phase,accepted=len(rows),expected=len(entries),complete=len(rows)==len(entries),rows=rows,plan_sha256=sha(plan),script_sha256=sha(Path(__file__)),scope='Downloaded metadata/source/model/data/acceptance identities rebound locally. Raw candidate archives and physical replay were independently verified remotely; no local raw-array replay claimed. Partial recovery is not a full matrix result.')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('REBOUND',len(rows),'OF',len(entries),'ROUTES')
