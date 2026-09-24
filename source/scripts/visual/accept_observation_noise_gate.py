"""Require every frozen engineering branch before scientific dispatch."""
import argparse,hashlib,json
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for k in ['plan','runs','output']: p.add_argument('--'+k,required=True)
a=p.parse_args(); plan=Path(a.plan); entries=json.loads(plan.read_text())['engineering']; assert len(entries)==48
reports=[]
for entry in entries:
    run=Path(a.runs)/f"job_{entry['index']}"; np=run/'noise_acceptance.json'; report=json.loads(np.read_text())
    assert report['entry']==entry and report['phase']=='engineering' and report['cases']==2
    assert report['plan_sha256']==sha(plan) and report['dispatcher_sha256']==sha(Path(__file__).with_name('run_observation_noise.py'))
    assert report['summary_sha256']==sha(run/'summary.json')
    summary=json.loads((run/'summary.json').read_text()); assert len(summary['cases'])==2
    assert summary['hashes']['model_manifest']==entry['manifest_sha256']
    manifest=plan.parent/entry['manifest']; assert sha(manifest)==entry['manifest_sha256']
    frozen=json.loads(manifest.read_text()); assert summary['hashes']['bank_manifest']==frozen['bank_manifest_sha256']
    for name,h in json.loads((run/'artifact_manifest.json').read_text()).items(): assert sha(run/name)==h
    for name,h in json.loads((run/'source/manifest.json').read_text()).items(): assert sha(run/'source'/name)==h
    accepted_path=run/('acceptance.json' if entry['family']=='action' else 'staged/staged_acceptance.json')
    accepted=json.loads(accepted_path.read_text()); assert accepted['cases']==2 and accepted['model_free_simulator_replay']
    verifier='accept_manifest_planner.py' if entry['family']=='action' else 'accept_state_staged.py'
    assert accepted['verifier_sha256']==sha(Path(__file__).with_name(verifier))
    if entry['family']=='action': assert accepted['mode_binding_verifier_sha256']==sha(Path(__file__).with_name('accept_action_manifest_planner.py'))
    else: assert accepted['original_summary_sha256']==sha(run/'summary.json')
    assert len(report['zero_noise_bitwise_comparisons'])==(2 if entry['sigma']==0 else 0)
    reports.append(dict(index=entry['index'],noise_acceptance_sha256=sha(np),physical_acceptance_sha256=sha(accepted_path),summary_sha256=sha(run/'summary.json')))
result=dict(routes=48,zero_noise_routes=16,noisy_routes=32,plan_sha256=sha(plan),verifier_sha256=sha(Path(__file__)),reports=reports)
output=Path(a.output); assert not output.exists(); output.write_text(json.dumps(result,indent=2)+'\n'); print('ALL48 ENGINEERING ROUTES ACCEPTED')
