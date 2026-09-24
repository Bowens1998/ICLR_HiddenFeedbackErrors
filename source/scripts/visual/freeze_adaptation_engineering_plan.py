"""Rebind unchanged original actors to accepted new engineering contexts."""
import json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    original=Path('runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json')
    bank=Path('data/planner_adaptation_context_engineering_v1');m=json.loads((bank/'manifest.json').read_text());a=json.loads((bank/'acceptance.json').read_text())
    assert a['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and a['manifest_sha256']==sha(bank/'manifest.json')
    assert m['seed_start']==1291001 and len(m['cases'])==2
    for row in m['cases']:assert row['sha256']==sha(bank/f"case_{row['index']:03d}.npz")
    p=json.loads(original.read_text());assert p['layout']=='pusht_nonlinear_pose' and len(p['routes'])==48
    routes=[i*8+j for i in range(6) for j in [2,3,6,7]]
    for i in routes:assert p['models'][p['routes'][i]['model_index']]['score'] in ['pose_encoded','state']
    p['bank_manifest_sha256']=sha(bank/'manifest.json');p['scope']='Adaptation engineering only: unchanged original actor/head identities, two fresh contexts. Execute only declared24 physical/state routes; no scientific training or outcome claim.'
    p['adaptation_engineering']=dict(original_plan_sha256=sha(original),context_acceptance_sha256=sha(bank/'acceptance.json'),execute_routes=routes,source_sha256=sha(__file__))
    out=Path('runs/hpg/planner_data_adaptation_v1/planning_engineering_plan.json')
    with out.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(','.join(map(str,routes)))

if __name__=='__main__':main()
