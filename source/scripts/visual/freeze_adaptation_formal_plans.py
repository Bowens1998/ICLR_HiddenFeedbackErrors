"""Bind unchanged original actors to accepted formal train/validation banks."""
import json, hashlib
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    original=Path('runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json')
    protocol=Path('docs/maintrack/PLANNER_DATA_ADAPTATION_PROTOCOL.md')
    rows=[]
    for split,count,seed in [('train',128,1301001),('validation',32,1311001)]:
        bank=Path('data/planner_adaptation_context_formal_v1')/('contexts_'+split)
        m=json.loads((bank/'manifest.json').read_text());a=json.loads((bank/'acceptance.json').read_text())
        assert a['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and a['manifest_sha256']==sha(bank/'manifest.json')
        assert m['seed_start']==seed and len(m['cases'])==len(a['rows'])==count
        for r,v in zip(m['cases'],a['rows']):
            assert r['sha256']==v['sha256']==sha(bank/f"case_{r['index']:03d}.npz")
            assert r['index']==v['index'] and r['seed']==v['seed'] and v['independently_replayed_branches']==33
        p=json.loads(original.read_text());assert p['layout']=='pusht_nonlinear_pose' and len(p['routes'])==48
        routes=[i*8+j for i in range(6) for j in [2,3,6,7]]
        for i in routes: assert p['models'][p['routes'][i]['model_index']]['score'] in ['pose_encoded','state']
        p['bank_manifest_sha256']=sha(bank/'manifest.json')
        p['scope']='Formal adaptation '+split+' collection with unchanged original actors and searches. All selected trajectories retained; no held-out evaluation.'
        p['adaptation_formal']=dict(split=split,original_plan_sha256=sha(original),context_acceptance_sha256=sha(bank/'acceptance.json'),execute_routes=routes,source_sha256=sha(__file__),protocol_sha256=sha(protocol))
        out=Path('runs/hpg/planner_data_adaptation_v1')/f'planning_{split}_plan.json'
        with out.open('x') as f: json.dump(p,f,indent=2);f.write('\n')
        rows.append(dict(split=split,cases=count,plan_sha256=sha(out),bank_manifest_sha256=sha(bank/'manifest.json'),independently_replayed_branches=count*33))
    out=Path('runs/hpg/planner_data_adaptation_v1/context_formal_recovery.json')
    with out.open('x') as f:json.dump(dict(status='PASS160_CONTEXTS5280_REPLAYED_BRANCHES',job='41478891',rows=rows),f,indent=2);f.write('\n')
    print('PASS',[(r['split'],r['cases']) for r in rows])

if __name__=='__main__':main()
