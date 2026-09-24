"""Compare complete auxiliary runs with audited matched direct-state references."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for k in ['action','scaling','pairing','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();ap=Path(a.action);sp=Path(a.scaling);pp=Path(a.pairing)
action=json.loads(ap.read_text());state=json.loads(sp.read_text());pairing=json.loads(pp.read_text())
assert action['manifest_sha256']==pairing['action_manifest_sha256'] and state['manifest_sha256']==pairing['state_scaling_manifest_sha256']
assert action['bank_manifest_sha256']==state['bank_manifest_sha256'] and action['cases']==state['cases']==128
assert len(pairing['rows'])==18 and len(action['tables'])==72
references={(r['replica'],r['arm'],r['checkpoint'],r['algorithm']):r for r in state['tables'] if r['episodes']==256 and r['updates']==21000 and r['arm'].endswith('state')}
assert len(references)==24;rows=[]
for r in action['tables']:
    arm=r['arm'].replace('_jepa','_state');s=references[r['replica'],arm,r['checkpoint'],r['algorithm']]
    x=np.asarray(action['per_case_costs'][str(r['route_index'])]);y=np.asarray(state['per_case_costs'][str(s['route_index'])])
    assert x.shape==y.shape==(128,)
    rows.append(dict(replica=r['replica'],arm=r['arm'],mode=r['mode'],checkpoint=r['checkpoint'],algorithm=r['algorithm'],
        action_route=r['route_index'],state_route=s['route_index'],action_cost=float(x.mean()),state_cost=float(y.mean()),
        mean_cost_difference=float((x-y).mean()),success_difference=r['success']-s['success']))
Path(a.output).write_text(json.dumps(dict(rows=rows,action_sha256=sha(ap),scaling_sha256=sha(sp),pairing_sha256=sha(pp),script_sha256=sha(Path(__file__)),
    scope='72 paired action-method minus direct-state development comparisons; data/update/initial-shared-module matching audited, but state supervision/capacity/recursive representation/objectives differ. Fixed-last primary, best sensitivity; no isolated auxiliary-loss causal claim across these model families.'),indent=2)+'\n')
print('COMPARED72 AUXILIARY/STATE ROUTES')
