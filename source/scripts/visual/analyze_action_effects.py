"""Keep independent training-pool variation in the complete auxiliary-mode matrix."""
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from action_comparison_pairs import comparison_pairs

p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sp=Path(a.summary);s=json.loads(sp.read_text());assert s['layout']=='action_auxiliary' and s['cases']==128
tables=s['tables'];pairs=comparison_pairs(tables);assert len(pairs)==len(s['contrasts'])==108
byroute={r['route_index']:r for r in tables};assert set(byroute)==set(range(72))
expected={(left,right,label) for left,right,label in pairs}
actual={(c['left_route'],c['right_route'],c['contrast']) for c in s['contrasts']};assert actual==expected
cells=defaultdict(list);effects=defaultdict(list)
for r in tables:cells[(r['arm'],r['mode'],r['checkpoint'],r['algorithm'])].append(r)
cell_rows=[]
for key,rs in sorted(cells.items()):
    rs=sorted(rs,key=lambda x:x['replica']);assert [r['replica'] for r in rs]==[0,1,2]
    cell_rows.append(dict(zip(['arm','mode','checkpoint','algorithm'],key),
        pool_costs=[r['mean_cost'] for r in rs],pool_successes=[r['success'] for r in rs],
        mean_cost=float(np.mean([r['mean_cost'] for r in rs])),mean_success=float(np.mean([r['success'] for r in rs]))))
for c in s['contrasts']:
    left,right=byroute[c['left_route']],byroute[c['right_route']]
    assert left['replica']==right['replica'] and left['arm']==right['arm'] and left['checkpoint']==right['checkpoint']
    key=(c['contrast'],left['arm'],left['checkpoint'],left['mode'],right['mode'],left['algorithm'],right['algorithm'])
    effects[key].append(dict(replica=left['replica'],left_route=left['route_index'],right_route=right['route_index'],
        mean_cost_difference=c['mean_cost_difference'],success_difference=left['success']-right['success']))
effect_rows=[]
for key,rs in sorted(effects.items()):
    rs=sorted(rs,key=lambda x:x['replica']);assert [r['replica'] for r in rs]==[0,1,2]
    delta=np.array([r['mean_cost_difference'] for r in rs]);assert np.isfinite(delta).all()
    effect_rows.append(dict(zip(['contrast','arm','checkpoint','left_mode','right_mode','left_algorithm','right_algorithm'],key),
        pools=rs,mean_cost_difference=float(delta.mean()),between_pool_difference_sd=float(delta.std(ddof=1)),
        cost_improvement_pools=int((delta<0).sum()),cost_worsening_pools=int((delta>0).sum())))
assert len(cell_rows)==24 and len(effect_rows)==36
out=dict(cells=cell_rows,effects=effect_rows,summary_sha256=hashlib.sha256(sp.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Complete72-route action-auxiliary development matrix:24 three-pool cells and36 paired-effect groups. Fixed-last primary, best sensitivity. Three-pool SD is descriptive, not a confidence interval; shared128 goals are not independent replicates. DA-LeWM-style adaptation is a prior-art baseline, not a novel method or exact author reproduction.')
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print('WROTE24 CELLS AND36 THREE-POOL EFFECTS')
