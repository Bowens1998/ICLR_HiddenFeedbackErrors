"""All nine primary and eight secondary contrasts from complete accepted goal cells."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json
from artifact_io import atomic_npz
from evaluation_lock import add_protocol_arguments,read_protocol
from input_lock import verify_environment
from goal_statistics import indices,paired_summary


def contrast_arrays(row,ac,b):
    if row['bank']=='confirmation_B':return b[row['left']][:,-1],b[row['right']][:,-1]
    if row['id'].startswith('C'):
        def value(condition):
            prefix='C/'+row['objective']+'/'+condition+'/'
            free=ac[prefix+'free'][...,-1]
            return free-ac[prefix+'actual'][...,-1] if row['metric']=='G_free_minus_actual' else free
        return value(row['left']),value(row['right'])
    kind='A_dual' if row['budget']=='A_dual_twelve_directions' else 'A_core'
    prefix=kind+'/'+row['objective']+'/'
    return ac[prefix+row['left']][...,-1],ac[prefix+row['right']][...,-1]


def main():
    p=argparse.ArgumentParser();add_protocol_arguments(p)
    p.add_argument('--acceptance',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    protocol,design=read_protocol(a.protocol_lock,a.protocol_sha256,ROOT);verify_environment(design,'fiber-projection-v1')
    source=Path(a.acceptance);r=json.loads((source/'report.json').read_text())
    assert (source/'DONE').exists() and r['status']=='PASS_COMPLETE_INDEPENDENT_CONFIRMATION_RECONSTRUCTION'
    assert r['protocol_sha256']==a.protocol_sha256 and r['scientific_design_sha256']==protocol['design_sha256']
    assert r['all_primary_and_secondary_cells_complete'] and r['goals_per_bank']==256
    data={}
    for k in ['AC','B']:
        path=source/(k+'_goal_cells.npz');assert sha(path)==r[k+'_goal_cells_sha256'];data[k]=dict(np.load(path))
        assert all(len(x)==256 and np.isfinite(x).all() for x in data[k].values())
    spec=design['statistical_design'];draws={bank:indices(design['root_seed'],bank,256,20000) for bank in ['confirmation_A_C','confirmation_B']}
    rows=[]
    for family,slots in [('primary',9),('secondary_C',8)]:
        assert len(spec[family])==slots
        for row in spec[family]:
            left,right=contrast_arrays(row,data['AC'],data['B'])
            assert left.shape==right.shape and left.shape[0]==256
            if row['bank']=='confirmation_A_C':assert left.shape==(256,len(row['groups']),len(row['streams']))
            gap=row.get('metric')=='G_free_minus_actual'
            summary=paired_summary(left,right,draws[row['bank']],family_size=slots,relative_mse=not gap)
            points=[]
            if left.ndim==3:
                for j,g in enumerate(row['groups']):
                    points.append(dict(group=g,left_mean=float(left[:,j].mean()),right_mean=float(right[:,j].mean()),
                        paired_change=float((left[:,j]-right[:,j]).mean())))
            rows.append(dict(contrast=row,summary=summary,all_model_points=points))
    descriptive={}
    for branch,arrays in data.items():
        descriptive[branch]={}
        for key,value in arrays.items():
            v=np.asarray(value,dtype=np.float64)
            # Horizon vector when present; scalars for action-response and dose summaries.
            flat=v.reshape(-1,v.shape[-1]) if v.ndim in [2,4] else v.reshape(-1,1)
            descriptive[branch][key]=dict(mean=flat.mean(0).tolist(),
                quantiles=np.quantile(flat,[0,.05,.25,.5,.75,.95,1],axis=0,method='linear').tolist())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_npz(out/'bootstrap_indices.npz',**draws)
    atomic_json(out/'effects.json',dict(status='COMPLETE_FROZEN_CONFIRMATION_STATISTICS',protocol_sha256=a.protocol_sha256,
        scientific_design_sha256=protocol['design_sha256'],acceptance_sha256=sha(source/'report.json'),
        bootstrap_indices_sha256=sha(out/'bootstrap_indices.npz'),contrasts=rows,descriptive=descriptive,
        source_sha256=sha(__file__),interpretation='All planned models and goals retained. Intervals conditional on fixed model roster and donor banks. C gap reduction alone is not repair; crossing zero is not equivalence.'))
    (out/'DONE').write_text('complete_statistics\n')


if __name__=='__main__':main()
