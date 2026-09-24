"""Apply the already-fixed validation rule and freeze the complete 18-run C roster."""
import argparse
import json
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha, atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
    cp=ROOT/'strengthening/configs/C_development_selection.json';contract=json.loads(cp.read_text())
    spec=json.loads((ROOT/'strengthening/configs/C_lambda_jobs.json').read_text());results=Path(a.results);rows=[];errors={}
    lock=ROOT/'strengthening/configs/C_lambda.lock.json';manifest=ROOT/'strengthening/configs/C_formal_jobs.json'
    if lock.exists() or manifest.exists():raise ValueError('Selection is already frozen; do not overwrite')
    for row in spec['rows']:
        folder=results/row['name'];assert (folder/'DONE').exists();path=folder/'report.json';r=json.loads(path.read_text())
        assert r['status']=='PASS_FIXED_C_CONTINUATION' and r['updates']==contract['updates']
        assert r['selection_contract_sha256']==sha(cp)
        assert r['pool']==0 and r['objective']==row['objective'] and r['condition']==row['condition'] and r['anchor_lambda']==row['lambda']
        value=r['validation']['free_running']['endpoint_mse'];assert math.isfinite(value) and value>=0
        errors[row['name']]=value;rows.append(dict(name=row['name'],report_sha256=sha(path),weights_sha256=r['weights_sha256'],
            initial_checkpoint=r['initial_A_checkpoint_sha256'],head_sha256=r['head_A_sha256'],
            splits=r['split_bindings'],schedule_sha256=r['schedule_sha256'],validation_free_endpoint=value))
    assert len({x['head_sha256'] for x in rows})==len({x['schedule_sha256'] for x in rows})==1
    for objective in ['teacher','labels']:
        group=[x for x in rows if x['name'].startswith(objective)]
        assert len({x['initial_checkpoint'] for x in group})==1
        assert errors[objective+'_T0']>0
    scores={str(v):sum(errors[f'{obj}_T2_{v}']/errors[f'{obj}_T0'] for obj in ['teacher','labels'])/2
            for v in contract['lambda_candidates']}
    selected=min(contract['lambda_candidates'],key=lambda v:(scores[str(v)],v))
    atomic_json(lock,dict(status='FROZEN_C_COMMON_LAMBDA_BEFORE_CONFIRMATION',selected_lambda=selected,
        validation_scores=scores,rows=rows,selection_contract_sha256=sha(cp),source_sha256=sha(__file__),
        rule=contract['selection_metric'],no_intervention_or_confirmation_results_used=True))
    formal=[]
    for pool in [0,1,2]:
        for obj in ['decoded_teacher','physical_labels']:
            for condition in ['T0','T1','T2']:
                formal.append(dict(name=f'pool{pool}_{obj}_{condition}',pool=pool,objective=obj,condition=condition,
                    **{'lambda':selected if condition=='T2' else 0.},profile_only=False))
    atomic_json(manifest,dict(phase='fixed_formal_training',lambda_lock_sha256=sha(lock),rows=formal))
    print(json.dumps(dict(selected_lambda=selected,scores=scores,formal_runs=len(formal),lock_sha256=sha(lock))))


if __name__=='__main__':main()
