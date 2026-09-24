"""Verify local fresh-bank recovery and unchanged pre-data model/route identities."""
import argparse
import copy
import json
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for key in ('bank','frozen','plan','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();bank=Path(a.bank)
    frozen=json.loads(Path(a.frozen).read_text());plan=json.loads(Path(a.plan).read_text())
    manifest=json.loads((bank/'manifest.json').read_text());ac=json.loads((bank/'acceptance.json').read_text())
    meta=plan['task_coordinate_evaluation'];fm=frozen['task_coordinate_evaluation']
    assert fm['status']=='FROZEN_MODELS_PENDING_BANK' and frozen['bank_manifest_sha256'] is None
    assert meta['status']=='READY_COMPLETE_FRESH_EVALUATION' and meta['frozen_models_sha256']==sha(a.frozen)
    assert sha(bank/'manifest.json')==plan['bank_manifest_sha256']==ac['manifest_sha256']
    assert sha(bank/'acceptance.json')==meta['bank_acceptance_sha256']
    assert ac['source_sha256']==sha(Path(__file__).with_name('accept_adaptation_contexts.py'))
    assert manifest['script_sha256']==sha(Path(__file__).with_name('prepare_prospective_pusht.py'))
    assert meta['bank_binding_source_sha256']==sha(Path(__file__).with_name('bind_task_coordinate_evaluation_bank.py'))
    assert fm['protocol_sha256']==sha(Path(__file__).resolve().parents[2]/'docs/maintrack/TASK_COORDINATE_FORMAL_PROTOCOL.md')
    assert fm['source_sha256']==sha(Path(__file__).with_name('freeze_task_coordinate_evaluation.py'))
    rebuilt=copy.deepcopy(plan);rebuilt['bank_manifest_sha256']=None;rebuilt['task_coordinate_evaluation']=copy.deepcopy(meta)
    for key in ('frozen_models_sha256','bank_acceptance_sha256','bank_binding_source_sha256'):del rebuilt['task_coordinate_evaluation'][key]
    rebuilt['task_coordinate_evaluation']['status']='FROZEN_MODELS_PENDING_BANK'
    assert rebuilt==frozen  # Nothing beyond bank binding may change, including routes and all fit bindings.
    assert len(plan['models'])==48 and len(plan['routes'])==96 and len(plan['bindings'])==36
    assert manifest['seed_start']==fm['seed_start']==1351001 and manifest['max_seeds']==fm['max_seeds']==2048
    assert ac['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and len(manifest['cases'])==len(ac['rows'])==fm['cases']==128
    seeds=[]
    for i,(c,row) in enumerate(zip(manifest['cases'],ac['rows'])):
        assert c['index']==row['index']==i and c['seed']==row['seed']
        assert row['independently_replayed_branches']==33
        assert sha(bank/f'case_{i:03d}.npz')==c['sha256']==row['sha256']
        seeds.append(c['seed'])
    rejected=[x['seed'] for x in manifest['rejected_seeds']]
    assert seeds==sorted(set(seeds)) and not set(seeds)&set(rejected)
    assert sorted(seeds+rejected)==list(range(1351001,max(seeds)+1))
    assert max(seeds)<1351001+2048
    result=dict(status='PASS128_RECOVERED_CONTEXTS_AND_UNCHANGED96_ROUTES',plan_sha256=sha(a.plan),frozen_sha256=sha(a.frozen),bank_manifest_sha256=sha(bank/'manifest.json'),bank_acceptance_sha256=sha(bank/'acceptance.json'),source_sha256=sha(__file__),accepted_cases=128,replayed_branches=4224,rejected_seeds=len(rejected),seed_range=[min(seeds),max(seeds)],scope='All locally copied case archives and accepted remote replay/source bindings verified. Complete frozen plan restored exactly after removing bank binding fields. Physics itself is not rerun by this recovery checker; rejected admissions are ledger-checked, not independently regenerated.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'])

if __name__=='__main__':main()
