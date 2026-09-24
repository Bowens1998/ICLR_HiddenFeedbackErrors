"""Check frozen model identities, then bind only an independently accepted bank."""
import argparse,json
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--bank');p.add_argument('--output');a=p.parse_args()
    assert sha(a.frozen)==a.expected_sha;plan=json.loads(Path(a.frozen).read_text());meta=plan['adaptation_evaluation']
    assert meta['status']=='FROZEN_MODELS_PENDING_BANK' and plan['bank_manifest_sha256'] is None
    assert meta['seed_start']==1321001 and meta['cases']==128 and meta['max_seeds']==2048
    assert len(plan['models'])==36 and len(plan['routes'])==72 and len(meta['fit_bindings'])==24
    for i,e in enumerate(plan['models']):
        assert e['original_model_index']==i//3 and e['adaptation_condition']==('original','expert','planner')[i%3]
        d=Path(e['training_path']);assert sha(d/'summary.json')==e['training_summary_sha256'] and sha(d/'last_weights.pt')==e['weights_sha256']
        for name in ('endpoint_head','goal_head'):
            if e.get(name):assert sha(e[name]['path'])==e[name]['sha256']
        if i%3:
            s=json.loads((d/'summary.json').read_text());fp=Path(s['fit_report']);ac=fp.parent/'acceptance.json';r=json.loads(fp.read_text());acceptance=json.loads(ac.read_text());task=2*(i//3)+(i%3==2)
            b=next(b for b in meta['fit_bindings'] if b['task']==task)
            assert sha(fp)==s['fit_report_sha256']==b['fit_report_sha256'] and sha(ac)==s['fit_acceptance_sha256']==b['fit_acceptance_sha256']
            assert acceptance['status']=='PASS_FORMAL_FIT_FROZEN_STATE_AND_CPU_PREDICTIONS' and acceptance['report_sha256']==sha(fp)
            assert r['weights_sha256']==e['weights_sha256'] and r['updates']==2100
        for j,algorithm in enumerate(('random','cem')):
            route=plan['routes'][2*i+j];assert route['model_index']==i and route['algorithm']==algorithm and route['parameterization']=='full'
    if a.check_only:print('PASS_FROZEN36_MODELS72_ROUTES');return
    assert a.bank and a.output;bank=Path(a.bank);m=json.loads((bank/'manifest.json').read_text());ac=json.loads((bank/'acceptance.json').read_text())
    assert m['seed_start']==1321001 and m['max_seeds']==2048 and len(m['cases'])==128
    assert ac['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and ac['manifest_sha256']==sha(bank/'manifest.json') and len(ac['rows'])==128
    for c,v in zip(m['cases'],ac['rows']):
        assert c['index']==v['index'] and c['seed']==v['seed'] and v['independently_replayed_branches']==33
        assert c['sha256']==v['sha256']==sha(bank/f"case_{c['index']:03d}.npz")
    plan['bank_manifest_sha256']=sha(bank/'manifest.json');meta['status']='READY_COMPLETE_FRESH_EVALUATION';meta['frozen_models_sha256']=a.expected_sha;meta['bank_acceptance_sha256']=sha(bank/'acceptance.json');meta['bank_binding_source_sha256']=sha(__file__)
    with Path(a.output).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('BOUND128_CONTEXTS72_UNCHANGED_ROUTES')

if __name__=='__main__':main()
