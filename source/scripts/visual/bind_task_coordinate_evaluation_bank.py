"""Check frozen model identities, then bind only an independently accepted bank."""
import argparse,json
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--bank');p.add_argument('--output');a=p.parse_args()
    assert sha(a.frozen)==a.expected_sha;plan=json.loads(Path(a.frozen).read_text());meta=plan['task_coordinate_evaluation']
    assert meta['status']=='FROZEN_MODELS_PENDING_BANK' and plan['bank_manifest_sha256'] is None
    assert meta['seed_start']==1351001 and meta['cases']==128 and meta['max_seeds']==2048
    assert len(plan['models'])==48 and len(plan['routes'])==96 and len(plan['bindings'])==36
    for i,e in enumerate(plan['models']):
        group, offset = divmod(i, 8)
        conditions=('original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels','original','clipped_state','unit_state')
        assert e['original_model_index']==2*group+(offset>=5) and e['adaptation_condition']==conditions[offset] and e['comparison_group']==group
        d=Path(e['training_path']);assert sha(d/'summary.json')==e['training_summary_sha256'] and sha(d/'last_weights.pt')==e['weights_sha256']
        for name in ('endpoint_head','goal_head'):
            if e.get(name):assert sha(e[name]['path'])==e[name]['sha256']
        if offset not in (0,5):
            s=json.loads((d/'summary.json').read_text());fp=Path(s['fit_report']);ac=fp.parent/'acceptance.json';r=json.loads(fp.read_text());acceptance=json.loads(ac.read_text())
            b=next(b for b in plan['bindings'] if b['model_index']==i)
            assert b['condition']==e['adaptation_condition']==s['condition']
            assert sha(d/'summary.json')==b['adapter_summary_sha256']
            assert sha(fp)==s['fit_report_sha256']==b['fit_report_sha256'] and sha(ac)==s['fit_acceptance_sha256']==b['fit_acceptance_sha256']
            status='PASS_FORMAL_FIT_FROZEN_STATE_AND_CPU_PREDICTIONS' if offset in (1,6) else 'PASS_TASK_COORDINATE_FORMAL_FIT_AND_CPU_PREDICTIONS'
            assert acceptance['status']==status and acceptance['report_sha256']==sha(fp)
            assert r['weights_sha256']==e['weights_sha256'] and r['updates']==2100
            if offset not in (1,6):
                assert r['protocol_sha256']==meta['protocol_sha256']
                assert r['index']==4*group+{2:0,3:1,4:2,7:3}[offset]
        for j,algorithm in enumerate(('random','cem')):
            route=plan['routes'][2*i+j];assert route['model_index']==i and route['algorithm']==algorithm and route['parameterization']=='full'
    if a.check_only:print('PASS_FROZEN48_MODELS96_ROUTES');return
    assert a.bank and a.output;bank=Path(a.bank);m=json.loads((bank/'manifest.json').read_text());ac=json.loads((bank/'acceptance.json').read_text())
    assert m['seed_start']==1351001 and m['max_seeds']==2048 and len(m['cases'])==128
    assert ac['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and ac['manifest_sha256']==sha(bank/'manifest.json') and len(ac['rows'])==128
    for c,v in zip(m['cases'],ac['rows']):
        assert c['index']==v['index'] and c['seed']==v['seed'] and v['independently_replayed_branches']==33
        assert c['sha256']==v['sha256']==sha(bank/f"case_{c['index']:03d}.npz")
    plan['bank_manifest_sha256']=sha(bank/'manifest.json');meta['status']='READY_COMPLETE_FRESH_EVALUATION';meta['frozen_models_sha256']=a.expected_sha;meta['bank_acceptance_sha256']=sha(bank/'acceptance.json');meta['bank_binding_source_sha256']=sha(__file__)
    with Path(a.output).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('BOUND128_CONTEXTS96_UNCHANGED_ROUTES')

if __name__=='__main__':main()
