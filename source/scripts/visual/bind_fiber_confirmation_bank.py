"""Bind a fresh mechanism bank to the unchanged accepted model/route roster."""
import argparse,json,subprocess,sys
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('source-plan','frozen','bank','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();assert sha(a.source_plan)=='1fd34ed8b462afc5901b6d82c9ef119ff689424be03803b92882064a5559c79b'
    subprocess.run([sys.executable,str(Path(__file__).with_name('bind_task_coordinate_evaluation_bank.py')),'--frozen',a.frozen,'--expected-sha','9c7abc90d7233fcda19b6b88f9129e4d4ac24f545371c0ba04685bf54c9ad0ae','--check-only'],check=True)
    r=json.loads(Path(a.source_plan).read_text());f=json.loads(Path(a.frozen).read_text());assert r['models']==f['models'] and r['routes']==f['routes'];d=Path(a.bank);m=json.loads((d/'manifest.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
    assert m['seed_start']==1371001 and m['max_seeds']==2048 and len(m['cases'])==128 and ac['status']=='PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and ac['manifest_sha256']==sha(d/'manifest.json') and len(ac['rows'])==128
    for c,v in zip(m['cases'],ac['rows']):
        assert c['index']==v['index'] and c['seed']==v['seed'] and c['sha256']==v['sha256']==sha(d/f"case_{c['index']:03d}.npz") and v['independently_replayed_branches']==33
    r['source_plan_sha256']=sha(a.source_plan);r['bank_manifest_sha256']=sha(d/'manifest.json');r['fiber_confirmation']=dict(status='FRESH_BANK_BOUND_UNCHANGED_MODELS',seed_start=1371001,cases=128,source_sha256=sha(__file__),protocol_sha256=sha(a.protocol),bank_acceptance_sha256=sha(d/'acceptance.json'),reference_routes=[16*g+k for g in range(6) for k in (0,1,10,11)])
    with Path(a.output).open('x') as out:json.dump(r,out,indent=2);out.write('\n')
    print('BOUND128_FRESH_MECHANISM_CONTEXTS')

if __name__=='__main__':main()
