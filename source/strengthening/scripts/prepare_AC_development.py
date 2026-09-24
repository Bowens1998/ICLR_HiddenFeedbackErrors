"""Run the unchanged context generator and full replay verifier on development roles."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha, atomic_json
from input_lock import add_design_arguments,input_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--role',choices=['diagnostic_development_A_C','donor_development_A_C','confirmation_A_C','donor_bank_A_C'],required=True)
    add_design_arguments(p);a=p.parse_args()
    config=ROOT/'strengthening/configs/bank_design.draft.json';s,binding=input_context(a,a.role,ROOT,__file__)
    out=Path(a.output);base=Path(a.base);sim=base/'releases/visual-v1/stable-worldmodel'
    subprocess.run([sys.executable,str(ROOT/'scripts/visual/prepare_prospective_pusht.py'),'--source',str(sim),
        '--output',str(out),'--cases',str(s['count']),'--seed-start',str(s['seed_start']),'--max-seeds',str(s['max_seeds'])],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/visual/accept_adaptation_contexts.py'),'--simulator',str(sim),
        '--bank',str(out),'--count',str(s['count']),'--seed-start',str(s['seed_start'])],check=True)
    m=json.loads((out/'manifest.json').read_text());ac=json.loads((out/'acceptance.json').read_text())
    assert len(m['cases'])==s['count'] and ac['manifest_sha256']==sha(out/'manifest.json')
    legacy=json.loads((ROOT/'strengthening/manifests/split_lineage.json').read_text());old_seeds=set()
    for info in legacy['banks'].values():
        path=Path(info['path'])/'manifest.json';assert sha(path)==info['manifest_sha256']
        old_seeds.update(x['seed'] for x in json.loads(path.read_text())['cases'])
    assert not old_seeds&{x['seed'] for x in m['cases']}
    atomic_json(out/'role.json',dict(**binding,role=a.role,parent_manifest_sha256=sha(out/'manifest.json'),
        acceptance_sha256=sha(out/'acceptance.json'),design_sha256=sha(config),count=s['count'],
        seed_start=s['seed_start'],no_legacy_seed_overlap=True,source_sha256=sha(__file__)))
    (out/'DONE').write_text('input_bank_verified\n')


if __name__=='__main__':main()
