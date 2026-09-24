"""Freeze every accepted checkpoint in a predeclared comparison matrix."""
import argparse,hashlib,json
from pathlib import Path
from visual_comparison_manifest import specifications,ARMS

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while c:=f.read(8*1024*1024):h.update(c)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--output',required=True);p.add_argument('--layout',choices=['scaling','seed','engineering'],required=True);a=p.parse_args();root=Path(a.training);models,routes=specifications(a.layout)
for entry in models:
    run=root/entry['training_path'];acc=json.loads((root/entry['cell_path']/'acceptance.json').read_text());assert acc['accepted_arms']==ARMS
    assert (run/'COMPLETE').exists();r=json.loads((run/'summary.json').read_text());assert r['arm']==entry['arm']
    if a.layout=='scaling':assert r['seed']==3072 and r['completed_updates']==entry['updates']==acc['updates']
    elif a.layout=='seed':assert r['expected_epochs']==100 and len(r['epochs'])==100 and acc['epochs']==100
    else:assert r['completed_updates']==213 and acc['updates']==213
    artifacts=json.loads((run/'artifact_manifest.json').read_text());filename=entry['checkpoint']+'_weights.pt'
    for name in [filename,'summary.json']:assert digest(run/name)==artifacts[name]
    entry.update(weights_sha256=artifacts[filename],training_summary_sha256=artifacts['summary.json'],data_manifest_sha256=r['data_manifest_sha256'],seed=r['seed'],acceptance_sha256=digest(root/entry['cell_path']/'acceptance.json'))
with Path(a.output).open('x') as f:json.dump({'layout':a.layout,'models':models,'routes':routes,'selection':'complete prespecified matrix; no planning outcome used for checkpoint or seed selection'},f,indent=2);f.write('\n')
print('FROZEN',len(models),'checkpoints',len(routes),'routes')
