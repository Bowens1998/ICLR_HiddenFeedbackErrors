"""Validate recovered source/summary/data identities without claiming local weights."""
import argparse,json,hashlib
from pathlib import Path
from visual_comparison_manifest import specifications,ARMS


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.runs);data=Path(a.data);models,_=specifications('scaling');seen=set();rows=[];cells=set()
for m in models:
    path=m['training_path']
    if path in seen:continue
    seen.add(path);folder=root/path;r=json.loads((folder/'summary.json').read_text());art=json.loads((folder/'artifact_manifest.json').read_text())
    assert sha(folder/'summary.json')==art['summary.json']
    sources=json.loads((folder/'source/manifest.json').read_text())
    for file,h in sources.items():assert sha(folder/'source'/file)==h
    dm=data/f"replica_{m['replica']}"/f"n{m['episodes']}"/'manifest.json'
    assert r['data_manifest_sha256']==sha(dm)
    assert (r['arm'],r['seed'],r['completed_updates'],r['expected_updates'],r['validation_every'])==(m['arm'],3072,m['updates'],m['updates'],210)
    assert [(e['update_start'],e['update_end']) for e in r['evaluations']]==[(i,i+210) for i in range(0,m['updates'],210)]
    assert all(e['train']['sequences']==210*128 and e['validation']['sequences']==r['clip_counts']['validation'] for e in r['evaluations'])
    assert r['best_update']==min(r['evaluations'],key=lambda e:e['validation']['loss'])['update_end']
    cell=root/m['cell_path'];acc=json.loads((cell/'acceptance.json').read_text());assert acc['accepted_arms']==ARMS and acc['updates']==m['updates']
    cells.add(m['cell_path']);rows.append(dict(path=path,replica=m['replica'],episodes=m['episodes'],updates=m['updates'],arm=m['arm'],summary_sha256=sha(folder/'summary.json'),source_manifest_sha256=sha(folder/'source/manifest.json'),training_acceptance_sha256=sha(cell/'acceptance.json'),best_weights_sha256=art['best_weights.pt'],last_weights_sha256=art['last_weights.pt']))
assert len(rows)==48 and len(cells)==12
result=dict(models=rows,accepted_cells=len(cells),scope='all48 recovered summaries/source files/data identities/budgets checked locally; checkpoint files and native inference were verified remotely, not locally downloaded; no task-performance claim')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('ACCEPTED_48_MODELS_12_CELLS_METADATA')
