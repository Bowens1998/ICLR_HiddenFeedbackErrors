"""Bind recovered, remotely accepted route metadata to a frozen comparison."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for key in ['runs','manifest','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();root=Path(a.runs);mp=Path(a.manifest);f=json.loads(mp.read_text());rows=[]
for i,route in enumerate(f['routes']):
    folder=root/f'job_{i}';af=folder/'acceptance.json'
    if not af.exists():continue
    r=json.loads((folder/'summary.json').read_text());acc=json.loads(af.read_text());entry=f['models'][route['model_index']]
    assert sha(folder/'summary.json')==json.loads((folder/'artifact_manifest.json').read_text())['summary.json']
    assert (r['route_index'],r['model_index'],r['arm'],r['checkpoint'],r['training_path'])==(i,route['model_index'],entry['arm'],entry['checkpoint'],entry['training_path'])
    assert (r['algorithm'],r['parameterization'])==(route['algorithm'],route['parameterization'])
    assert r['hashes']['model_manifest']==sha(mp) and r['hashes']['weights']==entry['weights_sha256']
    assert len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
    for name,h in json.loads((folder/'source/manifest.json').read_text()).items():assert sha(folder/'source'/name)==h
    rows.append(dict(route=i,summary_sha256=sha(folder/'summary.json'),acceptance_sha256=sha(af),source_manifest_sha256=sha(folder/'source/manifest.json')))
result=dict(routes=rows,expected_routes=len(f['routes']),complete=len(rows)==len(f['routes']),manifest_sha256=sha(mp),scope='recovered metadata and identities verified locally; full candidate arrays and physical replay verified remotely, not rechecked locally; incomplete matrices do not support full-factorial conclusions')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('VERIFIED_METADATA',len(rows),'OF',len(f['routes']))
