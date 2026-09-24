"""Bind every action baseline checkpoint to accepted training and a goal bank."""
import argparse,json,hashlib
from pathlib import Path
from action_comparison_manifest import specifications


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser()
for key in ['training','gate','planner-gate','bank','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();root=Path(a.training);gate=Path(a.gate);bank=Path(a.bank);out=Path(a.output)
assert not out.exists();g=json.loads(gate.read_text());assert len(g['rows'])==6 and g['no_auxiliary_checkpoints']=='bitwise identical to original trainer for both architectures and best/last'
planner_gate=Path(a.planner_gate);planner_acceptances=[]
for i in range(12):
    folder=planner_gate/f'job_{i}';af=folder/'acceptance.json';pa=json.loads(af.read_text());pr=json.loads((folder/'summary.json').read_text())
    assert pa['cases']==2 and pa['model_free_simulator_replay'] and pa['max_replay_state_difference']==0
    assert pr['route_index']==i and pr['score_space']=='latent' and 'engineering' in pr['comparison_scope']
    assert (pr['arm'],pr['mode'],pr['algorithm'])==(['transformer_jepa','gru_jepa'][i//6],['none','inverse','inverse_goal'][(i//2)%3],['random','cem'][i%2])
    assert sha(folder/'summary.json')==json.loads((folder/'artifact_manifest.json').read_text())['summary.json']
    assert pa['mode']==pr['mode'] and pr['hashes']['bank_manifest']==sha(bank/'manifest.json')
    planner_acceptances.append(dict(route=i,acceptance_sha256=sha(af),summary_sha256=sha(folder/'summary.json')))
models,routes=specifications();acceptances={};data_hashes={} 
for replica in range(3):
    f=root/f'replica_{replica}'/'acceptance.json';r=json.loads(f.read_text());assert r['updates']==21000 and len(r['rows'])==6
    assert {(v['arm'],v['mode']) for v in r['rows']}=={(arm,mode) for arm in ['transformer_jepa','gru_jepa'] for mode in ['none','inverse','inverse_goal']}
    acceptances[replica]=(r,sha(f))
for entry in models:
    folder=root/entry['training_path'];r=json.loads((folder/'summary.json').read_text());art=json.loads((folder/'artifact_manifest.json').read_text())
    assert (folder/'COMPLETE').exists() and sha(folder/'summary.json')==art['summary.json']
    assert (r['arm'],r['mode'],r['seed'],r['completed_updates'],r['batch_size'])==(entry['arm'],entry['mode'],3072,21000,128)
    acceptance,ah=acceptances[entry['replica']]
    assert next(v for v in acceptance['rows'] if (v['arm'],v['mode'])==(entry['arm'],entry['mode']))['summary_sha256']==art['summary.json']
    for f,h in json.loads((folder/'source/manifest.json').read_text()).items():assert sha(folder/'source'/f)==h
    ck=entry['checkpoint']
    for f in [f'{ck}_weights.pt',f'{ck}_auxiliary.pt']:assert sha(folder/f)==art[f]
    prior=data_hashes.setdefault(entry['replica'],r['data_manifest_sha256']);assert prior==r['data_manifest_sha256']
    entry.update(weights_sha256=art[f'{ck}_weights.pt'],auxiliary_weights_sha256=art[f'{ck}_auxiliary.pt'],training_summary_sha256=art['summary.json'],data_manifest_sha256=r['data_manifest_sha256'],acceptance_sha256=ah,selected_update=r['best_update'] if ck=='best' else 21000)
assert len(set(data_hashes.values()))==3
bm=json.loads((bank/'manifest.json').read_text());assert len(bm['cases'])==128
with out.open('x') as f:json.dump(dict(layout='action_auxiliary',planner_gate_acceptances=planner_acceptances,models=models,routes=routes,gate_sha256=sha(gate),bank_manifest_sha256=sha(bank/'manifest.json'),scope='all36 checkpoints and72 prespecified development routes; no planning-based selection; auxiliary heads unused at inference'),f,indent=2);f.write('\n')
print('FROZEN_36_CHECKPOINTS_72_ROUTES')
