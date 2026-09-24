"""Freeze six accepted short-run models for interface validation only."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
for key in ['training','acceptance','bank','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();root=Path(a.training);gate=Path(a.acceptance);bank=Path(a.bank)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
g=json.loads(gate.read_text());assert len(g['rows'])==6
assert g['no_auxiliary_checkpoints']=='bitwise identical to original trainer for both architectures and best/last'
models=[];routes=[]
for arm in ['transformer_jepa','gru_jepa']:
 for mode in ['none','inverse','inverse_goal']:
  path=f'{arm}/{mode}';folder=root/path;r=json.loads((folder/'summary.json').read_text());art=json.loads((folder/'artifact_manifest.json').read_text())
  assert r['completed_updates']==213 and r['mode']==mode and r['arm']==arm
  assert sha(folder/'summary.json')==next(v['summary_sha256'] for v in g['rows'] if v['arm']==arm and v['mode']==mode)
  assert sha(folder/'best_weights.pt')==art['best_weights.pt']
  i=len(models);models.append(dict(arm=arm,mode=mode,checkpoint='best',training_path=path,weights_sha256=art['best_weights.pt'],training_summary_sha256=art['summary.json'],data_manifest_sha256=r['data_manifest_sha256'],updates=213,seed=3072))
  for algorithm in ['random','cem']:routes.append(dict(model_index=i,algorithm=algorithm,parameterization='full'))
result=dict(layout='action_auxiliary',models=models,routes=routes,bank_manifest_sha256=sha(bank/'manifest.json'),gate_sha256=sha(gate),scope='engineering213-update models; first two already-examined development goals only; not scientific method performance')
with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
