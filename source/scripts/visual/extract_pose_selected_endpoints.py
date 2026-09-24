"""Hash-checked selected endpoint and goal estimates; no new simulator labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def decode(x,h):
 y=(np.asarray(x,dtype=float)-h['mean'])/h['scale']
 for i in [0,2,4]:
  y=y@h[f'{i}.weight'].astype(float).T+h[f'{i}.bias'].astype(float)
  if i<4:y=np.maximum(y,0)
 return y*h['target_scale']+h['target_mean']


def components(a,b):
 angle=np.arctan2(a[...,4],a[...,5])-np.arctan2(b[...,4],b[...,5])
 return np.stack([np.square(a[...,2:4]-b[...,2:4]).sum(-1),900*np.angle(np.exp(1j*angle))**2],-1)


def pose(state):
 return np.r_[state[:4],np.sin(state[4]),np.cos(state[4])]


def main():
 p=argparse.ArgumentParser()
 for k in ['plan','runs','bank','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
 plan=json.loads(Path(a.plan).read_text());assert plan['layout']=='pusht_nonlinear_pose'
 bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());assert sha(bank/'manifest.json')==plan['bank_manifest_sha256']
 assert len(bm['cases'])==128
 out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);rows=[]
 for offset in range(2,8):
  index=a.index*8+offset;route=plan['routes'][index];entry=plan['models'][route['model_index']]
  assert entry['score'] in ['pose_encoded','pose_predicted','state'] and entry['backbone_index']==a.index
  d=Path(a.runs)/f'job_{index}';r=json.loads((d/'summary.json').read_text());acc=json.loads((d/'acceptance.json').read_text());art=json.loads((d/'artifact_manifest.json').read_text())
  assert r['route_index']==index and len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay']
  assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['weights']==entry['weights_sha256']
  assert sha(d/'summary.json')==art['summary.json']
  heads={}
  if entry['score']!='state':
   for key in ['endpoint_head','goal_head']:
    hp=Path(entry[key]['path']);assert sha(hp)==entry[key]['sha256']
    with np.load(hp) as h:heads[key]={k:h[k] for k in h.files}
  else:
   mn=np.array(entry['target_normalization']['mean']);sd=np.array(entry['target_normalization']['std'])
  values={k:[] for k in ['estimated_endpoint','estimated_goal','true_endpoint','true_goal','seed','predicted_components','actual_components']};bindings=[]
  for row,item in zip(r['cases'],bm['cases']):
   assert row['index']==item['index'] and row['seed']==item['seed']
   name=f"case_{item['index']:03d}_predictions.npz";path=d/name;assert sha(path)==art[name]
   truthfile=bank/f"case_{item['index']:03d}.npz";assert sha(truthfile)==item['sha256']
   with np.load(path) as z,np.load(truthfile) as truth:
    t,c=row['selected_iteration'],row['selected_candidate'];token=z['tokens'][t,c].astype(float);goal=z['goal_tokens'].astype(float)
    assert float(z['costs'][t,c])==row['predicted_cost']
    ep=token*sd+mn if entry['score']=='state' else decode(token,heads['endpoint_head'])
    gp=goal*sd+mn if entry['score']=='state' else decode(goal,heads['goal_head'])
    te=pose(z['selected_states'][-1]);tg=pose(truth['goal_state'])
    pc=components(ep,gp);ac=components(te,tg)
    np.testing.assert_allclose(pc.sum(),row['predicted_cost'],rtol=5e-5 if entry['score']=='state' else 1e-10,atol=2e-3 if entry['score']=='state' else 1e-8)
    np.testing.assert_allclose(ac.sum(),row['realized_cost'],rtol=1e-9,atol=1e-7)
   for key,val in [('estimated_endpoint',ep),('estimated_goal',gp),('true_endpoint',te),('true_goal',tg),('seed',row['seed']),('predicted_components',pc),('actual_components',ac)]:values[key].append(val)
   bindings.append(dict(index=item['index'],archive_sha256=art[name],bank_case_sha256=item['sha256']))
  file=out/f'route_{index}.npz';np.savez_compressed(file,**{k:np.array(v) for k,v in values.items()})
  rows.append(dict(route_index=index,score=entry['score'],algorithm=route['algorithm'],file=file.name,file_sha256=sha(file),summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),bindings=bindings))
 result=dict(status='COMPLETE_EXTRACTION',backbone_index=a.index,plan_sha256=sha(a.plan),source_sha256=sha(__file__),rows=rows,
             scope='Selected endpoint/goal poses reconstructed from accepted hash-checked raw archives. No new simulator data, no oracle replanning, and no causal mechanism claim. Independent local component analysis required.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EXTRACTED_SIX_ROUTES',a.index)


if __name__=='__main__':main()
