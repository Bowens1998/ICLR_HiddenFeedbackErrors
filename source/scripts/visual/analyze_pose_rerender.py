"""Accept paired renderer predictions and bootstrap original episode clusters."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def errors(pred,target):
 position=np.sum((pred[:,2:4]-target[:,2:4])**2,-1)
 angle=np.angle(np.exp(1j*(np.arctan2(pred[:,4],pred[:,5])-np.arctan2(target[:,4],target[:,5]))))
 return dict(position_mse=position,angle_mse=angle**2,joint_precision=((position<400)&(np.abs(angle)<np.pi/9)).astype(float))


def main():
 p=argparse.ArgumentParser()
 for k in ['input','features','heads','plan','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();plan=json.loads(Path(a.plan).read_text());rows=[];deltas=[];shared=None;pixel_reference=None
 for i in range(6):
  folder=Path(a.input)/f'job_{i}';r=json.loads((folder/'report.json').read_text());f=Path(a.features)/f'job_{i}';heads=Path(a.heads)/f'job_{i}'
  assert r['status']=='COMPLETE_PAIRED_RENDER' and r['index']==i and r['examples']==1185
  assert r['source_sha256']==sha(Path(__file__).with_name('evaluate_pose_rerender.py'))
  assert r['plan_sha256']==sha(a.plan) and r['feature_report_sha256']==sha(f/'report.json')
  entry=plan['models'][plan['routes'][i*8+2]['model_index']]
  assert r['weights_sha256']==entry['weights_sha256'] and r['head_sha256']==entry['goal_head']['sha256']
  assert sha(folder/'predictions.npz')==r['files_sha256']['predictions.npz'];z=np.load(folder/'predictions.npz');features=np.load(f/'validation_features.npz')
  np.testing.assert_array_equal(z['identity'],features['identity']);np.testing.assert_array_equal(z['target'],features['target'])
  hr=json.loads((heads/'report.json').read_text());hrow=next(v for v in hr['rows'] if v['role']=='encoded');hp=heads/'encoded/validation_encoded.npz'
  assert sha(hp)==hrow['files_sha256']['validation_encoded.npz']
  with np.load(hp) as original:np.testing.assert_allclose(z['original'],original['prediction'],rtol=1e-5,atol=1e-5)
  if shared is None:shared=z['identity'].copy();pixel_reference=z['pixel_mae'].copy()
  else:
   np.testing.assert_array_equal(z['identity'],shared);np.testing.assert_array_equal(z['pixel_mae'],pixel_reference)
  for key in ['original','restored']:assert z[key].shape==(1185,6) and np.isfinite(z[key]).all()
  original=errors(z['original'],z['target']);restored=errors(z['restored'],z['target']);delta={key:restored[key]-original[key] for key in original};deltas.append(delta)
  rows.append(dict(index=i,original={k:float(v.mean()) for k,v in original.items()},restored={k:float(v.mean()) for k,v in restored.items()},mean_pixel_mae=float(z['pixel_mae'].mean()),report_sha256=sha(folder/'report.json')))
 episodes=np.unique(shared[:,0]);assert len(episodes)==64
 counts=np.array([np.sum(shared[:,0]==e) for e in episodes]);draws=np.random.default_rng(1252901).integers(0,64,(10000,64));effects=[]
 for index in list(range(6))+[None]:
  delta=deltas[index] if index is not None else {k:np.mean([d[k] for d in deltas],axis=0) for k in deltas[0]}
  for metric,d in delta.items():
   sums=np.array([d[shared[:,0]==e].sum() for e in episodes]);boot=sums[draws].sum(1)/counts[draws].sum(1)
   effects.append(dict(index=index,metric=metric,mean=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
 result=dict(status='PASS_PAIRED_RENDER_DIAGNOSTIC',rows=rows,effects=effects,source_sha256=sha(__file__),plan_sha256=sha(a.plan),bootstrap=dict(seed=1252901,draws=10000,clusters=64,shared_episode_indices=True,estimand='Window-weighted mean with episode-block resampling; pooled effect equally weights six frozen heads'),scope='Original versus current-renderer images at same validation poses; no training. Conditional episode-cluster intervals; not independent model seeds, fresh-goal test, or proof that all appearance shifts are harmless.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('MEANS', {domain:{k:float(np.mean([r[domain][k] for r in rows])) for k in rows[0][domain]} for domain in ['original','restored']})
 print('POOLED', [v for v in effects if v['index'] is None])


if __name__=='__main__':main()
