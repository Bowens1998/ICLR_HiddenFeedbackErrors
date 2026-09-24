"""Same frozen encoded head across ordinary validation endpoints and planning goals."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(pred,truth):
 distance=np.linalg.norm(pred[:,2:4]-truth[:,2:4],axis=1)
 angle=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(truth[:,4],truth[:,5]);angle=np.angle(np.exp(1j*angle))
 return dict(n=len(pred),position_mse=float(np.mean(distance**2)),angle_mse=float(np.mean(angle**2)),position_precision=float(np.mean(distance<20)),angle_precision=float(np.mean(np.abs(angle)<np.pi/9)),joint_precision=float(np.mean((distance<20)&(np.abs(angle)<np.pi/9))))


def main():
 p=argparse.ArgumentParser()
 for k in ['features','heads','selected','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();rows=[];bindings=[]
 for i in range(6):
  f=Path(a.features)/f'job_{i}';hroot=Path(a.heads)/f'job_{i}';s=Path(a.selected)/f'job_{i}'
  fr=json.loads((f/'report.json').read_text());hr=json.loads((hroot/'report.json').read_text());ha=json.loads((hroot/'acceptance.json').read_text());sr=json.loads((s/'report.json').read_text())
  assert not fr['engineering'] and not hr['engineering'] and ha['status']=='PASS'
  assert ha['report_sha256']==sha(hroot/'report.json') and hr['input_report_sha256']==sha(f/'report.json')
  row=next(v for v in hr['rows'] if v['role']=='encoded');hp=hroot/'encoded/weights.npz';assert sha(hp)==row['files_sha256']['weights.npz']
  assert sha(f/'validation_features.npz')==fr['files_sha256']['validation_features.npz']
  z=np.load(f/'validation_features.npz');head=np.load(hp)
  x=(z['encoded'].astype(float)-head['mean'])/head['scale']
  for layer in [0,2,4]:
   x=x@head[f'{layer}.weight'].astype(float).T+head[f'{layer}.bias'].astype(float)
   if layer<4:x=np.maximum(x,0)
  predicted=x*head['target_scale']+head['target_mean']
  vf=hroot/'encoded/validation_encoded.npz';assert sha(vf)==row['files_sha256']['validation_encoded.npz']
  with np.load(vf) as v:
   np.testing.assert_array_equal(v['identity'],z['identity']);np.testing.assert_allclose(v['prediction'],predicted,rtol=1e-10,atol=1e-9)
  selected=next(v for v in sr['rows'] if v['route_index']==i*8+2);sp=s/selected['file'];assert sha(sp)==selected['file_sha256'];g=np.load(sp)
  assert sr['backbone_index']==i and len(g['seed'])==128
  # Link the extracted goal decoder to this exact encoded readout via frozen plan.
  plan=json.loads(Path('runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json').read_text())
  assert sr['plan_sha256']==sha('runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json')
  entry=plan['models'][plan['routes'][i*8+2]['model_index']]
  assert entry['goal_head']['sha256']==sha(hp)
  rows.append(dict(index=i,arm=fr['arm'],validation=metrics(predicted,z['target']),planning_goals=metrics(g['estimated_goal'],g['true_goal'])))
  bindings.append(dict(index=i,head_sha256=sha(hp),feature_report_sha256=sha(f/'report.json'),selected_report_sha256=sha(s/'report.json')))
 result=dict(status='COMPLETE_POSTHOC_COMPARISON',rows=rows,bindings=bindings,source_sha256=sha(__file__),scope='Same frozen encoded head,1185 ordinary validation endpoints versus128 planning goals. Descriptive unpaired domains with different sampling; no causal attribution to distribution shift, no new labels or tuning, and overlapping validation windows are not independent episodes.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 for domain in ['validation','planning_goals']:
  print(domain,{k:float(np.mean([r[domain][k] for r in rows])) for k in rows[0][domain]})


if __name__=='__main__':main()
