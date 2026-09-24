"""Post-hoc selected-plan physical-score residuals; not causal calibration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['runs','summary','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();full=json.loads(Path(a.summary).read_text());assert full['status']=='COMPLETE_DEVELOPMENT' and len(full['rows'])==48
 ix=np.random.default_rng(1251901).integers(0,128,(10000,128));rows=[];contrasts=[];arrays={}
 def interval(d):
  return dict(mean=float(d.mean()),ci95=np.quantile(d[ix].mean(-1),[.025,.975]).tolist())
 for binding in full['bindings']:
  i=binding['index'];d=Path(a.runs)/f'job_{i}'
  assert sha(d/'summary.json')==binding['summary_sha256'] and sha(d/'acceptance.json')==binding['acceptance_sha256']
  r=json.loads((d/'summary.json').read_text())
  if r['score_space']=='latent':continue # latent squared distance is not a physical cost
  assert [v['seed'] for v in r['cases']]==full['goal_seeds']
  predicted=np.array([v['predicted_cost'] for v in r['cases']]);actual=np.array([v['realized_cost'] for v in r['cases']])
  assert np.isfinite(predicted).all() and np.isfinite(actual).all()
  np.testing.assert_array_equal(actual,full['rows'][i]['cost'])
  arrays[i]=dict(predicted=predicted,actual=actual,residual=actual-predicted)
  rows.append(dict(route_index=i,score=r['score_space'],algorithm=r['algorithm'],mean_predicted=float(predicted.mean()),mean_actual=float(actual.mean()),mean_residual=float((actual-predicted).mean())))
 for score,offset in [('pose_encoded',2),('pose_predicted',4),('state',6)]:
  pooled={key:[] for key in ['predicted','actual','residual']}
  for backbone in range(6):
   random=backbone*8+offset;cem=random+1
   delta={key:arrays[cem][key]-arrays[random][key] for key in pooled}
   np.testing.assert_allclose(delta['residual'],delta['actual']-delta['predicted'],rtol=1e-12,atol=1e-9)
   for key,d in delta.items():
    pooled[key].append(d);contrasts.append(dict(score=score,backbone=backbone,metric=key,**interval(d)))
  for key,ds in pooled.items():contrasts.append(dict(score=score,backbone=None,metric=key,**interval(np.mean(ds,axis=0))))
 result=dict(status='COMPLETE_POSTHOC_DIAGNOSTIC',rows=rows,contrasts=contrasts,summary_sha256=sha(a.summary),source_sha256=sha(__file__),
             bootstrap=dict(seed=1251901,draws=10000,shared_goals=True),
             scope='CEM-minus-random differences at separately selected plans. Physical-unit scorers only. Residual=actual-predicted; increased residual is not by itself miscalibration causation or a same-action counterfactual. Existing development goals; nominal conditional intervals, no multiplicity adjustment or training-population claim.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 for v in contrasts:
  if v['backbone'] is None:print(v)


if __name__=='__main__':main()
