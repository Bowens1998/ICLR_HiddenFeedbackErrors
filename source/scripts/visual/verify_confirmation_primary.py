"""Independently reconstruct the five primary estimands from accepted metrics."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def main():
 p=argparse.ArgumentParser()
 for k in ('runs','summary','output'):p.add_argument('--'+k,required=True)
 a=p.parse_args();r=json.loads(Path(a.summary).read_text());allrows=[]
 for g in range(6):
  d=Path(a.runs)/f'job_{g}';ac=json.loads((d/'acceptance.json').read_text());binding=r['bindings'][g];assert sha(d/'acceptance.json')==binding['acceptance_sha256'] and sha(d/'report.json')==binding['report_sha256'];assert ac['seeds']==r['seeds'];allrows.append(ac['rows'])
 # Reconstruct through explicit per-goal averaging, independent of contrast helper.
 draws=np.random.default_rng(1368001).integers(0,128,size=(20000,128));checks=[]
 for i,(c,left,right) in enumerate([(1,'fiber','free'),(2,'fiber','free'),(0,'fiber','shuffled'),(1,'fiber','shuffled'),(2,'fiber','shuffled')]):
  points=np.zeros(128);model=[]
  for g in range(6):
   l=np.asarray(allrows[g][c]['metrics'][left]['position_mse']);v=np.asarray(allrows[g][c]['metrics'][right]['position_mse']);delta=(l[:,:,4]-v[:,:,4]).mean(axis=0);points+=delta/6;model.append(delta.mean())
  bootstrap=points[draws].mean(axis=1);ci=np.quantile(bootstrap,[.005,.995]);saved=r['primary_contrasts'][i];np.testing.assert_allclose(points,saved['paired_goal_differences'],rtol=1e-12,atol=1e-9);np.testing.assert_allclose(ci,saved['primary_99_percentile_interval'],rtol=1e-12,atol=1e-9);np.testing.assert_allclose(model,saved['per_model_differences'],rtol=1e-12,atol=1e-9);np.testing.assert_allclose(points.mean(),saved['mean_difference'],rtol=1e-12,atol=1e-9);assert saved['confirmed_improvement']==bool(ci[1]<0);checks.append(dict(index=i,mean=float(points.mean()),interval99=ci.tolist(),confirmed=bool(ci[1]<0)))
 assert sum(x['confirmed'] for x in checks)==r['primary_confirmed'];result=dict(status='PASS5_INDEPENDENT_PRIMARY_ESTIMAND_RECONSTRUCTIONS',checks=checks,summary_sha256=sha(a.summary),source_sha256=sha(__file__))
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(result['status'])

if __name__=='__main__':main()
