"""Separate selected endpoint and goal error from recovered PushT poses."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def error(a,b):
 pos=np.sum((a[:,:][...,2:4]-b[...,2:4])**2,axis=-1)
 delta=np.arctan2(a[...,4],a[...,5])-np.arctan2(b[...,4],b[...,5])
 angle=(delta+np.pi)%(2*np.pi)-np.pi
 return np.column_stack([pos,900*angle**2]),(pos<400)&(np.abs(angle)<np.pi/9)


def main():
 p=argparse.ArgumentParser()
 for k in ['input','summary','plan','bank','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();full=json.loads(Path(a.summary).read_text());plan=json.loads(Path(a.plan).read_text());assert full['status']=='COMPLETE_DEVELOPMENT' and full['plan_sha256']==sha(a.plan)
 rows=[];sources=[]
 for index in range(6):
  folder=Path(a.input)/f'job_{index}';report=json.loads((folder/'report.json').read_text())
  assert report['status']=='COMPLETE_EXTRACTION' and report['backbone_index']==index and report['plan_sha256']==sha(a.plan)
  assert report['source_sha256']==sha(Path(__file__).with_name('extract_pose_selected_endpoints.py'))
  assert [r['route_index'] for r in report['rows']]==list(range(index*8+2,index*8+8))
  shared_goal={}
  for row in report['rows']:
   route=row['route_index'];binding=full['bindings'][route]
   assert row['summary_sha256']==binding['summary_sha256'] and row['acceptance_sha256']==binding['acceptance_sha256']
   file=folder/row['file'];assert sha(file)==row['file_sha256'];z=np.load(file)
   np.testing.assert_array_equal(z['seed'],full['goal_seeds'])
   for key in ['estimated_endpoint','estimated_goal','true_endpoint','true_goal']:assert z[key].shape==(128,6) and np.isfinite(z[key]).all()
   for i,b in enumerate(row['bindings']):
    path=Path(a.bank)/f"case_{b['index']:03d}.npz";assert sha(path)==b['bank_case_sha256']
    with np.load(path) as truth:
     raw=truth['goal_state'];np.testing.assert_array_equal(z['true_goal'][i],np.r_[raw[:4],np.sin(raw[4]),np.cos(raw[4])])
   # Goal encoding/decoding is independent of planner for a given interface.
   if row['algorithm']=='random':shared_goal[row['score']]=z['estimated_goal']
   else:np.testing.assert_array_equal(z['estimated_goal'],shared_goal[row['score']])
   predicted,_=error(z['estimated_endpoint'],z['estimated_goal']);actual,_=error(z['true_endpoint'],z['true_goal'])
   endpoint,endpoint_ok=error(z['estimated_endpoint'],z['true_endpoint']);goal,goal_ok=error(z['estimated_goal'],z['true_goal'])
   np.testing.assert_allclose(predicted,z['predicted_components'],rtol=1e-10,atol=1e-8)
   np.testing.assert_allclose(actual,z['actual_components'],rtol=1e-10,atol=1e-8)
   np.testing.assert_allclose(actual.sum(-1),full['rows'][route]['cost'],rtol=1e-9,atol=1e-7)
   rows.append(dict(route_index=route,score=row['score'],algorithm=row['algorithm'],endpoint_component_mse=endpoint.mean(0).tolist(),goal_component_mse=goal.mean(0).tolist(),endpoint_precision=float(endpoint_ok.mean()),goal_precision=float(goal_ok.mean()),predicted_components=predicted.mean(0).tolist(),actual_components=actual.mean(0).tolist()))
  sources.append(dict(index=index,report_sha256=sha(folder/'report.json')))
 result=dict(status='COMPLETE_POSTHOC_DIAGNOSTIC',rows=rows,sources=sources,summary_sha256=sha(a.summary),source_sha256=sha(__file__),scope='Position squared error and900*wrapped-angle squared error separately; precision uses20pixel andpi/9 thresholds. Same selected trajectories, no oracle replanning or proof of irrecoverable representation loss. Goal truth independently rebound to local bank; endpoint truth inherits remote accepted physics replay.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('COMPLETE36_ROUTES_ENDPOINT_GOAL_DIAGNOSTIC')


if __name__=='__main__':main()
