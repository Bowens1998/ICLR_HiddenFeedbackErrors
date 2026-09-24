"""Exact train-endpoint pose proximity, with fixed physical task thresholds."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def coverage(query,train):
 tangle=np.arctan2(train[:,4],train[:,5]);qangle=np.arctan2(query[:,4],query[:,5])
 minimum=[];full_minimum=[];covered=[];full_covered=[]
 for start in range(0,len(query),64):
  q=query[start:start+64]
  block=np.sum((q[:,None,2:4]-train[None,:,2:4])**2,-1)/400
  agent=np.sum((q[:,None,:2]-train[None,:,:2])**2,-1)/400
  angle=np.angle(np.exp(1j*(qangle[start:start+64,None]-tangle[None,:])))**2/(np.pi/9)**2
  minimum.extend(np.min(block+angle,axis=1));full_minimum.extend(np.min(block+agent+angle,axis=1))
  covered.extend(np.any((block<1)&(angle<1),axis=1));full_covered.extend(np.any((block<1)&(agent<1)&(angle<1),axis=1))
 return dict(n=len(query),block_pose_nearest_quantiles=np.quantile(minimum,[0,.25,.5,.75,.9,1]).tolist(),agent_block_pose_nearest_quantiles=np.quantile(full_minimum,[0,.25,.5,.75,.9,1]).tolist(),fraction_with_task_close_train_endpoint=float(np.mean(covered)),fraction_with_agent_and_task_close_train_endpoint=float(np.mean(full_covered)))


def main():
 p=argparse.ArgumentParser()
 for k in ['features','bank','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());goals=[]
 for item in bm['cases']:
  f=bank/f"case_{item['index']:03d}.npz";assert sha(f)==item['sha256']
  with np.load(f) as z:
   s=z['goal_state'];goals.append(np.r_[s[:4],np.sin(s[4]),np.cos(s[4])])
 goals=np.array(goals);assert goals.shape==(128,6);rows=[]
 for rep in range(3):
  folder=Path(a.features)/f'job_{2*rep}';r=json.loads((folder/'report.json').read_text());data={}
  for split in ['train','validation']:
   f=folder/f'{split}_features.npz';assert sha(f)==r['files_sha256'][f.name]
   with np.load(f) as z:data[split]=z['target']
  other=Path(a.features)/f'job_{2*rep+1}'
  for split in data:
   with np.load(other/f'{split}_features.npz') as z:np.testing.assert_array_equal(z['target'],data[split])
  rows.append(dict(replica=rep,training_endpoints=len(data['train']),validation=coverage(data['validation'],data['train']),planning_goals=coverage(goals,data['train']),feature_report_sha256=sha(folder/'report.json')))
 result=dict(status='COMPLETE_POSTHOC_COVERAGE',rows=rows,bank_manifest_sha256=sha(bank/'manifest.json'),source_sha256=sha(__file__),
             metric='Squared block distance /20^2 + wrapped angle squared /(pi/9)^2; full-pose alternative adds squared agent distance /20^2. Exact minimum over all fitted train endpoints. Close coverage requires each component below its threshold.',
             quantile_levels=[0,.25,.5,.75,.9,1],scope='Descriptive finite-sample physical proximity; not support density, image-domain equivalence, causal proof, or a deployment metric (uses privileged states). No threshold tuning or new training.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 for row in rows:print(json.dumps(row))


if __name__=='__main__':main()
