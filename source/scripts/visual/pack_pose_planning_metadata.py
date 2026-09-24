"""Recover complete accepted PushT metadata; raw traces stay in the release."""
import argparse,hashlib,json,tarfile
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.release);files=[];rows=[]
 for stage,n in [('planner_engineering',2),('planner_development',128)]:
  for i in range(48):
   d=root/'runs'/stage/f'job_{i}'
   assert (d/'COMPLETE').exists()
   r=json.loads((d/'summary.json').read_text());acc=json.loads((d/'acceptance.json').read_text());art=json.loads((d/'artifact_manifest.json').read_text())
   assert r['route_index']==i and len(r['cases'])==acc['cases']==n and acc['model_free_simulator_replay']
   assert acc['verifier_sha256']==sha(root/'scripts/visual/accept_pose_planner.py')
   assert art['summary.json']==sha(d/'summary.json')
   for f in ['summary.json','acceptance.json','artifact_manifest.json','COMPLETE','source/manifest.json']:files.append(d/f)
   rows.append(dict(stage=stage,index=i,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json')))
 for f in ['planning_plan.json','engineering_matrix_acceptance.json']:files.append(root/f)
 with tarfile.open(a.output,'x:gz') as t:
  for f in files:t.add(f,arcname=str(f.relative_to(root)),recursive=False)
 print(json.dumps(dict(status='PASS',metadata_files=len(files),archive_sha256=sha(a.output),rows=rows,scope='Complete accepted metadata only; archived raw candidate traces remain remote and were verified by each original acceptance.')))


if __name__=='__main__':main()
