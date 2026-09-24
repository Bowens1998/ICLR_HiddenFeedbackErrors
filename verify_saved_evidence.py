"""Portable entry point for the retained CPU scientific verifiers."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
TASKS={
 'legacy':('feedback_diagnostic_evidence.zip',['verify.py']),
 'core':('feedback_confirmation_evidence.zip',['verify_statistics.py','--root','.','--output','REPLAY_STATISTICS.json']),
 'decision':('feedback_confirmation_evidence.zip',['verify_decisions.py']),
 'velocity_qualification':('velocity_readout_qualification.zip',['verify_qualification.py','--artifacts','artifacts','--output','REPLAY_QUALIFICATION.json']),
 'motion':('feedback_motion_mechanism.zip',['run_verification.py']),
 'reporting':('feedback_reporting_supplement.zip',['strengthening/posthoc_analyses_20260922/supplement_v1/verify.py']),
 'second_readout':('feedback_second_readout_scoring.zip',['strengthening/posthoc_analyses_20260922/second_readout_v1/verify.py']),
 'independent_readout':('independent_evaluation_readout.zip',['verify.py','--output','REPLAY_EVALUATION_READOUT']),
 'pointmaze':('pointmaze_confirmation.zip',['verify.py','--output','REPLAY_POINTMAZE']),
}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--package',choices=['all',*TASKS],default='all')
 parser.add_argument('--work-dir',type=Path,default=Path('replay'))
 parser.add_argument('--list',action='store_true')
 args=parser.parse_args()
 if args.list:
  for key,(archive,command) in TASKS.items(): print(key,':',archive,'→','python '+' '.join(command))
  return
 packages=TASKS if args.package=='all' else {args.package:TASKS[args.package]}
 manifest={r['supplied_archive']:r for r in json.loads((ROOT/'ARCHIVE_MANIFEST.json').read_text())}
 env=os.environ.copy();env.pop('PYTHONPATH',None)
 env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 args.work_dir.mkdir(parents=True,exist_ok=True)
 extracted_this_run=set()
 for name,(archive,command) in packages.items():
  source=ROOT/'archives'/archive
  assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest[archive]['sha256'],archive
  destination=args.work_dir/Path(archive).stem
  if destination.exists() and destination not in extracted_this_run:
   raise SystemExit('Use a new --work-dir; existing extraction is not trusted as a fresh replay.')
  if not destination.exists():
   destination.mkdir()
   with ZipFile(source) as z:
    assert all(not p.startswith('/') and '..' not in Path(p).parts for p in z.namelist())
    z.extractall(destination)
   extracted_this_run.add(destination)
  receipt=args.work_dir/(name+'_runner_receipt.json')
  if receipt.exists():raise SystemExit('Use a new --work-dir to retain the previous replay receipt.')
  before=time.monotonic()
  run=subprocess.run([sys.executable,'-B',*command],cwd=destination,env=env)
  record=dict(package=name,archive=archive,archive_sha256=manifest[archive]['sha256'],
              command=['python','-B',*command],exit_code=run.returncode,wall_seconds=time.monotonic()-before,
              scope='Saved-evidence reconstruction; no model training, image encoding, simulator replay or new constrained optimization.')
  receipt.write_text(json.dumps(record,indent=2)+'\n')
  if run.returncode:raise SystemExit(run.returncode)

if __name__=='__main__':main()
