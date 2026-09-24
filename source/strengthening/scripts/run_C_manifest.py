"""One Slurm array row maps to one fixed C development or formal training spec."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import atomic_json, sha


def main():
    p=argparse.ArgumentParser()
    for key in ['manifest','base','heads','cache','A-runs','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int);a=p.parse_args()
    path=Path(a.manifest);path=path if path.is_absolute() else ROOT/path
    spec=json.loads(path.read_text());idx=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index
    row=spec['rows'][idx]
    if spec['phase'] not in ['development_profile','development_lambda_selection','fixed_formal_training']:raise ValueError('Unknown stage')
    if spec['phase']=='fixed_formal_training':
        lock=ROOT/'strengthening/configs/C_lambda.lock.json'
        binding=json.loads(lock.read_text())
        assert spec['lambda_lock_sha256']==sha(lock)
        if row['condition']=='T2':assert row['lambda']==binding['selected_lambda']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    launch=out/(row['name']+'_launch.json')
    if launch.exists():raise ValueError('Launch already recorded; inspect job before retry')
    atomic_json(launch,dict(manifest_sha256=sha(path),index=idx,row=row,source_sha256=sha(__file__)))
    args=[sys.executable,str(Path(__file__).with_name('train_rolling.py'))]
    for k in ['base','heads','cache','A-runs']:args+=['--'+k,getattr(a,k.replace('-','_'))]
    args+=['--output',str(out/row['name']),'--pool',str(row['pool']),'--objective',row['objective'],
           '--condition',row['condition'],'--anchor-lambda',str(row['lambda'])]
    if row['profile_only']:args+=['--profile-only']
    subprocess.run(args,check=True)


if __name__=='__main__':main()
