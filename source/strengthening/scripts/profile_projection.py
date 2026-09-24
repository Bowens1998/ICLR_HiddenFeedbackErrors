"""CPU numerical QC with fixed eight validation windows, all A groups and dual subset."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import atomic_json, sha, require_role
from projection_shard import run_shard


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--group',type=int);a=p.parse_args();g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    source=Path(a.inputs)/f'group_{g}';r=json.loads((source/'report.json').read_text())
    assert r['status']=='PASS_A_NUMPY_METRICS_AND_DEVELOPMENT_INPUTS'
    require_role(dict(role=r['role'],parent_manifest_sha256=r['cache_report_sha256']),{'projection_numerical_development'})
    assert r['inputs_sha256']==sha(source/'inputs.npz');data=dict(np.load(source/'inputs.npz'));h=[]
    for name in ['head_A','head_B']:
        assert r['head_sha256'][name]==sha(source/(name+'.npz'));h.append(dict(np.load(source/(name+'.npz'))))
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=True);start=time.monotonic();reports={}
    for mode in (['core','dual'] if g in [0,1] else ['core']):
        families=[]
        for i in range(8):
            f=dict(tokens=[],guidance=[],head_sets=[],reference_heads=[],member_ids=[])
            for hs in ([h[:1],h] if mode=='dual' else [h[:1]]):
                for q,obj in enumerate(['latent','decoded_teacher','physical_labels']):
                    for branch in ['observed','donor']:
                        f['tokens'].append(data['predicted'][q,i]);f['guidance'].append(data[branch][i])
                        f['head_sets'].append(hs);f['reference_heads'].append(h[0]);f['member_ids'].append(f'{len(hs)}heads/{obj}/{branch}')
            families.append(f)
        reports[mode]=run_shard(out/mode,dict(input_report_sha256=sha(source/'report.json'),
            input_arrays_sha256=r['inputs_sha256'],head_sha256=r['head_sha256'],role=r['role'],group=g,mode=mode),families)
    atomic_json(out/'report.json',dict(status='PASS_REAL_HEAD_PROJECTION_NUMERICS',group=g,reports=reports,
        elapsed_seconds=time.monotonic()-start,source_sha256=sha(__file__),input_report_sha256=sha(source/'report.json'),
        scope='Numerical equality, regions and budget on first eight fixed validation windows. No future corrected rollout or donor efficacy calculation.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
