"""Separate repeated numerical QP solves from deterministic fixed-insertion rollout."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json
from verifier import verify_token


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.base)/'releases/feedback-strengthening-v1/artifacts'
    old=root/'B_development_QP_v1';new=root/'interface_migration_B_QP_v1';cache=root/'B_development_cache_v1'
    reports=[json.loads((x/'report.json').read_text()) for x in [old,new]]
    designs=[json.loads((x/'design.json').read_text()) for x in [old,new]]
    assert designs[0]['assignments']==designs[1]['assignments'] and designs[0]['families']==designs[1]['families']
    for k in ['recipient_report_sha256','donor_report_sha256','head_sha256','spec_sha256']:
        assert designs[0]['binding'][k]==designs[1]['binding'][k]
    h=root/'native_readout_v1/weights.npz';assert sha(h)==designs[0]['binding']['head_sha256'];head=dict(np.load(h))
    cr=json.loads((cache/'report.json').read_text());assert sha(cache/'report.json')==designs[0]['binding']['recipient_report_sha256']
    rows=[]
    for fi,f in enumerate(designs[0]['families']):
        i=f['case'];row=cr['rows'][i];assert sha(cache/row['file'])==row['file_sha256']
        token=np.load(cache/row['file'])['free_tokens'][3,:,:384].reshape(75264)
        qs=[];validations=[]
        for directory in [old,new]:
            r=json.loads((directory/f'qp/family_{fi:04d}.json').read_text());path=directory/f'qp/family_{fi:04d}.npz'
            assert sha(path)==r['arrays_sha256'];z=dict(np.load(path));qs.append(z)
            checks=[verify_token(token,x,[head],head,expected_norm=r['matching']['effective_norm']) for x in z['corrected']]
            assert all(x['accepted'] for x in checks);validations.append(checks)
        future={}
        for key in ['actual_tokens','donor_tokens','actual_pose','donor_pose','actual_block_error','donor_block_error']:
            oldr=np.load(root/f'B_development_rollout_v1/case_{i:03d}.npz')[key]
            newr=np.load(root/f'interface_migration_B_rollout_v1/case_{i:03d}.npz')[key]
            future[key]=dict(max_absolute_difference=float(np.max(abs(oldr.astype(np.float64)-newr))),
                             identical_elements=int(np.count_nonzero(oldr==newr)),elements=int(oldr.size))
        rows.append(dict(case=i,old_and_new_full_head_region_norm_checks=validations,
            full_direction_max_absolute_difference=float(np.max(abs(qs[0]['full_standardized_directions']-qs[1]['full_standardized_directions']))),
            inserted_FP32_max_absolute_difference=float(np.max(abs(qs[0]['corrected'].astype(np.float64)-qs[1]['corrected']))),
            different_FP32_components=int(np.count_nonzero(qs[0]['corrected']!=qs[1]['corrected'])),future_differences=future))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(status='PASS_SAME_INPUTS_BOTH_NATIVE_QP_SOLVES_VALID',rows=rows,
        source_sha256=sha(__file__),old_QP_report_sha256=sha(old/'report.json'),new_QP_report_sha256=sha(new/'report.json'),
        scope='Previously exposed fixed development inputs only. QP exact-bit equality is diagnosed separately from full-head/region/norm admissibility and fixed-token rollout replay. No tolerance, model or confirmation input changed.'))
    (out/'DONE').write_text('numerical_migration_audited\n')


if __name__=='__main__':main()
