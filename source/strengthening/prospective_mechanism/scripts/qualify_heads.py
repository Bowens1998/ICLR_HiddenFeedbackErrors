"""Qualify all four frozen C/D heads; any failed gate blocks formal S1 without refit."""
import argparse
from pathlib import Path
import time
import numpy as np
from s1_common import (atomic_json,atomic_npz,checked_json,load_protocol,load_view,sha,source_hashes)
from s1_readout import numpy_forward,relu_forward,pose_errors,qualification_gate,_V8_PATH


def check_selected_lock(lock,protocol_sha):
    if lock.get('status')!='ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION' or lock.get('protocol_sha256')!=protocol_sha:
        raise ValueError('A complete externally frozen four-head lock is required')
    rows=lock.get('heads',[])
    if len(rows)!=4 or {(r['group'],r['head_role']) for r in rows}!={(g,h) for g in [0,1] for h in ['C','D']}:
        raise ValueError('Missing, duplicate or changed head roster')
    return rows


def main():
    p=argparse.ArgumentParser()
    for k in ['protocol','protocol-sha256','selected-lock','selected-lock-sha256','a-bindings','a-bindings-sha256','output']:
        p.add_argument('--'+k,required=True)
    a=p.parse_args();cfg=load_protocol(a.protocol,a.protocol_sha256)
    lock=checked_json(a.selected_lock,a.selected_lock_sha256)
    rows=check_selected_lock(lock,a.protocol_sha256)
    old=checked_json(a.a_bindings,a.a_bindings_sha256)
    references={r['group']:r for r in old['groups']}
    # Authenticate the whole selected roster before opening any qualification arrays.
    for row in rows:
        report=checked_json(row['fit_report'],row['fit_report_sha256'])
        if report.get('status')!='PASS_S1_FIXED_HEAD_FIT' or report.get('protocol_sha256')!=a.protocol_sha256 or report.get('group')!=row['group'] or report.get('head_role')!=row['head_role']:
            raise ValueError('Selected fit binding mismatch')
        if sha(row['checkpoint'])!=row['checkpoint_sha256'] or sha(row['forward_fixture'])!=row['forward_fixture_sha256']:
            raise ValueError('Selected checkpoint or fixture changed')
        if report['selected']['sha256']!=row['checkpoint_sha256'] or report['views_report_sha256']!=row['views_report_sha256'] or report['views']!=row['views']:
            raise ValueError('Fit-to-view binding mismatch')
        ref=references[row['group']]['head_A']
        if sha(ref['path'])!=ref['sha256']: raise ValueError('Frozen g_A changed')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();results=[]
    for row in sorted(rows,key=lambda r:(r['group'],r['head_role'])):
        g,h=row['group'],row['head_role']
        with np.load(row['checkpoint'],allow_pickle=False) as z:head=dict(z)
        ref=references[g]['head_A']
        with np.load(ref['path'],allow_pickle=False) as z:reference=dict(z)
        report=checked_json(Path(row['views'])/'report.json',row['views_report_sha256'])
        if report.get('status')!='PASS_S1_DISJOINT_PARENT_VIEWS' or report.get('protocol_sha256')!=a.protocol_sha256:
            raise ValueError('Qualification view report mismatch')
        scores={};ref_scores={};files={}
        for domain in ['expert','planner']:
            data=load_view(row['views'],report,head_role=h,old_role='qualification',group=g,domain=domain,
                protocol_sha=a.protocol_sha256,allowed_roles={'qualification'})
            pred=numpy_forward(data['observed'],head,h)
            baseline=relu_forward(reference,data['observed'],physical=True)
            metrics=pose_errors(pred,data['labels'],reference['target_scale'])
            old_metrics=pose_errors(baseline,data['labels'],reference['target_scale'])
            scores[domain]={k:float(v.mean()) for k,v in metrics.items()}
            ref_scores[domain]={k:float(v.mean()) for k,v in old_metrics.items()}
            target=out/f'group_{g}'/h/(domain+'.npz')
            atomic_npz(target,prediction=pred,reference_prediction=baseline,truth=data['labels'],
                parent_ids=data['parent_ids'],pixel_sha256=data['pixel_sha256'],
                **{'head_'+k:v for k,v in metrics.items()},**{'reference_'+k:v for k,v in old_metrics.items()})
            files[domain]=dict(file=str(target.relative_to(out)),sha256=sha(target),rows=len(pred),parents=len(np.unique(data['parent_ids'])))
        gate=qualification_gate(scores,ref_scores,cfg['qualification'])
        results.append(dict(group=g,head_role=h,checkpoint_sha256=row['checkpoint_sha256'],reference_head_sha256=ref['sha256'],
            scores=scores,reference_scores=ref_scores,gate=gate,files=files))
    passed=all(r['gate']['passed'] for r in results)
    atomic_json(out/'report.json',dict(status='COMPLETE_S1_QUALIFICATION_PASS' if passed else 'COMPLETE_S1_QUALIFICATION_BLOCKED_GATES',
        protocol_sha256=a.protocol_sha256,selected_lock_sha256=a.selected_lock_sha256,a_bindings_sha256=a.a_bindings_sha256,
        all_four_completed=True,all_gates_passed=passed,formal_qualification_gate_passed=passed,rows=results,
        failure_policy='Any failed head/metric/domain stops formal S1; all four retained; no refit, dropout or sample extension',
        source_sha256=sha(__file__),adapter_source_sha256=source_hashes(),reused_gelu_source_sha256=sha(_V8_PATH),
        elapsed_seconds=time.monotonic()-started,
        scope='Reused observed qualification roles only; no intervention token, future effect or formal scientific result accessed.'))
    (out/'DONE').write_text('complete_all_four_qualification_regardless_of_gate\n')


if __name__=='__main__':main()
