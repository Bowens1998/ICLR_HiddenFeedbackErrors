"""Native visual-only corrected rollout with fresh full-head and channel verification."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import sha,atomic_json,require_role
from artifact_io import atomic_npz,state_hash
from native_model import load_native
from native_rollout import prepare_native_inputs,native_feedback_rollout
from verifier import verify_token
from nonlinear_pose_cost import numpy_pose
from evaluation_lock import (add_protocol_arguments,evaluation_context,partition,
    check_evaluation_parent,require_output)


def main():
    p=argparse.ArgumentParser()
    for k in ['assets','environment','bank','cache','head','projection','output']:p.add_argument('--'+k,required=True)
    add_protocol_arguments(p,partitioned=True);a=p.parse_args();bank=Path(a.bank);cache=Path(a.cache);qp=Path(a.projection);out=Path(a.output)
    formal=bool(a.protocol_lock or a.protocol_sha256);role='confirmation_B' if formal else 'diagnostic_development_B'
    ctx=evaluation_context(a,role,ROOT,__file__);g,shard,cases=partition(a,ctx,'B')
    for key,path in [('cache',a.cache),('QP',a.projection),('rollout',a.output)]:require_output(path,ctx,'B_'+key)
    if shard is not None:qp=qp/f'shard_{shard:03d}';out=out/f'shard_{shard:03d}'
    cr=json.loads((cache/'report.json').read_text());require_role(cr,{role});check_evaluation_parent(cr,ctx)
    bm=json.loads((bank/'manifest.json').read_text());assert cr['parent_manifest_sha256']==sha(bank/'manifest.json')
    pr=json.loads((qp/'report.json').read_text());design=json.loads((qp/'design.json').read_text())
    assert pr['status'] in ['PASS_CLOSEDLOOP_DEVELOPMENT_QP','PASS_CLOSEDLOOP_QP'] and (qp/'DONE').exists() and (qp/'qp/DONE').exists()
    if formal:
        check_evaluation_parent(pr,ctx);assert pr['cases']==cases and pr['shard']==shard
        assert [f['case'] for f in design['families']]==cases
    assert sha(qp/'design.json')==pr['design_sha256'] and sha(qp/'qp/report.json')==pr['qp_report_sha256']
    assert design['binding']['recipient_report_sha256']==sha(cache/'report.json') and sha(a.head)==cr['head_sha256']
    head=dict(np.load(a.head));model,cfg,mb=load_native(a.assets,a.environment);before=state_hash(model)
    assert mb==cr['model_binding'];torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.inference_mode():
        for fi,f in enumerate(design['families']):
            i=f['case'];c=bm['cases'][i];source=bank/c['file'];saved=cache/cr['rows'][i]['file']
            assert sha(source)==c['sha256'] and sha(saved)==cr['rows'][i]['file_sha256']
            raw=dict(np.load(source));baseline=dict(np.load(saved));history,blocks,_=prepare_native_inputs(raw,cfg['img_size'])
            free=native_feedback_rollout(model,history,blocks);np.testing.assert_array_equal(free[0].cpu().numpy(),baseline['free_tokens'])
            qpr=json.loads((qp/f'qp/family_{fi:04d}.json').read_text());arrays=qp/f'qp/family_{fi:04d}.npz'
            assert sha(arrays)==qpr['arrays_sha256'] and qpr['member_ids']==f['member_ids']
            corrections=np.load(arrays)['corrected'];results={};checks={}
            for j,branch in enumerate(['actual','donor']):
                check=verify_token(baseline['free_tokens'][3,:,:384].reshape(75264),corrections[j],[head],head,
                    expected_norm=qpr['matching']['effective_norm']);assert check['accepted'];checks[branch]=check
                value=native_feedback_rollout(model,history,blocks,torch.as_tensor(corrections[j],device='cuda'))
                torch.testing.assert_close(value[:,:3],free[:,:3],rtol=0,atol=0)
                torch.testing.assert_close(value[:,3,:,384:],free[:,3,:,384:],rtol=0,atol=0)
                # Known controls are replaced at indices3..6; index7 is the untouched native terminal prediction.
                torch.testing.assert_close(value[:,3:7,:,394:],free[:,3:7,:,394:],rtol=0,atol=0)
                np.testing.assert_array_equal(value[0,3,:,:384].cpu().numpy().reshape(75264),corrections[j])
                assert torch.isfinite(value).all();results[branch+'_tokens']=value[0,3:].cpu().numpy()
            for branch in ['actual','donor']:
                visual=results[branch+'_tokens'][...,:384].reshape(5,75264)
                results[branch+'_pose']=numpy_pose(visual,head)
                results[branch+'_block_error']=np.square(results[branch+'_pose'][...,2:4]-baseline['truth'][...,2:4]).sum(-1)
            target=out/f'case_{i:03d}.npz';atomic_npz(target,**results,free_tokens=baseline['free_tokens'][3:],truth=baseline['truth'])
            rows.append(dict(case=i,seed=c['seed'],file=target.name,file_sha256=sha(target),checks=checks,
                matching=qpr['matching'],free_replay_exact=True,earlier_history_exact=True,current_proprio_exact=True,
                known_action_channels_exact=True,qp_report_sha256=sha(qp/f'qp/family_{fi:04d}.json')))
    assert state_hash(model)==before and len(rows)==len(cases)
    atomic_json(out/'report.json',dict(**ctx.binding,status='PASS_NATIVE_VISUAL_ONLY_CLOSEDLOOP',role=role,shard=shard,cases=cases,rows=rows,
        projection_report_sha256=sha(qp/'report.json'),cache_report_sha256=sha(cache/'report.json'),model_binding=mb,
        head_sha256=sha(a.head),source_sha256=sha(__file__),frozen_tensors_unchanged=True,
        scope='Complete fixed goal partition, both actual/donor corrections verified in full75264-D space. Earlier history, current proprio and known future action channels preserved.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
