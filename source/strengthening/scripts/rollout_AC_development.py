"""A six/twelve-direction and C six-direction corrected rollout acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import sha,atomic_json,require_role
from artifact_io import atomic_npz
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only,verify_frozen
from evaluation_precision import configure_evaluation_precision
from image_planner_cost import ImagePlannerCost
from ac_rollout import rollout_population
from projection import Direction,match_family
from verifier import verify_token
from nonlinear_pose_cost import numpy_pose
from score_feedback_ranking import state_hash
from evaluation_lock import (add_protocol_arguments,evaluation_context,partition,
    check_evaluation_parent,require_output,bound_input)


def main():
    p=argparse.ArgumentParser()
    for k in ['base','bank','actions','cache','projection','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int);add_protocol_arguments(p,partitioned=True);a=p.parse_args()
    formal=bool(a.protocol_lock or a.protocol_sha256);role='confirmation_A_C' if formal else 'diagnostic_development_A_C'
    ctx=evaluation_context(a,role,ROOT,__file__);g,shard,cases=partition(a,ctx,'AC')
    for key,path in [('cache',a.cache),('QP',a.projection),('rollout',a.output)]:require_output(path,ctx,'AC_'+key)
    base=Path(a.base);bank=Path(a.bank);cache=Path(a.cache)/f'group_{g}';qp=Path(a.projection)/f'group_{g}'
    if shard is not None:qp=qp/f'shard_{shard:03d}'
    cr=json.loads((cache/'report.json').read_text());require_role(cr,{role});check_evaluation_parent(cr,ctx)
    assert cr['parent_manifest_sha256']==sha(bank/'manifest.json');bm=json.loads((bank/'manifest.json').read_text())
    pr=json.loads((qp/'report.json').read_text());design=json.loads((qp/'design.json').read_text())
    assert pr['status'] in ['PASS_CLOSEDLOOP_DEVELOPMENT_QP','PASS_CLOSEDLOOP_QP'] and (qp/'DONE').exists()
    if formal:
        check_evaluation_parent(pr,ctx);assert pr['cases']==cases and pr['group']==g and pr['shard']==shard
        assert sorted({f['case'] for f in design['families']})==cases
    assert sha(qp/'design.json')==pr['design_sha256'] and sha(qp/'qp/report.json')==pr['qp_report_sha256']
    assert design['binding']['recipient_report_sha256']==sha(cache/'report.json')
    obs=dict(np.load(cache/'observed.npz'));assert sha(cache/'observed.npz')==cr['observed_sha256']
    heads=[]
    for k in ['head_A','head_B']:
        assert sha(cache/(k+'.npz'))==cr['head_sha256'][k];heads.append(dict(np.load(cache/(k+'.npz'))))
    legacy=json.loads((ROOT/'strengthening/manifests/legacy_model_roster.json').read_text())['groups'][g]
    cp=Path(legacy['models'][0]['checkpoint']);assert sha(cp)==legacy['models'][0]['checkpoint_sha256']
    tr=json.loads((cp.parent/'summary.json').read_text());original=torch.load(cp,map_location='cpu',weights_only=True)
    configure_evaluation_precision();torch.set_num_threads(4);out=Path(a.output)/f'group_{g}'
    if shard is not None:out=out/f'shard_{shard:03d}'
    out.mkdir(parents=True,exist_ok=False)
    families=[]
    for fi,f in enumerate(design['families']):
        row=json.loads((qp/f'qp/family_{fi:04d}.json').read_text());file=qp/f'qp/family_{fi:04d}.npz'
        assert sha(file)==row['arrays_sha256'] and row['member_ids']==f['member_ids']
        arrays=dict(np.load(file));families.append(dict(descriptor=f,projection=row,arrays=arrays,results={},checks={}))
    all_rows=[]
    for mr in cr['models']:
        m=mr['model'];assert sha(m['checkpoint'])==m['checkpoint_sha256']
        savedpath=cache/mr['file'];assert sha(savedpath)==mr['file_sha256'];baseline=dict(np.load(savedpath))
        model=make_model(base/'releases/visual-v1/official',base/'assets/pusht-v1/models/config.json',legacy['architecture'],tr['seed'])
        model.load_state_dict(original,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        model.load_state_dict(torch.load(m['checkpoint'],map_location='cpu',weights_only=True),strict=True);verify_frozen(model,boundary);before=state_hash(model)
        with torch.inference_mode():
            for family in families:
                f=family['descriptor'];members=[(j,x) for j,x in enumerate(f['member_ids']) if x.split('/')[1]==m['name']]
                if not members:continue
                i=f['case'];s=f['stream'];c=bm['cases'][i];context=bank/f'case_{i:03d}.npz';assert sha(context)==c['sha256'];z=dict(np.load(context))
                route=Path(a.actions)/f'route_{4*g+s}';ar=json.loads((route/'report.json').read_text())
                bound_input(route/'report.json',ctx)
                assert sha(route/'report.json')==cr['references'][s]['actions_sha256']
                actionfile=route/ar['cases'][i]['file'];assert sha(actionfile)==ar['cases'][i]['file_sha256'];az=dict(np.load(actionfile))
                cost=ImagePlannerCost(model,z['history_pixels'],z['goal_pixels'],z['prefix'],tr['normalization'],None,'latent')
                actions=torch.as_tensor(az['population_actions'],device='cuda');selected=int(az['selected_index']);encoded=model.action_encoder(cost.normalized_actions(actions))
                normalized_before=cost.normalized_actions(actions).clone();initial_before=cost.initial.clone()
                free=rollout_population(model,cost.initial,encoded,selected);np.testing.assert_array_equal(free.cpu().numpy(),baseline['free_tokens'][s,i])
                reset=rollout_population(model,cost.initial,encoded,selected,torch.as_tensor(obs['observed_tokens'][s,i,0],device='cuda'))
                for j,member in members:
                    constraint,_,source=member.split('/');hs=heads if constraint=='dual_A_B' else heads[:1]
                    replacement=family['arrays']['corrected'][j]
                    check=verify_token(free[0].cpu().numpy(),replacement,hs,heads[0],expected_norm=family['projection']['matching']['effective_norm'])
                    assert check['accepted']
                    value=rollout_population(model,cost.initial,encoded,selected,torch.as_tensor(replacement,device='cuda'))
                    np.testing.assert_array_equal(value[0].cpu().numpy(),replacement);assert torch.isfinite(value).all()
                    native_value=None;native_matching=None
                    if f['kind'].startswith('C_'):
                        # Duplicate the same direction only to reuse the two-member numerical verifier;
                        # its own full feasible norm (with numerical backoff) is descriptive, not C's matched budget.
                        direction=Direction(family['arrays']['full_standardized_directions'][j],{})
                        full,native_matching=match_family([free[0].cpu().numpy()]*2,[direction]*2,[hs]*2,[heads[0]]*2)
                        native_value=rollout_population(model,cost.initial,encoded,selected,torch.as_tensor(full[0],device='cuda')).cpu().numpy()
                    family['results'][member]=dict(corrected=value.cpu().numpy(),free=free.cpu().numpy(),reset=reset.cpu().numpy(),full_feasible=native_value)
                    family['checks'][member]=dict(check=check,full_feasible_matching=native_matching)
                torch.testing.assert_close(cost.initial,initial_before,rtol=0,atol=0)
                torch.testing.assert_close(cost.normalized_actions(actions),normalized_before,rtol=0,atol=0)
        verify_frozen(model,boundary);assert state_hash(model)==before
        all_rows.append(dict(model=m,frozen_unchanged=True));del model,boundary,cost;torch.cuda.empty_cache()
    result_rows=[]
    for fi,family in enumerate(families):
        f=family['descriptor'];members=f['member_ids'];assert set(members)==set(family['results'])
        arrays={key:np.stack([family['results'][m][key] for m in members]) for key in ['corrected','free','reset']}
        if f['kind'].startswith('C_'):arrays['full_feasible']=np.stack([family['results'][m]['full_feasible'] for m in members])
        truth=obs['truth'][f['stream'],f['case']]
        for key in list(arrays):
            pose=numpy_pose(arrays[key],heads[0]);arrays[key+'_pose_A']=pose
            arrays[key+'_block_error_A']=np.square(pose[...,2:4]-truth[...,2:4]).sum(-1)
            arrays[key+'_pose_B']=numpy_pose(arrays[key],heads[1])
        path=out/f'family_{fi:04d}.npz';atomic_npz(path,**arrays,truth=truth)
        result_rows.append(dict(descriptor=f,file=path.name,file_sha256=sha(path),checks=family['checks'],matching=family['projection']['matching']))
    atomic_json(out/'report.json',dict(**ctx.binding,status='PASS_AC_CLOSEDLOOP',role=role,group=g,shard=shard,cases=cases,models=all_rows,families=result_rows,
        cache_report_sha256=sha(cache/'report.json'),projection_report_sha256=sha(qp/'report.json'),source_sha256=sha(__file__),
        scope='Complete fixed goal partition, all four streams; full FP32 insertion/head/budget and exact free replay. A dual single branches are rerun at the new twelve-member norm. C additionally records individually feasible doses as descriptive only.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
