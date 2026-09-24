"""All frozen A/C models on every development goal and four original action streams."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
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
from nonlinear_pose_cost import numpy_pose
from score_feedback_ranking import state_hash
from evaluation_lock import add_protocol_arguments,evaluation_context,bound_input,require_output


def main():
    p=argparse.ArgumentParser()
    for key in ['base','bank','actions','physics','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--group',type=int);add_protocol_arguments(p)
    a=p.parse_args();g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    base=Path(a.base);bank=Path(a.bank);role=require_role(json.loads((bank/'role.json').read_text()),
        ['diagnostic_development_A_C','donor_development_A_C','confirmation_A_C','donor_bank_A_C'])
    ctx=evaluation_context(a,role['role'],ROOT,__file__);count=ctx.count
    require_output(a.output,ctx,'AC_donor_cache' if role['role'].startswith('donor_') else 'AC_cache')
    if ctx.formal:
        assert Path(a.actions)==Path(ctx.design['banks'][role['role']]['actions_output'])
        assert Path(a.physics)==Path(ctx.design['banks'][role['role']]['physics_output'])
    bm=json.loads((bank/'manifest.json').read_text());assert len(bm['cases'])==role['count']==count
    assert [c['index'] for c in bm['cases']]==list(range(count))
    assert role['parent_manifest_sha256']==sha(bank/'manifest.json') and (bank/'DONE').exists()
    rosterpath=ROOT/'strengthening/manifests/development_model_roster.json';roster=json.loads(rosterpath.read_text());entry=roster['A_groups'][g]
    legacy=json.loads((ROOT/'strengthening/manifests/legacy_model_roster.json').read_text())['groups'][g]
    original=legacy['models'][0];cp=Path(original['checkpoint']);assert sha(cp)==original['checkpoint_sha256']
    tr=json.loads((cp.parent/'summary.json').read_text());config=base/'assets/pusht-v1/models/config.json';assert sha(config)==tr['config_sha256']
    originals=torch.load(cp,map_location='cpu',weights_only=True)
    models=[dict(m,kind='A',condition='A',name='A_'+m['objective']) for m in entry['models']]
    if g%2==0:models += [dict(m,kind='C') for m in roster['C_models'] if m['pool']==g//2]
    assert len(models)==(9 if g%2==0 else 3)
    if role['role'].startswith('donor_'):
        # Guidance needs only the shared frozen encoder; no C/objective grid on donors.
        models=models[:1]
    heads={}
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=False)
    for k in ['head_A','head_B']:
        assert sha(entry[k]['path'])==entry[k]['sha256'];heads[k]=dict(np.load(entry[k]['path']))
        atomic_npz(out/(k+'.npz'),**heads[k])
    contexts=[]
    for c in bm['cases']:
        path=bank/f"case_{c['index']:03d}.npz";assert sha(path)==c['sha256'];contexts.append(dict(np.load(path)))
    refs=[];bindings=[]
    for s in range(4):
        route=4*g+s;ad=Path(a.actions)/f'route_{route}';pd=Path(a.physics)/f'route_{route}'
        ar=json.loads((ad/'report.json').read_text());pr=json.loads((pd/'report.json').read_text())
        assert (ad/'DONE').exists() and (pd/'DONE').exists() and len(ar['cases'])==len(pr['cases'])==count
        assert pr['status'] in ['PASS_COMPLETE_DEVELOPMENT_SELECTED_PHYSICS','PASS_COMPLETE_SELECTED_PHYSICS']
        bound_input(ad/'report.json',ctx);bound_input(pd/'report.json',ctx)
        assert pr['binding']['action_report_sha256']==sha(ad/'report.json')
        assert ar['binding']['bank_manifest_sha256']==sha(bank/'manifest.json')
        examples=[]
        for c,arow,prow in zip(bm['cases'],ar['cases'],pr['cases'],strict=True):
            assert c['index']==arow['index']==prow['index'] and c['seed']==arow['seed']==prow['seed']
            ap=ad/arow['file'];pp=pd/prow['file'];assert sha(ap)==arow['file_sha256'] and sha(pp)==prow['file_sha256']
            az=dict(np.load(ap));pz=dict(np.load(pp));np.testing.assert_array_equal(az['selected_actions'],pz['actions'][10:])
            np.testing.assert_array_equal(contexts[c['index']]['history_pixels'],pz['pixels'][:3])
            st=pz['states'][[15,20,25,30,35]];truth=np.c_[st[:,:4],np.sin(st[:,4]),np.cos(st[:,4])]
            examples.append(dict(actions=az['population_actions'],selected=int(az['selected_index']),pixels=pz['pixels'][3:],truth=truth))
        refs.append(examples);bindings.append(dict(route=route,actions_sha256=sha(ad/'report.json'),physics_sha256=sha(pd/'report.json')))
    configure_evaluation_precision();torch.set_num_threads(4);rows=[];start=time.monotonic()
    observed_ref=np.empty((4,count,5,192),np.float32);initial_ref=np.empty((count,3,192),np.float32)
    truth=np.asarray([[x['truth'] for x in s] for s in refs])
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
    sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    for mi,m in enumerate(models):
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        if 'report_sha256' in m:assert sha(Path(m['checkpoint']).parent/'report.json')==m['report_sha256']
        model=make_model(base/'releases/visual-v1/official',config,legacy['architecture'],tr['seed'])
        model.load_state_dict(originals,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        model.load_state_dict(torch.load(m['checkpoint'],map_location='cpu',weights_only=True),strict=True);verify_frozen(model,boundary)
        before=state_hash(model);free=np.empty((4,count,5,192),np.float32);teacher=np.empty_like(free);response=[]
        with torch.inference_mode():
            for i,z in enumerate(contexts):
                cost=ImagePlannerCost(model,z['history_pixels'],z['goal_pixels'],z['prefix'],tr['normalization'],None,'latent')
                if mi==0:initial_ref[i]=cost.initial.cpu().numpy()
                else:np.testing.assert_array_equal(initial_ref[i],cost.initial.cpu().numpy())
                for s,examples in enumerate(refs):
                    x=examples[i];actions=torch.as_tensor(x['actions'],device='cuda');encoded=model.action_encoder(cost.normalized_actions(actions));j=x['selected']
                    if mi==0:
                        obs=[]
                        for pixel in x['pixels']:
                            tensor=torch.as_tensor(pixel,device='cuda').permute(2,0,1)[None,None].float()/255.
                            obs.append(model.encode({'pixels':(tensor-im)/sd})['emb'][0,0])
                        observed=torch.stack(obs);observed_ref[s,i]=observed.cpu().numpy()
                    else:observed=torch.as_tensor(observed_ref[s,i],device='cuda')
                    pred=rollout_population(model,cost.initial,encoded,j)
                    # Original production interface provides an independent endpoint computation.
                    _,anchor=cost(actions);torch.testing.assert_close(pred[-1],anchor[j],rtol=0,atol=0)
                    identity=rollout_population(model,cost.initial,encoded,j,pred[0].clone())
                    torch.testing.assert_close(pred,identity,rtol=0,atol=0)
                    observed_history=rollout_population(model,cost.initial,encoded,j,observed=observed)
                    torch.testing.assert_close(observed_history[0],pred[0],rtol=0,atol=0)
                    assert torch.isfinite(pred).all() and torch.isfinite(observed_history).all()
                    free[s,i]=pred.cpu().numpy();teacher[s,i]=observed_history.cpu().numpy()
                    if ctx.formal or i<2:
                        alt=model.action_encoder(cost.normalized_actions(-actions));probe=rollout_population(model,cost.initial,alt,j)
                        sensitivity=float((probe-pred).square().mean())
                        if not np.isfinite(sensitivity):raise ValueError('Nonfinite action response')
                        response.append(dict(stream=s,case=i,mean_square=sensitivity,legitimate_zero=sensitivity==0))
        verify_frozen(model,boundary);assert state_hash(model)==before
        fp=numpy_pose(free,heads['head_A']);tp=numpy_pose(teacher,heads['head_A']);target=out/(m['name']+'.npz')
        atomic_npz(target,free_tokens=free,observed_history_tokens=teacher,free_pose=fp,observed_history_pose=tp,
            free_block_error=np.square(fp[...,2:4]-truth[...,2:4]).sum(-1),
            observed_history_block_error=np.square(tp[...,2:4]-truth[...,2:4]).sum(-1))
        rows.append(dict(model=m,file=target.name,file_sha256=sha(target),all_endpoint_and_identity_exact=True,
            frozen_unchanged=True,action_response=response))
        del model,boundary,cost;torch.cuda.empty_cache();print('MODEL_ACCEPTED',g,m['name'],flush=True)
    atomic_npz(out/'observed.npz',observed_tokens=observed_ref,initial=initial_ref,truth=truth,
        observed_pose=numpy_pose(observed_ref,heads['head_A']),seeds=np.asarray([c['seed'] for c in bm['cases']]))
    atomic_json(out/'report.json',dict(**ctx.binding,status='PASS_AC_BASELINE',group=g,role=role['role'],expected_cases=count,
        parent_manifest_sha256=sha(bank/'manifest.json'),role_sha256=sha(bank/'role.json'),references=bindings,models=rows,
        observed_sha256=sha(out/'observed.npz'),head_sha256={k:sha(out/(k+'.npz')) for k in heads},
        roster_sha256=sha(rosterpath),source_sha256=sha(__file__),adapter_sha256=sha(ROOT/'strengthening/adapters/ac_rollout.py'),
        elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='All fixed goals and original four streams; strict original batch arithmetic. Baseline and observed-history diagnostics; no readout-preserving corrections. Zero action responsiveness is retained as a descriptive result.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
