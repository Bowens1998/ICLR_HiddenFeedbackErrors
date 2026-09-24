"""All frozen models on identical accepted original-policy action sequences."""
import argparse,json,hashlib,time,shutil
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from evaluation_precision import configure_evaluation_precision
from adaptation_freeze import configure_dynamics_only,verify_frozen
from extract_pose_selected_endpoints import sha,pose,decode


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','runs','bank','official','config','protocol','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());assert len(plan['models'])==48 and len(plan['routes'])==96
    protocol_sha=sha(a.protocol);bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());assert len(bm['cases'])==128 and sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    histories=[]
    for c in bm['cases']:
        fp=bank/f"case_{c['index']:03d}.npz";assert sha(fp)==c['sha256']
        with np.load(fp) as z:histories.append({k:z[k].copy() for k in ('history_pixels','goal_pixels','prefix')})
    refs=[];reference_bindings=[];reference_routes=[16*a.index+k for k in (0,1,10,11)]
    for ri in reference_routes:
        d=Path(a.runs)/f'job_{ri}';r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text())
        assert ac['cases']==128 and ac['model_free_simulator_replay'] and ac['population_replay_exact_all_cases']
        assert ac['verifier_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_population_planner.py'))
        assert r['hashes']['model_manifest']==sha(a.plan) and sha(d/'summary.json')==am['summary.json']
        e=plan['models'][plan['routes'][ri]['model_index']];assert e['adaptation_condition']=='original'
        examples=[];bind=[]
        for i,c in enumerate(r['cases']):
            assert c['index']==i and c['seed']==bm['cases'][i]['seed'];fp=d/f'case_{i:03d}_predictions.npz';assert sha(fp)==am[fp.name]
            with np.load(fp) as z:
                examples.append(dict(actions=z['selected_actions'].copy(),terminal_pixels=z['terminal_pixels'].copy(),true_pose=pose(z['selected_states'][-1]),actor_tokens=z['tokens'][c['selected_iteration'],c['selected_candidate']].copy()))
            bind.append(dict(case=i,archive_sha256=am[fp.name],action_sha256=hashlib.sha256(examples[-1]['actions'].tobytes()).hexdigest()))
        refs.append(examples);reference_bindings.append(dict(route=ri,model_index=plan['routes'][ri]['model_index'],summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),cases=bind))
    configure_evaluation_precision();torch.set_num_threads(4);rows=[];real_reference={};goal_reference={};anchor_checks=0;same_image_checks=0
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    with torch.inference_mode():
        for slot in range(8):
            mi=8*a.index+slot;e=plan['models'][mi];orig=plan['models'][8*a.index+(0 if slot<5 else 5)];od=Path(orig['training_path']);td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
            assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256'] and sha(od/'last_weights.pt')==orig['weights_sha256'] and sha(a.config)==tr['config_sha256']
            model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(torch.load(od/'last_weights.pt',map_location='cpu',weights_only=True));model=model.cuda();boundary=configure_dynamics_only(model)
            model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True));verify_frozen(model,boundary)
            head=None;head_file=None
            if e['score']!='state':
                assert e['endpoint_head']==e['goal_head'];hp=Path(e['endpoint_head']['path']);assert sha(hp)==e['endpoint_head']['sha256'];head=dict(np.load(hp));head_file=f'head_{mi}.npz';shutil.copyfile(hp,out/head_file)
            def physical(token):
                return decode(token,head) if head is not None else token.astype(float)*np.asarray(e['target_normalization']['std'])+np.asarray(e['target_normalization']['mean'])
            pred=[[] for _ in refs];real=[[] for _ in refs];truth=[[] for _ in refs];goals=[]
            for i,h in enumerate(histories):
                cost=ImagePlannerCost(model,h['history_pixels'],h['goal_pixels'],h['prefix'],tr['normalization'],e['target_normalization'],'state' if e['score']=='state' else 'latent')
                goals.append(cost.goal.cpu().numpy())
                for k,examples in enumerate(refs):
                    x=examples[i];actions=torch.as_tensor(x['actions'],device='cuda').unsqueeze(0).expand(300,-1,-1).clone();_,tokens=cost(actions);token=tokens[0].cpu().numpy();assert np.isfinite(token).all()
                    if reference_bindings[k]['model_index']==mi:np.testing.assert_array_equal(token,x['actor_tokens']);anchor_checks+=1
                    pixels=torch.as_tensor(x['terminal_pixels'],device='cuda').permute(2,0,1)[None,None].float()/255
                    rt=model.encode({'pixels':(pixels-im)/sd})['emb'][0,0].cpu().numpy();assert np.isfinite(rt).all()
                    pred[k].append(token);real[k].append(rt);truth[k].append(x['true_pose'])
            pred=np.asarray(pred);real=np.asarray(real);truth=np.asarray(truth);goals=np.asarray(goals)
            if e['score'] in real_reference:
                np.testing.assert_array_equal(real,real_reference[e['score']]);np.testing.assert_array_equal(goals,goal_reference[e['score']]);same_image_checks+=640
            else:real_reference[e['score']]=real.copy();goal_reference[e['score']]=goals.copy()
            verify_frozen(model,boundary);name=f'model_{mi}.npz'
            np.savez_compressed(out/name,predicted_tokens=pred,real_tokens=real,predicted_pose=physical(pred),real_pose=physical(real),true_pose=truth,goal_tokens=goals,seeds=np.asarray([c['seed'] for c in bm['cases']]),reference_routes=np.asarray(reference_routes))
            rows.append(dict(model_index=mi,entry=e,file=name,sha256=sha(out/name),head_file=head_file,head_sha256=sha(out/head_file) if head_file else None,frozen_tensors_unchanged=True))
            print('MODEL_COMPLETE',mi,flush=True);del model,boundary,cost;torch.cuda.empty_cache()
    assert anchor_checks==512 and same_image_checks==3840 and protocol_sha==sha(a.protocol)
    sources={p:sha(Path(__file__).with_name(p)) for p in ('run_task_coordinate_matched_actions.py','image_planner_cost.py','factorial_model.py','evaluation_precision.py','adaptation_freeze.py','extract_pose_selected_endpoints.py','lewm_adapter.py')}
    report=dict(status='MATCHED_ACTION_EXTRACTION_REQUIRES_NUMPY_ACCEPTANCE',index=a.index,rows=rows,reference_bindings=reference_bindings,anchor_checks=anchor_checks,same_image_checks=same_image_checks,plan_sha256=sha(a.plan),protocol_sha256=protocol_sha,source_sha256=sha(__file__),sources=sources,elapsed_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),scope='All8 conditions on four fixed original-policy action streams128goals. Canonical repeated-action300 diagnostic, no search or new physical labels.512 original-actor token anchors and3840 continued/original same-image comparisons exact. Frozen tensors checked; independent NumPy acceptance still required.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('EXTRACTED',a.index,flush=True)

if __name__=='__main__':main()
