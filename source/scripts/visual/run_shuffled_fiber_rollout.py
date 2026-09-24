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
    for k in ('plan','runs','bank','official','config','protocol','output','horizon','fibers'):p.add_argument('--'+k,required=True)
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
                examples.append(dict(population_actions=z['parameters'][c['selected_iteration']].reshape(300,25,2).copy(),selected_index=c['selected_candidate'],actions=z['selected_actions'].copy(),terminal_pixels=z['terminal_pixels'].copy(),true_pose=pose(z['selected_states'][-1]),actor_tokens=z['tokens'][c['selected_iteration'],c['selected_candidate']].copy()))
            bind.append(dict(case=i,archive_sha256=am[fp.name],action_sha256=hashlib.sha256(examples[-1]['actions'].tobytes()).hexdigest()))
        refs.append(examples);reference_bindings.append(dict(route=ri,model_index=plan['routes'][ri]['model_index'],summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),cases=bind))
    configure_evaluation_precision();torch.set_num_threads(4);rows=[];real_reference={};goal_reference={};anchor_checks=0;goal_checks=0;readout_checks=0;input_bindings=[]
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    with torch.inference_mode():
        for slot in (2,3,4):
            mi=8*a.index+slot;e=plan['models'][mi];orig=plan['models'][8*a.index+(0 if slot<5 else 5)];od=Path(orig['training_path']);td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
            assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256'] and sha(od/'last_weights.pt')==orig['weights_sha256'] and sha(a.config)==tr['config_sha256']
            # Match the original planner: parameters are normal tensors; only forwards use inference mode.
            with torch.inference_mode(False):
                model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(torch.load(od/'last_weights.pt',map_location='cpu',weights_only=True));model=model.cuda();boundary=configure_dynamics_only(model)
                model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True));verify_frozen(model,boundary)
                assert all(not p.is_inference() for p in model.parameters())
            head=None;head_file=None
            if e['score']!='state':
                assert e['endpoint_head']==e['goal_head'];hp=Path(e['endpoint_head']['path']);assert sha(hp)==e['endpoint_head']['sha256'];head=dict(np.load(hp));head_file=f'head_{mi}.npz';shutil.copyfile(hp,out/head_file)
            def physical(token):
                return decode(token,head) if head is not None else token.astype(float)*np.asarray(e['target_normalization']['std'])+np.asarray(e['target_normalization']['mean'])
            hd=Path(a.horizon)/f'job_{a.index}';hr=json.loads((hd/'report.json').read_text());hac=json.loads((hd/'acceptance.json').read_text())
            assert hac['status']=='PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS' and hac['report_sha256']==sha(hd/'report.json') and hac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_horizon.py'))
            assert hr['plan_sha256']==sha(a.plan);hrow=hr['rows'][slot];assert hrow['entry']==e and hrow['sha256']==sha(hd/hrow['file']);hz=dict(np.load(hd/hrow['file']))
            fd=Path(a.fibers)/f'job_{a.index}';fr=json.loads((fd/'report.json').read_text());fac=json.loads((fd/'acceptance.json').read_text())
            assert fac['status']=='PASS1536_PAIRED_SHUFFLED_FIBER_INPUTS' and fac['constrained_valid']==fac['shuffled_valid']==1536 and fac['report_sha256']==sha(fd/'report.json') and fac['source_sha256']==sha(Path(__file__).with_name('accept_shuffled_fiber_inputs.py'))
            assert fr['protocol_sha256']==protocol_sha and fr['source_sha256']==sha(Path(__file__).with_name('prepare_shuffled_fiber_inputs.py'))
            fb=next(x for x in fr['bindings'] if x.get('model_index')==mi);assert fb['input_sha256']==hrow['sha256'] and fb['head_sha256']==e['endpoint_head']['sha256'] and fb['output_sha256']==sha(fd/fb['output_file']);fz=dict(np.load(fd/fb['output_file']))
            input_bindings.append(dict(model_index=mi,horizon_report_sha256=sha(hd/'report.json'),horizon_acceptance_sha256=sha(hd/'acceptance.json'),horizon_file_sha256=hrow['sha256'],fiber_report_sha256=sha(fd/'report.json'),fiber_acceptance_sha256=sha(fd/'acceptance.json'),fiber_file_sha256=fb['output_sha256']))
            values={path:[[] for _ in refs] for path in ('free','fiber','shuffled','reset')};goals=[]
            for i,h in enumerate(histories):
                cost=ImagePlannerCost(model,h['history_pixels'],h['goal_pixels'],h['prefix'],tr['normalization'],e['target_normalization'],'latent')
                goal=cost.goal.cpu().numpy();np.testing.assert_array_equal(goal,hz['goal_tokens'][i]);goals.append(goal);goal_checks+=1
                for k,examples in enumerate(refs):
                    x=examples[i];actions=torch.as_tensor(x['population_actions'],device='cuda');encoded=model.action_encoder(cost.normalized_actions(actions));selected=x['selected_index'];j=128*k+i
                    np.testing.assert_array_equal(fz['predicted'][j],hz['free_tokens'][k,i,0]);np.testing.assert_array_equal(fz['observed'][j],hz['observed_tokens'][k,i,0]);np.testing.assert_array_equal(x['true_pose'],hz['true_pose'][k,i,-1])
                    correction=fz['constrained'][j]
                    for key in ('constrained','shuffled'):
                        err=abs((physical(fz[key][j])-physical(fz['predicted'][j]))/head['target_scale']);assert np.max(err)<=1e-6;readout_checks+=1
                    replacements={'shuffled':torch.as_tensor(fz['shuffled'][j],device='cuda'),'fiber':torch.as_tensor(correction,device='cuda'),'reset':torch.as_tensor(fz['observed'][j],device='cuda')}
                    for path in values:
                        history=cost.initial[None].expand(300,-1,-1).clone()
                        for step in range(2,7):
                            next_token=model.predict(history[:,-3:],encoded[:,step-2:step+1])[:,-1:]
                            if step==2 and path!='free':
                                next_token=next_token.clone();next_token[selected,0]=replacements[path]
                            history=torch.cat([history,next_token],1)
                        tokens=history[selected,3:].cpu().numpy();assert np.isfinite(tokens).all()
                        if path=='free':np.testing.assert_array_equal(tokens,hz['free_tokens'][k,i]);anchor_checks+=1
                        else:np.testing.assert_array_equal(tokens[0],fz[{'fiber':'constrained','shuffled':'shuffled','reset':'observed'}[path]][j])
                        values[path][k].append(tokens)
            verify_frozen(model,boundary);name=f'model_{mi}.npz';arrays={}
            for path,value in values.items():
                value=np.asarray(value);arrays[path+'_tokens']=value;arrays[path+'_pose']=physical(value)
            np.savez_compressed(out/name,**arrays,observed_tokens=hz['observed_tokens'],observed_pose=hz['observed_pose'],true_pose=hz['true_pose'],horizons=hz['horizons'],seeds=hz['seeds'],reference_routes=hz['reference_routes'],goal_tokens=np.asarray(goals))
            rows.append(dict(model_index=mi,entry=e,file=name,sha256=sha(out/name),head_file=head_file,head_sha256=sha(out/head_file),frozen_tensors_unchanged=True))
            print('MODEL_COMPLETE',mi,flush=True);del model,boundary,cost;torch.cuda.empty_cache()
    assert anchor_checks==1536 and readout_checks==3072 and goal_checks==384 and protocol_sha==sha(a.protocol)
    sources={p:sha(Path(__file__).with_name(p)) for p in ('run_shuffled_fiber_rollout.py','image_planner_cost.py','factorial_model.py','evaluation_precision.py','adaptation_freeze.py','extract_pose_selected_endpoints.py','lewm_adapter.py','accept_task_coordinate_horizon.py','accept_shuffled_fiber_inputs.py','prepare_shuffled_fiber_inputs.py','readout_fiber.py')}
    report=dict(status='SHUFFLED_FIBER_ROLLOUT_REQUIRES_NUMPY_ACCEPTANCE',index=a.index,rows=rows,reference_bindings=reference_bindings,anchor_checks=anchor_checks,readout_checks=readout_checks,goal_checks=goal_checks,input_bindings=input_bindings,plan_sha256=sha(a.plan),protocol_sha256=protocol_sha,source_sha256=sha(__file__),sources=sources,elapsed_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),scope='Six-direction matched-displacement guided-versus-shuffled feedback intervention at the first predicted block: free, matched and shuffled readout-preserving corrections, and full observed reset. All3 JEPA objectives,128goals and4fixed references. Exact complete free-trajectory anchors; no new training/search or later observed feedback.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('EXTRACTED',a.index,flush=True)

if __name__=='__main__':main()
