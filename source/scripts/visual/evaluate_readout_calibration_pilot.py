"""Original four action streams and all 128 goals under old/new frozen heads."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import sha
from adaptation_freeze import configure_dynamics_only,verify_frozen
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision

def main():
    p=argparse.ArgumentParser()
    for k in ['pilot','plan','bank','planner','horizon','official','config','output','protocol']:p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int,choices=[0,1],required=True);a=p.parse_args()
    pilot=Path(a.pilot)/f'group_{a.group}';pr=json.loads((pilot/'report.json').read_text());pa=json.loads((pilot/'acceptance.json').read_text())
    assert pa['report_sha256']==sha(pilot/'report.json') and pa['protocol_sha256']==sha(a.protocol)
    if not pa['head_gate_passed']:
        print('GATE_FAILURE_RETAINED',a.group,flush=True);return
    assert pa['status']=='PASS_CALIBRATION_AND_CONTINUATION_ARRAYS'
    plan=json.loads(Path(a.plan).read_text());bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert len(bm['cases'])==128 and sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    out=Path(a.output)/f'group_{a.group}';out.mkdir(parents=True,exist_ok=False)
    horizons=Path(a.horizon)/f'job_{a.group}';hr=json.loads((horizons/'report.json').read_text());ha=json.loads((horizons/'acceptance.json').read_text())
    assert ha['report_sha256']==sha(horizons/'report.json') and hr['plan_sha256']==sha(a.plan)
    histories=[]
    for c in bm['cases']:
        fp=bank/f"case_{c['index']:03d}.npz";assert sha(fp)==c['sha256']
        with np.load(fp) as z:histories.append({k:z[k].copy() for k in ['history_pixels','goal_pixels','prefix']})
    refs=[];ref_bind=[];routes=[16*a.group+k for k in [0,1,10,11]]
    for route in routes:
        base=Path(a.planner)/f'job_{route}';r=json.loads((base/'summary.json').read_text());ac=json.loads((base/'acceptance.json').read_text());manifest=json.loads((base/'artifact_manifest.json').read_text())
        assert ac['model_free_simulator_replay'] and ac['population_replay_exact_all_cases'] and ac['cases']==128
        assert r['hashes']['model_manifest']==sha(a.plan) and manifest['summary.json']==sha(base/'summary.json')
        rows=[]
        for i,c in enumerate(r['cases']):
            assert c['index']==i and c['seed']==bm['cases'][i]['seed']
            fp=base/f'case_{i:03d}_predictions.npz'
            # This active-release archive was protected from storage compaction.
            assert sha(fp)==manifest[fp.name]
            with np.load(fp) as z:
                actions=z['parameters'][c['selected_iteration']].reshape(300,25,2).copy()
                np.testing.assert_array_equal(actions[c['selected_candidate']],z['selected_actions'])
                rows.append({'actions':actions,'selected':c['selected_candidate']})
        refs.append(rows);ref_bind.append({'route':route,'summary_sha256':sha(base/'summary.json'),'acceptance_sha256':sha(base/'acceptance.json')})
    configure_evaluation_precision();torch.set_num_threads(2);report_rows=[];started=time.monotonic();anchors=0
    for kind in ['old','new']:
        head_path=pilot/('original_head.npz' if kind=='old' else 'fitted_head.npz');head=dict(np.load(head_path))
        np.savez_compressed(out/(kind+'_head.npz'),**head)
        for objective,slot in [('decoded_teacher',3),('physical_labels',4)]:
            mi=8*a.group+slot;entry=plan['models'][mi];old_dir=Path(entry['training_path']);tr=json.loads((old_dir/'summary.json').read_text())
            assert sha(old_dir/'summary.json')==entry['training_summary_sha256'] and sha(a.config)==tr['config_sha256']
            weights=old_dir/'last_weights.pt' if kind=='old' else pilot/objective/'last_weights.pt'
            expected=entry['weights_sha256'] if kind=='old' else next(x for x in pr['continuations'] if x['objective']==objective)['weights_sha256']
            assert sha(weights)==expected
            model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
            hrow=next(x for x in hr['rows'] if x['model_index']==mi);hp=horizons/hrow['file'];assert sha(hp)==hrow['sha256'];hz=dict(np.load(hp))
            free=np.empty((4,128,5,192),np.float32);teacher=np.empty_like(free)
            with torch.inference_mode():
                for i,h in enumerate(histories):
                    cost=ImagePlannerCost(model,h['history_pixels'],h['goal_pixels'],h['prefix'],tr['normalization'],None,'latent')
                    for k,examples in enumerate(refs):
                        x=examples[i];encoded=model.action_encoder(cost.normalized_actions(torch.as_tensor(x['actions'],device='cuda')))
                        observed=torch.as_tensor(hz['observed_tokens'][k,i],device='cuda')
                        history=cost.initial[None].expand(300,-1,-1).clone();th=history.clone();f=[];t=[]
                        for step in range(2,7):
                            nf=model.predict(history[:,-3:],encoded[:,step-2:step+1])[:,-1:]
                            nt=model.predict(th[:,-3:],encoded[:,step-2:step+1])[:,-1:]
                            f.append(nf[x['selected'],0]);t.append(nt[x['selected'],0])
                            history=torch.cat([history,nf],1)
                            th=torch.cat([th,observed[step-2][None,None].expand(300,1,-1)],1)
                        free[k,i]=torch.stack(f).cpu().numpy();teacher[k,i]=torch.stack(t).cpu().numpy()
                        np.testing.assert_array_equal(free[k,i,0],teacher[k,i,0])
                        if kind=='old':
                            np.testing.assert_array_equal(free[k,i],hz['free_tokens'][k,i]);np.testing.assert_array_equal(teacher[k,i],hz['teacher_tokens'][k,i]);anchors+=2
            verify_frozen(model,boundary)
            filename=kind+'_'+objective+'.npz'
            np.savez_compressed(out/filename,free_tokens=free,teacher_tokens=teacher,observed_tokens=hz['observed_tokens'],
                free_pose=numpy_pose(free,head),teacher_pose=numpy_pose(teacher,head),observed_pose=numpy_pose(hz['observed_tokens'],head),
                true_pose=hz['true_pose'],seeds=hz['seeds'],reference_routes=hz['reference_routes'])
            report_rows.append({'head':kind,'objective':objective,'file':filename,'sha256':sha(out/filename),'weights_sha256':expected,'source_horizon_file_sha256':sha(hp),'frozen_tensors_unchanged':True})
            print('MODEL_COMPLETE',a.group,kind,objective,flush=True);del model,boundary;torch.cuda.empty_cache()
    assert anchors==2048
    result={'status':'PILOT_FIXED_ACTION_EVALUATION_REQUIRES_ACCEPTANCE','group':a.group,'rows':report_rows,'old_complete_trajectory_anchors':anchors,
        'pilot_report_sha256':sha(pilot/'report.json'),'pilot_acceptance_sha256':sha(pilot/'acceptance.json'),
        'plan_sha256':sha(a.plan),'protocol_sha256':sha(a.protocol),'source_sha256':sha(__file__),'reference_bindings':ref_bind,
        'head_sha256':{k:sha(out/(k+'_head.npz')) for k in ['old','new']},'elapsed_seconds':time.monotonic()-started,
        'scope':'All 128 previously consumed goals and four unchanged action streams; new-head validation gate applied before this evaluation. No new search, goal filtering or primary confirmation.'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
