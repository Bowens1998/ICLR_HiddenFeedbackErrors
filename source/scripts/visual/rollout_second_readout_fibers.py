"""Frozen-action second-head feedback diagnostic with complete free anchors."""
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
    for k in ['fibers','plan','bank','planner','official','config','output','protocol']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=[0,1,2],required=True);a=p.parse_args();d=Path(a.fibers)/f'job_{a.index}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text());g=r['group']
    assert ac['status']=='PASS_SECOND_HEAD_3072_PROJECTIONS_AND_MATCHING' and ac['source_files_checked'] and ac['report_sha256']==sha(d/'report.json') and ac['protocol_sha256']==sha(a.protocol)
    head=dict(np.load(d/'head.npz'));assert sha(d/'head.npz')==r['head_sha256'];plan=json.loads(Path(a.plan).read_text());bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert len(bm['cases'])==128 and sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'head.npz',**head);started=time.monotonic()
    histories=[]
    for c in bm['cases']:
        fp=bank/f"case_{c['index']:03d}.npz";assert sha(fp)==c['sha256']
        with np.load(fp) as z:histories.append({k:z[k].copy() for k in ['history_pixels','goal_pixels','prefix']})
    refs=[];bindings=[]
    for route in [16*g+k for k in [0,1,10,11]]:
        base=Path(a.planner)/f'job_{route}';sr=json.loads((base/'summary.json').read_text());sa=json.loads((base/'acceptance.json').read_text());manifest=json.loads((base/'artifact_manifest.json').read_text())
        assert sa['model_free_simulator_replay'] and sa['population_replay_exact_all_cases'] and sa['cases']==128
        assert sr['hashes']['model_manifest']==sha(a.plan) and manifest['summary.json']==sha(base/'summary.json');examples=[]
        for i,c in enumerate(sr['cases']):
            assert c['index']==i and c['seed']==bm['cases'][i]['seed'];fp=base/f'case_{i:03d}_predictions.npz';assert sha(fp)==manifest[fp.name]
            with np.load(fp) as z:
                actions=z['parameters'][c['selected_iteration']].reshape(300,25,2).copy();np.testing.assert_array_equal(actions[c['selected_candidate']],z['selected_actions']);examples.append((actions,c['selected_candidate']))
        refs.append(examples);bindings.append({'route':route,'summary_sha256':sha(base/'summary.json'),'acceptance_sha256':sha(base/'acceptance.json')})
    configure_evaluation_precision();torch.set_num_threads(2);rows=[];anchors=0
    for row in r['rows']:
        assert row['sha256']==sha(d/row['file']) and row['weights_sha256']==sha(row['weights_path']);z=dict(np.load(d/row['file']));entry=row['entry'];tr=json.loads((Path(entry['training_path'])/'summary.json').read_text())
        assert sha(a.config)==tr['config_sha256'] and sha(Path(entry['training_path'])/'summary.json')==entry['training_summary_sha256']
        model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(row['weights_path'],map_location='cpu',weights_only=True),strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        arrays={b+'_tokens':np.empty((4,128,5,192),np.float32) for b in ['free','actual','donor','reset']}
        with torch.inference_mode():
            for i,h in enumerate(histories):
                cost=ImagePlannerCost(model,h['history_pixels'],h['goal_pixels'],h['prefix'],tr['normalization'],None,'latent')
                for k,examples in enumerate(refs):
                    actions,selected=examples[i];encoded=model.action_encoder(cost.normalized_actions(torch.as_tensor(actions,device='cuda')));j=k*128+i
                    for branch in ['free','actual','donor','reset']:
                        history=cost.initial[None].expand(300,-1,-1).clone();saved=[]
                        for step in range(2,7):
                            token=model.predict(history[:,-3:],encoded[:,step-2:step+1])[:,-1:]
                            if step==2:
                                np.testing.assert_array_equal(token[selected,0].cpu().numpy(),z['predicted'][j])
                                if branch!='free':
                                    replacement=z['observed'][j] if branch=='reset' else z[branch+'_matched'][j]
                                    token=token.clone();token[selected,0]=torch.as_tensor(replacement,device='cuda')
                            saved.append(token[selected,0]);history=torch.cat([history,token],1)
                        arrays[branch+'_tokens'][k,i]=torch.stack(saved).cpu().numpy()
                    np.testing.assert_array_equal(arrays['free_tokens'][k,i],z['free_baseline_tokens'][k,i]);anchors+=1
        verify_frozen(model,boundary)
        for branch in ['free','actual','donor','reset']:arrays[branch+'_pose']=numpy_pose(arrays[branch+'_tokens'],head)
        name=row['objective']+'.npz';np.savez_compressed(out/name,**arrays,true_pose=z['true_pose'],observed_tokens=z['observed_tokens'],seeds=z['seeds'],reference_routes=z['reference_routes'])
        rows.append({'objective':row['objective'],'file':name,'sha256':sha(out/name),'weights_sha256':row['weights_sha256'],'frozen_tensors_unchanged':True});print('ROLLED',a.index,row['objective'],flush=True);del model,boundary;torch.cuda.empty_cache()
    assert anchors==1536
    report={'status':'SECOND_HEAD_ROLLOUT_REQUIRES_ACCEPTANCE','index':a.index,'group':g,'rows':rows,'complete_free_anchors':anchors,'fiber_report_sha256':sha(d/'report.json'),'fiber_acceptance_sha256':sha(d/'acceptance.json'),'head_sha256':sha(out/'head.npz'),'protocol_sha256':sha(a.protocol),'source_sha256':sha(__file__),'reference_bindings':bindings,'elapsed_seconds':time.monotonic()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
