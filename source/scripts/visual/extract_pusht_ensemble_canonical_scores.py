"""PushT physical cross-scores of fixed selected actions; no future-state score inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose,numpy_cost
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['plan','bank','runs','summary','official','model-config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--actor',type=int,choices=range(6),required=True);p.add_argument('--interface',choices=['pose_encoded','state'],required=True);p.add_argument('--engineering',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());full=json.loads(Path(a.summary).read_text());assert plan['layout']=='pusht_nonlinear_pose' and len(plan['models'])==24 and len(plan['routes'])==48
    bank=Path(a.bank);assert sha(bank/'manifest.json')==plan['bank_manifest_sha256'];bm=json.loads((bank/'manifest.json').read_text());assert len(bm['cases'])==128
    off=2 if a.interface=='pose_encoded' else 6;rid=a.actor*8+off;ids=[(r*2+a.actor%2)*8+off for r in range(3)]
    entries=[plan['models'][plan['routes'][i]['model_index']] for i in ids];assert [e['replica'] for e in entries]==[0,1,2]
    assert all(e['score']==a.interface for e in entries) and len({e['arm'] for e in entries})==1
    accepted=[];bindings=[];manifests=[]
    for i in [rid,rid+1]:
        d=Path(a.runs)/f'job_{i}';r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());binding=next(v for v in full['bindings'] if v['index']==i)
        assert sha(d/'summary.json')==binding['summary_sha256'] and sha(d/'acceptance.json')==binding['acceptance_sha256']
        assert len(r['cases'])==ac['cases']==128 and r['route_index']==i and r['score_space']==a.interface
        assert r['hashes']['model_manifest']==sha(a.plan) and r['algorithm']==('random' if i==rid else 'cem')
        am=json.loads((d/'artifact_manifest.json').read_text());assert am['summary.json']==sha(d/'summary.json')
        accepted.append(r);manifests.append(am);bindings.append(dict(route=i,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json')))
    precision=configure_evaluation_precision();torch.set_num_threads(2);models=[];heads=[];norms=[]
    for e in entries:
        d=Path(e['training_path']);assert sha(d/'summary.json')==e['training_summary_sha256'] and sha(d/'last_weights.pt')==e['weights_sha256']
        tr=json.loads((d/'summary.json').read_text());assert tr['seed']==3072 and sha(a.model_config)==tr['config_sha256']
        model=make_model(a.official,a.model_config,e['arm'],tr['seed']);model.load_state_dict(torch.load(d/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);models.append(model.cuda().eval());norms.append(tr['normalization'])
        pair=[]
        if a.interface=='pose_encoded':
            for role in ['endpoint_head','goal_head']:
                hp=Path(e[role]['path']);assert sha(hp)==e[role]['sha256'];pair.append(dict(np.load(hp,allow_pickle=False)))
        heads.append(pair)
    count=2 if a.engineering else 128;arrays={};rows=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for case,b in enumerate(bm['cases'][:count]):
            assert b['index']==case;path=bank/f'case_{case:03d}.npz';assert sha(path)==b['sha256']
            with np.load(path,allow_pickle=False) as z:history=z['history_pixels'];goal=z['goal_pixels'];prefix=z['prefix'];seed=int(z['seed'])
            assert seed==b['seed'];actions=[];original=[];archives=[];original_tokens=[];original_goals=[]
            for i,r,am in zip([rid,rid+1],accepted,manifests):
                row=r['cases'][case];assert row['index']==case and row['seed']==seed
                path=Path(a.runs)/f'job_{i}'/f'case_{case:03d}_predictions.npz';assert sha(path)==am[path.name]
                with np.load(path,allow_pickle=False) as z:
                    actions.append(z['selected_actions']);original_tokens.append(z['tokens'][row['selected_iteration'],row['selected_candidate']]);original_goals.append(z['goal_tokens'])
                original.append(row['predicted_cost']);archives.append(am[path.name])
            actions=np.stack(actions);assert actions.shape==(2,25,2) and np.isfinite(actions).all() and abs(actions).max()<=np.float32(.35)
            at=torch.as_tensor(actions,device='cuda',dtype=torch.float32);scores=[];tokens=[];goals=[];batch_scores=[];batch_tokens=[];max_batch=0.;max_numpy=0.
            for model,e,pair,norm in zip(models,entries,heads,norms):
                base=ImagePlannerCost(model,history,goal,prefix,norm,e['target_normalization'],'latent' if pair else 'state')
                cost=NonlinearPoseCost(base,*pair) if pair else base
                batched,bt=cost(at);single_values=[];single_tokens=[]
                for k in range(2):
                    cost.verify_native(at[k:k+1]);single,st=cost(at[k:k+1]);repeat,rt=cost(at[k:k+1])
                    torch.testing.assert_close(single,repeat,rtol=0,atol=0);torch.testing.assert_close(st,rt,rtol=0,atol=0)
                    single_values.append(single);single_tokens.append(st)
                value=torch.cat(single_values);token=torch.cat(single_tokens)
                max_batch=max(max_batch,float(abs(value-batched).max()));batch_scores.append(batched.cpu().numpy());batch_tokens.append(bt.cpu().numpy())
                values=value.cpu().numpy();ts=token.cpu().numpy();gt=cost.goal.cpu().numpy()
                if pair:ep=numpy_pose(ts,pair[0]);gp=numpy_pose(gt,pair[1])
                else:
                    tn=e['target_normalization'];mean=np.asarray(tn['mean'],dtype=np.float32);std=np.asarray(tn['std'],dtype=np.float32)
                    ep=(ts*std+mean).astype(float);gp=(gt*std+mean).astype(float)
                reference=numpy_cost(ep,gp);np.testing.assert_allclose(values,reference,rtol=2e-5,atol=2e-5)
                max_numpy=max(max_numpy,float(abs(values-reference).max()));scores.append(values);tokens.append(ts);goals.append(gt)
            scores=np.stack(scores);batch_scores=np.stack(batch_scores)
            key=f'case_{case}';arrays[key+'_scores']=scores;arrays[key+'_tokens']=np.stack(tokens);arrays[key+'_goals']=np.stack(goals);arrays[key+'_actions']=actions;arrays[key+'_batch2_scores']=batch_scores;arrays[key+'_batch2_tokens']=np.stack(batch_tokens);arrays[key+'_original_tokens']=np.stack(original_tokens);arrays[key+'_original_goals']=np.stack(original_goals)
            rows.append(dict(case=case,seed=seed,input_archive_sha256=b['sha256'],selected_archive_sha256=archives,original_scores=original,max_batch_difference=max_batch,max_cpu_score_difference=max_numpy,batch_score_violations=int((~np.isclose(scores,batch_scores,rtol=2e-5,atol=2e-5)).sum()),original_score_violations=int((~np.isclose(scores[a.actor//2],original,rtol=2e-5,atol=2e-5)).sum())))
    out=Path(a.output)/f'actor_{a.actor}'/a.interface;out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'scores.npz',**arrays)
    root=Path(__file__).resolve().parents[2];sources=['scripts/visual/extract_pusht_ensemble_canonical_scores.py','scripts/visual/factorial_model.py','scripts/visual/lewm_adapter.py','scripts/visual/image_planner_cost.py','scripts/visual/nonlinear_pose_cost.py','scripts/visual/evaluation_precision.py','scripts/visual/analyze_coverage_goals.py']
    report=dict(status='ENGINEERING_COMPLETE_REQUIRES_ACCEPTANCE' if a.engineering else 'EXTRACTED_REQUIRES_ACCEPTANCE',actor=a.actor,interface=a.interface,cases=rows,members=entries,bindings=bindings,plan_sha256=sha(a.plan),authoritative_summary_sha256=sha(a.summary),scores_sha256=sha(out/'scores.npz'),source_sha256={s:sha(root/s) for s in sources},protocol_sha256=sha(root/'docs/maintrack/ENSEMBLE_DECISION_RISK_PROTOCOL.md'),numerical_policy='canonical_single_action_v1',numerical_protocol_sha256=sha(root/'docs/maintrack/PUSHT_ENSEMBLE_NUMERICAL_PROTOCOL.md'),precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),scope='Consumed fixed PushT pairs; canonical single-action policy, exact repeated forwards and native checks. Batch2 and original-search disagreements retained, not accepted as equivalent. CPU acceptance pending; no risk fit or new outcomes.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['cases','members','source_sha256']}))


if __name__=='__main__':main()
