"""Frozen one-neighbor support projection; actual labels only evaluate choices."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pose_selected_endpoints import error,sha
from extract_pose_selected_endpoints import decode


def nearest(query,train,mean,scale):
    q=(np.asarray(query,dtype=float)-mean)/scale;t=(np.asarray(train,dtype=float)-mean)/scale
    indices=[];distances=[]
    for start in range(0,len(q),16):
        ds=((q[start:start+16,None]-t[None])**2).mean(-1)
        ix=np.argmin(ds,axis=1);indices.extend(ix);distances.extend(ds[np.arange(len(ix)),ix])
    return np.array(indices),np.array(distances)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
    plan_path=Path('runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json');plan=json.loads(plan_path.read_text())
    prior_path=Path('outputs/maintrack/pusht_pair_substitution.json');prior=json.loads(prior_path.read_text())
    full=json.loads(Path('outputs/maintrack/pusht_pose_planning_summary.json').read_text());rows=[];arrays={};bindings=[]
    for actor in range(6):
        fd=Path('runs/pusht_pose_features_full_v1')/f'job_{actor}';fr=json.loads((fd/'report.json').read_text());fa=json.loads((fd/'acceptance.json').read_text())
        assert fa['status']=='PASS' and not fa['engineering'] and not fr['engineering'] and fa['report_sha256']==sha(fd/'report.json')
        assert sha(fd/'train_features.npz')==fr['files_sha256']['train_features.npz']
        train=np.load(fd/'train_features.npz')['encoded'];assert len(train)==fr['counts']['train']
        td=Path('runs/pusht_terminal_encoding_full_v1')/f'actor_{actor}';tr=json.loads((td/'report.json').read_text());ta=json.loads((td/'acceptance.json').read_text())
        assert ta['status']=='PASS' and not ta['engineering'] and ta['report_sha256']==sha(td/'report.json') and tr['plan_sha256']==sha(plan_path)
        ed=Path('runs/pusht_ensemble_canonical_full_v1')/f'actor_{actor}'/'pose_encoded';er=json.loads((ed/'report.json').read_text());ea=json.loads((ed/'acceptance.json').read_text())
        assert ea['status']=='PASS' and ea['report_sha256']==sha(ed/'report.json') and ea['scores_sha256']==er['scores_sha256']==sha(ed/'scores.npz')
        eq=np.load(ed/'scores.npz');route0=actor*8+2;entry=plan['models'][plan['routes'][route0]['model_index']]
        assert fr['weights_sha256']==entry['weights_sha256'] and er['members'][actor//2]['weights_sha256']==entry['weights_sha256']
        scores={k:[] for k in ['uncorrected','projected']};metrics=[]
        for k,route in enumerate([route0,route0+1]):
            row=next(v for v in tr['rows'] if v['route']==route);assert sha(td/row['file'])==row['file_sha256']
            head=dict(np.load(td/row['head_file']));assert sha(td/row['head_file'])==entry['goal_head']['sha256']==row['head_sha256']
            z=dict(np.load(td/row['file']));np.testing.assert_array_equal(z['seed'],full['goal_seeds'])
            for i,c in enumerate(row['cases']):
                assert c['seed']==er['cases'][i]['seed'] and c['archive_sha256']==er['cases'][i]['selected_archive_sha256'][k]
            original=np.stack([eq[f'case_{i}_original_tokens'][k] for i in range(128)])
            np.testing.assert_allclose(decode(original,head),z['estimated_endpoint'],rtol=1e-10,atol=1e-8)
            poses={}
            for name,token in [('imagined',original),('real_image',z['real_tokens'])]:
                ix,dist=nearest(token,train,head['mean'],head['scale']);projected=decode(train[ix],head)
                before=decode(token,head);c0,ok0=error(before,z['true_endpoint']);c1,ok1=error(projected,z['true_endpoint'])
                metrics.append(dict(algorithm='random' if k==0 else 'cem',query=name,before_position_mse=float(c0[:,0].mean()),after_position_mse=float(c1[:,0].mean()),before_angle_mse=float(c0[:,1].mean()),after_angle_mse=float(c1[:,1].mean()),before_precision=float(ok0.mean()),after_precision=float(ok1.mean())))
                arrays[f'actor_{actor}_{k}_{name}_indices']=ix;arrays[f'actor_{actor}_{k}_{name}_distance']=dist
                if name=='imagined':poses=dict(uncorrected=before,projected=projected)
            for name,pose in poses.items():scores[name].append(error(pose,z['estimated_goal'])[0].sum(1))
        actual=np.stack([full['rows'][r]['cost'] for r in [route0,route0+1]],1);success=np.stack([full['rows'][r]['success'] for r in [route0,route0+1]],1);policies={}
        for name,score in scores.items():
            s=np.stack(score,1);choice=s[:,1]<s[:,0];cost=actual[np.arange(128),choice.astype(int)]
            policies[name]=dict(cost=float(cost.mean()),success=float(success[np.arange(128),choice.astype(int)].mean()),regret=float((cost-actual.min(1)).mean()),cem_fraction=float(choice.mean()))
            arrays[f'actor_{actor}_{name}_choice']=choice
        old=next(g for g in prior['groups'] if g['interface']=='pose_encoded')['per_actor'][actor]
        for metric in ['cost','success','regret','cem_fraction']:np.testing.assert_allclose(policies['uncorrected'][metric],old['policies']['uncorrected'][metric],rtol=0,atol=1e-9)
        rows.append(dict(actor=actor,train_references=len(train),metrics=metrics,policies=policies,random_cost=float(actual[:,0].mean()),random_success=float(success[:,0].mean())))
        bindings.append(dict(actor=actor,feature_report_sha256=sha(fd/'report.json'),feature_acceptance_sha256=sha(fd/'acceptance.json'),terminal_report_sha256=sha(td/'report.json'),terminal_acceptance_sha256=sha(td/'acceptance.json'),ensemble_report_sha256=sha(ed/'report.json'),ensemble_acceptance_sha256=sha(ed/'acceptance.json')))
    pooled={name:{k:float(np.mean([r['policies'][name][k] for r in rows])) for k in ['cost','success','regret','cem_fraction']} for name in ['uncorrected','projected']}
    out.mkdir();np.savez_compressed(out/'neighbors.npz',**arrays)
    result=dict(status='COMPLETE6_SUPPORT_PROJECTION_BASELINES',rows=rows,pooled=pooled,bindings=bindings,source_sha256=sha(__file__),protocol_sha256=sha('docs/maintrack/PUSHT_SUPPORT_PROJECTION_PROTOCOL.md'),neighbors_sha256=sha(out/'neighbors.npz'),prior_sha256=sha(prior_path),scope='Original encoded-pose JEPA actions; training-only references and frozen normalization. No future outcomes enter projected selector; real-image projection is privileged diagnostic only. Consumed goals, no full replanning or novel-method/total-budget claim.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(pooled)


if __name__=='__main__':main()
