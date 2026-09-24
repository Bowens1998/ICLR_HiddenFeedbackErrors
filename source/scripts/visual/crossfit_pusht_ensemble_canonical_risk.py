"""Prespecified score-versus-ensemble ridge development comparison, shared goal folds."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha
from crossfit_planner_regret import goal_folds,fit_ridge,predict


def main():
    p=argparse.ArgumentParser()
    for k in ['scores','data','baseline','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--task',choices=['pusht'],required=True);p.add_argument('--numerical-audit',required=True);a=p.parse_args()
    audit=json.loads(Path(a.numerical_audit).read_text());assert audit['status']=='COMPLETE12_CANONICAL_NUMERICAL_AUDITS' and len(audit['rows'])==12
    data=Path(a.data);dr=json.loads((data/'report.json').read_text());br=json.loads((Path(a.baseline)/'report.json').read_text())
    assert dr['data_sha256']==sha(data/'paired_data.npz') and br['source_data_report_sha256']==sha(data/'report.json')
    z=dict(np.load(data/'paired_data.npz'));result=[];arrays={};bindings=[]
    interfaces=['circle_encoded','state'] if a.task=='reacher' else ['pose_encoded','state']
    for interface in interfaces:
        rows=[r for r in dr['rows'] if r['task']==a.task and r['interface']==interface];assert len(rows)==6
        inputs=[];labels=[];ensemble=[];seeds=None
        for i,row in enumerate(rows):
            key=f"{a.task}_{row['provenance'][0]['route']}";x=z[key+'_inputs'];y=z[key+'_labels'];seed=z[key+'_seeds']
            if seeds is None:seeds=seed
            else:np.testing.assert_array_equal(seeds,seed)
            folder=Path(a.scores)/f'actor_{i}'/interface;r=json.loads((folder/'report.json').read_text());ac=json.loads((folder/'acceptance.json').read_text())
            assert r['status']=='EXTRACTED_REQUIRES_ACCEPTANCE' and ac['status']=='PASS' and ac['cases']==len(seed)
            assert ac['report_sha256']==sha(folder/'report.json') and ac['scores_sha256']==r['scores_sha256']==sha(folder/'scores.npz')
            assert ac['actor']==r['actor']==i and ac['interface']==r['interface']==interface
            for rb,db in zip(r['bindings'],row['provenance']):assert rb==db
            assert [v['seed'] for v in r['cases']]==list(seed)
            q=dict(np.load(folder/'scores.npz'));s=np.stack([q[f'case_{j}_scores'] for j in range(len(seed))]);assert s.shape==(len(seed),3,2)
            assert r['numerical_policy']==ac['numerical_policy']=='canonical_single_action_v1'
            ar=next(v for v in audit['rows'] if v['actor']==i and v['interface']==interface)
            assert ar['report_sha256']==sha(folder/'report.json') and ar['acceptance_sha256']==sha(folder/'acceptance.json')
            np.testing.assert_array_equal(np.array([v['original_scores'] for v in r['cases']]),x[:,:2])
            assert np.isfinite(s).all() and (s>=0).all()
            original=np.column_stack([np.log1p(x[:,:2]),x[:,2]])
            mean=s.mean(1);std=s.std(1);votes=(s[:,:,1]<s[:,:,0]).mean(1)
            inputs.append(np.column_stack([original,np.log1p(mean),np.log1p(std),votes]));labels.append(y);ensemble.append(mean[:,1]<mean[:,0])
            bindings.append(dict(actor=i,interface=interface,report_sha256=sha(folder/'report.json'),acceptance_sha256=sha(folder/'acceptance.json')))
        x=np.stack(inputs);y=np.stack(labels);n=len(seeds);fold=goal_folds(n);pred={k:np.full((6,n),np.nan) for k in ['ridge','augmented_ridge']};constant=np.zeros((6,n),dtype=bool);models=[]
        for f in range(4):
            train=fold!=f;test=~train
            assert not set(seeds[train])&set(seeds[test])
            for name,dim in [('ridge',3),('augmented_ridge',8)]:
                m=fit_ridge(x[:,train,:dim].reshape(-1,dim),y[:,train,0].reshape(-1));pred[name][:,test]=predict(x[:,test,:dim],m)
                constant[:,test]=m['target_mean']<0
                models.append(dict(fold=f,method=name,parameters={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}))
        choices={k:v<0 for k,v in pred.items()};choices.update(train_constant=constant,ensemble_mean=np.stack(ensemble),always_random=np.zeros((6,n),bool),always_cem=np.ones((6,n),bool))
        costs={k:np.where(v,y[:,:,3],y[:,:,2]) for k,v in choices.items()};success={k:np.where(v,y[:,:,5],y[:,:,4]) for k,v in choices.items()}
        original=next(g for g in br['groups'] if g['task']==a.task and g['interface']==interface)
        for name in ['ridge','train_constant','always_random','always_cem']:
            np.testing.assert_allclose(costs[name].mean(),original['costs'][name],rtol=0,atol=1e-12)
            np.testing.assert_allclose(success[name].mean(),original['successes'][name],rtol=0,atol=1e-12)
        group=dict(task=a.task,interface=interface,costs={k:float(v.mean()) for k,v in costs.items()},successes={k:float(v.mean()) for k,v in success.items()},cem_selection_fraction={k:float(v.mean()) for k,v in choices.items()},changes_from_ridge={k:int((v!=choices['ridge']).sum()) for k,v in choices.items()},fold_models=models,per_model=[dict(index=i,costs={k:float(v[i].mean()) for k,v in costs.items()},successes={k:float(v[i].mean()) for k,v in success.items()}) for i in range(6)])
        result.append(group);key=a.task+'_'+interface;arrays[key+'_inputs']=x;arrays[key+'_seeds']=seeds;arrays[key+'_fold']=fold
        for name,value in pred.items():arrays[key+'_'+name+'_predicted_delta']=value
        for name,value in choices.items():arrays[key+'_'+name+'_choice']=value
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'predictions.npz',**arrays)
    r=dict(status='COMPLETE2_ENSEMBLE_CROSSFIT_GROUPS',task=a.task,numerical_policy='canonical_single_action_v1',numerical_audit_sha256=sha(a.numerical_audit),groups=result,bindings=bindings,data_report_sha256=sha(data/'report.json'),baseline_report_sha256=sha(Path(a.baseline)/'report.json'),predictions_sha256=sha(out/'predictions.npz'),source_sha256=sha(__file__),ridge_source_sha256=sha(Path(__file__).with_name('crossfit_planner_regret.py')),scope='PushT supplementary canonical single-action ensemble scores; original actor ridge features remain original-search scores. Original-search and batch2 scores not claimed equivalent; full no-outcome numerical audit bound. Consumed development banks; shared goal folds, fixed ridge1 and feature set. Three same-architecture pools share initialization. Both searches18000 actor candidates plus6 canonical selected-action scores across3 members/encodings and3-model training; actor canonical replay is required for this declared policy; no equal-total-budget or independent confirmation claim. No search or method selection based on outcomes.')
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    for g in result:print(g['task'],g['interface'],g['costs'],g['successes'],g['changes_from_ridge'])


if __name__=='__main__':main()
