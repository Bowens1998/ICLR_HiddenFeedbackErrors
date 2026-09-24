"""Fixed development ridge selector with shared-goal folds; no independent test claim."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha


def goal_folds(n):
    order=np.random.default_rng(1271901).permutation(n);fold=np.empty(n,dtype=int);fold[order]=np.arange(n)%4
    return fold


def fit_ridge(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-12);ym=float(y.mean());ys=max(float(y.std()),1e-12)
    z=(x-mean)/scale;target=(y-ym)/ys
    weights=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1]),z.T@target/len(z))
    return dict(mean=mean,scale=scale,target_mean=ym,target_scale=ys,weights=weights)


def predict(x,model):
    return ((x-model['mean'])/model['scale'])@model['weights']*model['target_scale']+model['target_mean']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.data);r=json.loads((source/'report.json').read_text());assert r['status']=='COMPLETE48_PAIRED_SEARCH_DIAGNOSTICS' and r['data_sha256']==sha(source/'paired_data.npz')
    z=dict(np.load(source/'paired_data.npz'));arrays={};groups=[]
    for group in r['groups']:
        task,interface=group['task'],group['interface'];rows=[v for v in r['rows'] if v['task']==task and v['interface']==interface]
        assert len(rows)==6 and [v['backbone'] for v in rows]==list(range(6))
        inputs=[];labels=[];seed_reference=None
        for row in rows:
            key=f"{task}_{row['provenance'][0]['route']}";xx=z[key+'_inputs'];yy=z[key+'_labels'];seeds=z[key+'_seeds']
            assert np.isfinite(xx).all() and (xx[:,:2]>=0).all()
            if seed_reference is None:seed_reference=seeds
            else:np.testing.assert_array_equal(seeds,seed_reference)
            inputs.append(np.column_stack([np.log1p(xx[:,:2]),xx[:,2]]));labels.append(yy)
        x=np.stack(inputs);y=np.stack(labels);n=x.shape[1];folds=goal_folds(n)
        predicted=np.full((6,n),np.nan);constant=np.zeros((6,n),dtype=bool);models=[]
        for fold in range(4):
            train=folds!=fold;test=~train
            assert not set(seed_reference[train])&set(seed_reference[test])
            model=fit_ridge(x[:,train].reshape(-1,3),y[:,train,0].reshape(-1))
            predicted[:,test]=predict(x[:,test],model)
            constant[:,test]=model['target_mean']<0
            models.append(dict(fold=fold,train_goals=int(train.sum()),test_goals=int(test.sum()),parameters={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}))
        assert np.isfinite(predicted).all();choice=predicted<0
        costs=dict(ridge=np.where(choice,y[:,:,3],y[:,:,2]),train_constant=np.where(constant,y[:,:,3],y[:,:,2]),always_random=y[:,:,2],always_cem=y[:,:,3],oracle_pair=np.minimum(y[:,:,2],y[:,:,3]))
        successes=dict(ridge=np.where(choice,y[:,:,5],y[:,:,4]),train_constant=np.where(constant,y[:,:,5],y[:,:,4]),always_random=y[:,:,4],always_cem=y[:,:,5])
        per_model=[dict(index=i,costs={k:float(v[i].mean()) for k,v in costs.items()},successes={k:float(v[i].mean()) for k,v in successes.items()},cem_selection_fraction=float(choice[i].mean())) for i in range(6)]
        groups.append(dict(task=task,interface=interface,costs={k:float(v.mean()) for k,v in costs.items()},successes={k:float(v.mean()) for k,v in successes.items()},cem_selection_fraction=float(choice.mean()),per_model=per_model,fold_models=models))
        key=task+'_'+interface;arrays[key+'_predicted_delta']=predicted;arrays[key+'_choice_cem']=choice;arrays[key+'_fold']=folds;arrays[key+'_seeds']=seed_reference
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'predictions.npz',**arrays)
    result=dict(status='COMPLETE8_CROSSFIT_GROUPS',groups=groups,source_data_report_sha256=sha(source/'report.json'),predictions_sha256=sha(out/'predictions.npz'),source_sha256=sha(__file__),scope='Four shared-goal folds; fixed ridge1 with training-only input/target normalization. Consumed development evidence, no independent confirmation or novelty. Both-search18000 total candidates; fixed9000-output choices are not optimized18000-budget competitors. Cost and success separately reported; oracle unavailable.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for g in groups:print(g['task'],g['interface'],g['costs'],'choose_cem',g['cem_selection_fraction'])


if __name__=='__main__':main()
