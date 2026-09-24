"""Goal-disjoint ridge-probe feasibility before a physical motion intervention."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_readout_fiber_rollout_inputs import forward

PENALTIES=np.array([.0001,.001,.01,.1,1.,10.,100.])


def fit(x,y,penalty):
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-6,scale,1.);v=(x-mean)/scale;ym=y.mean(0)
    gram=v.T@v/len(v);rhs=v.T@(y-ym)/len(v)
    w=np.linalg.solve(gram+penalty*np.eye(v.shape[1]),rhs)
    return dict(mean=mean,scale=scale,weight=w,intercept=ym)


def predict(x,w):return ((x-w['mean'])/w['scale'])@w['weight']+w['intercept']


def features(z,h):
    obs=z['observed'][:,1:4].astype(float);raw=z['raw_states'][:,1:4]
    decoded=forward(h,obs)[0]*h['target_scale']+h['target_mean']
    true=np.concatenate([raw[:,:,:4],np.sin(raw[:,:,4:5]),np.cos(raw[:,:,4:5])],axis=-1)
    return dict(latent_current=obs[:,-1],latent_history=obs.reshape(len(obs),-1),decoded_pose_history=decoded.reshape(len(obs),-1),true_pose_history=true.reshape(len(obs),-1))


def main():
    p=argparse.ArgumentParser()
    for k in ('cache','horizon','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];bindings=[]
    for g in range(6):
        d=Path(a.cache)/f'job_{2*g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text());assert ac['status']=='PASS_FULL_FRESH_PROCESS_REENCODING' and ac['cache_report_sha256']==sha(d/'report.json')
        z={}
        for split in ('train','validation'):
            rr=next(x for x in r['rows'] if x['stream']=='planner_'+split);assert sha(d/rr['file'])==rr['sha256'];z[split]=dict(np.load(d/rr['file']))
        ids=z['train']['identities'][:,2];vids=z['validation']['identities'][:,2];assert not set(ids)&set(vids)
        hd=Path(a.horizon)/f'job_{g}';hr=json.loads((hd/'report.json').read_text());row=hr['rows'][2];assert sha(hd/row['head_file'])==r['entry']['endpoint_head']['sha256']==row['head_sha256'];h=dict(np.load(hd/row['head_file']))
        xx={s:features(v,h) for s,v in z.items()};yy={s:v['raw_states'][:,3,5:7] for s,v in z.items()}
        goals=np.unique(ids);np.random.default_rng(1365001).shuffle(goals);folds=np.array_split(goals,5);fold_id=np.full(len(ids),-1)
        for f,goals_f in enumerate(folds):fold_id[np.isin(ids,goals_f)]=f
        assert (fold_id>=0).all();baseline=np.mean((yy['validation']-yy['train'].mean(0))**2,axis=0);artifacts={}
        for name in xx['train']:
            x=xx['train'][name];v=xx['validation'][name];scores=[]
            for penalty in PENALTIES:
                predictions=np.empty_like(yy['train'])
                for f in range(5):
                    mask=fold_id==f;w=fit(x[~mask],yy['train'][~mask],penalty);predictions[mask]=predict(x[mask],w)
                scores.append(float(np.mean(np.sum((predictions-yy['train'])**2,axis=-1))))
            chosen=int(np.argmin(scores));w=fit(x,yy['train'],PENALTIES[chosen]);prediction=predict(v,w);train_prediction=predict(x,w);mse=np.mean((prediction-yy['validation'])**2,axis=0);r2=1-mse/np.var(yy['validation'],axis=0)
            fp=out/f'{g}_{name}.npz';np.savez_compressed(fp,**w,train_prediction=train_prediction,validation_prediction=prediction,train_goal_seeds=ids,validation_goal_seeds=vids,fold_id=fold_id,penalty=PENALTIES[chosen])
            artifacts[name]=dict(file=fp.name,sha256=sha(fp));rows.append(dict(group=g,features=name,dimensions=x.shape[1],selected_penalty=float(PENALTIES[chosen]),cv_scores=scores,validation_axis_r2=r2.tolist(),validation_axis_mse=mse.tolist(),validation_summed_mse=float(mse.sum()),train_mean_baseline_axis_mse=baseline.tolist(),train_summed_mse=float(np.mean(np.sum((train_prediction-yy['train'])**2,axis=-1))),motion_constraint_adequate=bool(np.all(r2>=.5))))
            print(g,name,'R2',r2.tolist(),flush=True)
        bindings.append(dict(group=g,cache_report_sha256=sha(d/'report.json'),cache_acceptance_sha256=sha(d/'acceptance.json'),train_sha256=sha(d/'planner_train.npz'),validation_sha256=sha(d/'planner_validation.npz'),head_file=row['head_file'],head_sha256=row['head_sha256'],artifacts=artifacts,train_goals=len(np.unique(ids)),validation_goals=len(np.unique(vids))))
    result=dict(status='MOTION_PROBES_FIT_REQUIRES_RECONSTRUCTION',rows=rows,bindings=bindings,all_history_latent_probes_adequate=all(x['motion_constraint_adequate'] for x in rows if x['features']=='latent_history'),penalties=PENALTIES.tolist(),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
