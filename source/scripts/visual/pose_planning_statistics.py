"""Conditional paired-goal contrasts for all48 PushT development routes."""
import numpy as np

INTERFACES=['latent','pose_encoded','pose_predicted','state']


def route(replica,arch,interface,algorithm):
    return ((replica*2+arch)*4+INTERFACES.index(interface))*2+['random','cem'].index(algorithm)


def reversal_status(latent,pose):
    if latent['mean']<=0 or pose['mean']>=0:return 'direction_not_reproduced'
    if latent['ci95'][0]>0 and pose['ci95'][1]<0:return 'supported_conditional'
    return 'inconclusive_interval'


def analyze(cost,success,draws=10000):
    cost=np.asarray(cost,dtype=float);success=np.asarray(success,dtype=float)
    if cost.shape!=(48,128) or success.shape!=cost.shape or not np.isfinite(cost).all() or not np.isin(success,[0,1]).all():raise ValueError('Requires complete finite48x128 cost and binary success')
    ix=np.random.default_rng(1248901).integers(0,128,(draws,128));rows=[];effects={};claims=[]
    def add(name,terms,scope,replica=None,arch=None):
        result={}
        for metric,array in [('cost',cost),('success',success)]:
            d=sum(w*array[r] for w,r in terms);boot=d[ix].mean(-1)
            row=dict(contrast=name,scope=scope,replica=replica,arch=arch,metric=metric,mean=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),negative_cases=int((d<0).sum()),positive_cases=int((d>0).sum()),ties=int((d==0).sum()),terms=[dict(weight=w,route_index=r) for w,r in terms])
            rows.append(row);result[metric]=row
        return result
    primary={k:[] for k in ['latent_cem_minus_random','pose_cem_minus_random','interaction','pose_minus_latent_random','pose_minus_latent_cem']}
    for rep in range(3):
        for arch in range(2):
            r=lambda inter,alg:route(rep,arch,inter,alg)
            lat=[(1,r('latent','cem')),(-1,r('latent','random'))];cir=[(1,r('pose_encoded','cem')),(-1,r('pose_encoded','random'))]
            specs=dict(latent_cem_minus_random=lat,pose_cem_minus_random=cir,interaction=cir+[(-w,i) for w,i in lat])
            for alg in ['random','cem']:specs['pose_minus_latent_'+alg]=[(1,r('pose_encoded',alg)),(-1,r('latent',alg))]
            result={}
            for name,terms in specs.items():
                result[name]=add(name,terms,'primary',rep,arch);primary[name].extend([(w/6,i) for w,i in terms])
            claims.append(dict(replica=rep,arch=arch,status=reversal_status(result['latent_cem_minus_random']['cost'],result['pose_cem_minus_random']['cost'])))
            for alg in ['random','cem']:
                for left,right in [('pose_predicted','pose_encoded'),('pose_encoded','state'),('pose_predicted','state')]:add(left+'_minus_'+right+'_'+alg,[(1,r(left,alg)),(-1,r(right,alg))],'secondary',rep,arch)
    pooled={name:add(name,terms,'primary_equal_backbone_mean') for name,terms in primary.items()}
    claims.append(dict(replica=None,arch=None,status=reversal_status(pooled['latent_cem_minus_random']['cost'],pooled['pose_cem_minus_random']['cost'])))
    for rep in range(3):
        for inter in INTERFACES:
            for alg in ['random','cem']:add('transformer_minus_gru_'+inter+'_'+alg,[(1,route(rep,0,inter,alg)),(-1,route(rep,1,inter,alg))],'secondary_architecture',rep)
    assert len(rows)==190
    return dict(contrasts=rows,reversal_claims=claims,bootstrap=dict(draws=draws,seed=1248901,shared_goal_indices=True,scope='Nominal conditional goal intervals, not training-population uncertainty or adjusted secondary significance'))
