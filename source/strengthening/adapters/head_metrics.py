"""Readout qualification uses observed validation strata only."""
import numpy as np


def metrics(prediction, truth, target_scale):
    pred=np.asarray(prediction,float);truth=np.asarray(truth,float)
    if pred.shape!=truth.shape or pred.shape[-1]!=6 or not np.isfinite(pred).all():raise ValueError('Invalid readout output')
    err=pred-truth
    angle=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(truth[:,4],truth[:,5])
    angle=np.arctan2(np.sin(angle),np.cos(angle))
    block=float(np.mean(np.sum(err[:,2:4]**2,axis=-1)));agent=float(np.mean(np.sum(err[:,:2]**2,axis=-1)))
    return dict(six_normalized_mse=float(np.mean((err/target_scale)**2)),block_position_mse=block,
                agent_position_mse=agent,block_rms_distance=float(np.sqrt(block)),agent_rms_distance=float(np.sqrt(agent)),
                wrapped_angle_mse=float(np.mean(angle**2)),wrapped_angle_mae=float(np.mean(abs(angle))),
                sin_mse=float(np.mean(err[:,4]**2)),cos_mse=float(np.mean(err[:,5]**2)),
                circle_norm_quantiles=np.quantile(np.linalg.norm(pred[:,4:6],axis=-1),[0,.05,.5,.95,1]).tolist(),
                frames=len(pred))


def gate(current, baseline):
    checks={}
    for domain in ['expert','planner']:
        checks[domain]={}
        for metric,factor in [('six_normalized_mse',.9),('block_position_mse',.9),
                              ('agent_position_mse',1.05),('wrapped_angle_mse',1.05)]:
            old=baseline[domain][metric];new=current[domain][metric]
            checks[domain][metric]=dict(passed=bool(old>0 and new<=factor*old),ratio=float(new/old) if old>0 else None)
    return dict(passed=all(v['passed'] for row in checks.values() for v in row.values()),checks=checks)


def nested_indices(domains, size, orders):
    """Balanced unique allocation where feasible, then fixed domain sampling weights."""
    n=min(int(size),len(domains));available={k:len(v) for k,v in orders.items()}
    planner=min(n//2,available['planner']);expert=min(n-planner,available['expert'])
    planner=min(n-expert,available['planner'])
    ids=np.r_[orders['expert'][:expert],orders['planner'][:planner]]
    if len(ids)!=n or len(np.unique(ids))!=n:raise ValueError('Insufficient unique frame inventory')
    return ids
