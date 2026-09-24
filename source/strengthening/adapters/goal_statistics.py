"""Paired percentile bootstrap: average fixed models/streams within each goal first."""
import numpy as np
from contracts import namespace_seed


def goal_means(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim<1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('All planned goal rows must be present and finite')
    return values.reshape(len(values),-1).mean(axis=1)


def indices(root_seed,bank,goals,draws=20000):
    if bank not in ['confirmation_A_C','confirmation_B']:raise ValueError('Unknown confirmation population')
    if goals<2 or draws<1:raise ValueError('Invalid bootstrap dimensions')
    rng=np.random.default_rng(namespace_seed(root_seed,bank+'/goal_bootstrap'))
    return rng.integers(0,goals,size=(draws,goals),dtype=np.int32)


def paired_summary(left,right,draw_indices,family_size=9,alpha=.05,relative_mse=True):
    left=goal_means(left);right=goal_means(right)
    if left.shape!=right.shape:raise ValueError('Goal pairing mismatch')
    draws=np.asarray(draw_indices)
    if draws.ndim!=2 or draws.shape[1]!=len(left) or not np.issubdtype(draws.dtype,np.integer):
        raise ValueError('Resample goals, not model/stream rows')
    if np.any(draws<0) or np.any(draws>=len(left)) or family_size<1 or not 0<alpha<1:
        raise ValueError('Invalid confidence family or draw indices')
    delta=left-right;low=alpha/(2*family_size);quantiles=[low,1-low]
    boot_left=np.empty(len(draws));boot_right=np.empty(len(draws));boot_delta=np.empty(len(draws))
    for start in range(0,len(draws),500):
        ix=draws[start:start+500]
        boot_left[start:start+len(ix)]=left[ix].mean(1)
        boot_right[start:start+len(ix)]=right[ix].mean(1)
        boot_delta[start:start+len(ix)]=delta[ix].mean(1)
    mean_left=float(left.mean());mean_right=float(right.mean())
    relative=None
    if relative_mse and mean_right>0 and np.all(boot_right>0):
        relative=dict(estimate=(mean_left-mean_right)/mean_right,
            interval=np.quantile((boot_left-boot_right)/boot_right,quantiles,method='linear').tolist())
    top=np.argsort(-abs(delta),kind='stable')[:5]
    return dict(goals=len(left),draws=len(draws),family_size=family_size,nominal_interval_coverage=1-alpha/family_size,
        quantiles=quantiles,left_mean=mean_left,right_mean=mean_right,paired_change=float(delta.mean()),
        paired_interval=np.quantile(boot_delta,quantiles,method='linear').tolist(),
        relative_mse_change=relative,relative_status=('NOT_APPLICABLE' if not relative_mse else
            ('DEFINED' if relative is not None else 'UNDEFINED_NONPOSITIVE_COMPARATOR')),
        goal_change_quantiles=np.quantile(delta,[0,.05,.25,.5,.75,.95,1],method='linear').tolist(),
        largest_absolute_goal_contributions=[dict(goal_index=int(i),change=float(delta[i]),contribution_to_mean=float(delta[i]/len(delta))) for i in top],
        interpretation='Conditional on the fixed model roster, heads, donor bank and action-source rule; nominal percentile intervals, not finite-sample exact coverage.')
