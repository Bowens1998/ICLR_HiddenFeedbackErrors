"""Paired plans with identical block means and bounded within-block variation."""
import numpy as np

def paired_plans(rng,groups=5,steps=5,mean_limit=.25,action_limit=.35,jitter_std=.1):
    means=rng.uniform(-mean_limit,mean_limit,(groups,1,2))
    residual=rng.normal(0,jitter_std,(groups,steps,2));residual-=residual.mean(1,keepdims=True)
    ratios=np.where(residual>0,(action_limit-means)/np.maximum(residual,1e-300),(action_limit+means)/np.maximum(-residual,1e-300))
    scale=np.minimum(1,ratios.min((1,2)))[:,None,None]
    held=np.broadcast_to(means,(groups,steps,2)).copy().astype('float32');varied=(means+residual*scale).astype('float32')
    return held.reshape(-1,2),varied.reshape(-1,2)
