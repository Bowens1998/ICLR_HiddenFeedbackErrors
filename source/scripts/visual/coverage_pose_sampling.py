"""Outcome-independent samples for a fixed-label PushT coverage intervention."""
import numpy as np


def expert_indices(count,labels,seed):
    if count<labels:raise ValueError('Not enough distinct expert endpoints')
    return np.random.default_rng(seed).choice(count,labels,replace=False)


def broad_proposals(seed,attempts,position_bounds=(8.,504.)):
    rng=np.random.default_rng(seed)
    for _ in range(attempts):
        positions=rng.uniform(*position_bounds,size=4)
        angle=rng.uniform(0,2*np.pi)
        yield np.r_[positions,angle,0.,0.]
