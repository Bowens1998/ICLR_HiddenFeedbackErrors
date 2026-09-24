"""Disjoint training pools with nested sizes, independent of task outcomes."""
import numpy as np

def select_pools(lengths, excluded, seed=956001, replicas=3, small=256, large=1024):
    lengths=np.asarray(lengths)
    if not 0 < small <= large or replicas < 1:raise ValueError('invalid pool sizes')
    eligible=np.flatnonzero((lengths>=40)&~np.isin(np.arange(len(lengths)),list(excluded)))
    if len(eligible)<replicas*large:raise ValueError('insufficient eligible episodes')
    order=np.random.default_rng(seed).permutation(eligible)
    return [{small:np.sort(order[r*large:r*large+small]),large:np.sort(order[r*large:(r+1)*large])} for r in range(replicas)]
