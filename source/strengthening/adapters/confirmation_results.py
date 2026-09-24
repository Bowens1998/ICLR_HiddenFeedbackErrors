"""Independent readout reconstruction and complete goal-cell storage for analysis."""
import numpy as np


def physical_pose(tokens, head):
    """Column-oriented FP64 implementation; does not call the rollout scorer."""
    a=np.asarray(tokens,dtype=np.float64);shape=a.shape[:-1]
    x=((a.reshape(-1,a.shape[-1])-head['mean'])/head['scale']).T
    h1=np.maximum(head['0.weight']@x+head['0.bias'][:,None],0)
    h2=np.maximum(head['2.weight']@h1+head['2.bias'][:,None],0)
    y=head['4.weight']@h2+head['4.bias'][:,None]
    return (y.T*head['target_scale']+head['target_mean']).reshape(*shape,6)


def block_error(pose, truth):
    return np.square(np.asarray(pose)[...,2:4]-np.asarray(truth)[...,2:4]).sum(-1)


class CompleteCells:
    def __init__(self):
        self.arrays={}

    def put(self,key,goal,group,stream,value,groups):
        value=np.asarray(value,dtype=np.float64)
        if not np.isfinite(value).all():raise ValueError('Nonfinite planned result '+key)
        if key not in self.arrays:
            self.arrays[key]=np.full((256,groups,4,*value.shape),np.nan)
        if self.arrays[key].shape!=(256,groups,4,*value.shape):raise ValueError('Changed result axes')
        cell=self.arrays[key][goal,group,stream]
        if np.isfinite(cell).all():
            np.testing.assert_array_equal(cell,value)
        elif not np.isnan(cell).all():raise ValueError('Partially filled result')
        else:self.arrays[key][goal,group,stream]=value

    def validate(self):
        if not self.arrays or not all(np.isfinite(x).all() for x in self.arrays.values()):
            raise ValueError('Incomplete fixed roster; complete-case subset is forbidden')


def expected_ac_members(kind,group):
    if kind in ['A_core','A_dual']:
        models=['A_'+q for q in ['latent','decoded_teacher','physical_labels']]
        constraints=['single_A'] if kind=='A_core' else ['single_A','dual_A_B']
    else:
        objective=kind.removeprefix('C_')
        if objective not in ['decoded_teacher','physical_labels'] or group%2:
            raise ValueError('Unplanned C family')
        models=[f'pool{group//2}_{objective}_{t}' for t in ['T0','T1','T2']];constraints=['single_A']
    return [f'{c}/{m}/{s}' for c in constraints for m in models for s in ['actual','donor']]


def expected_ac_families(group,cases):
    kinds=['A_core']+(['A_dual'] if group in [0,1] else [])
    if group in [0,2,4]:kinds+=['C_decoded_teacher','C_physical_labels']
    return [(s,i,k) for s in range(4) for i in cases for k in kinds]
