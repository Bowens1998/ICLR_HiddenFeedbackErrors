"""Independent NumPy reconstruction for the fixed negative-slope intervention."""
import numpy as np


def numpy_leaky_pose(tokens,head):
    x=(np.asarray(tokens,dtype=np.float64)-head['mean'])/head['scale']
    for i in (0,2,4):
        x=x@head[f'{i}.weight'].T+head[f'{i}.bias']
        if i<4:x=np.where(x>=0,x,.01*x)
    return x*head['target_scale']+head['target_mean']
