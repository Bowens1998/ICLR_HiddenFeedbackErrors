"""Independent float64 evaluation of the fixed Linear-GELU-Linear state head."""
import numpy as np
from scipy.special import erf

def decode_head(tokens,weights):
    x=np.asarray(tokens,dtype=np.float64)
    w0,b0,w1,b1=[np.asarray(weights[k],dtype=np.float64) for k in ['0.weight','0.bias','2.weight','2.bias']]
    assert x.shape[-1]==192 and w0.shape==(256,192) and b0.shape==(256,) and w1.shape==(6,256) and b1.shape==(6,)
    hidden=x@w0.T+b0
    hidden=.5*hidden*(1+erf(hidden/np.sqrt(2)))
    return hidden@w1.T+b1
