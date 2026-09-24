"""Fixed-ReLU-region convex projection; no claim of a global nonlinear fiber."""
import numpy as np
from scipy import sparse
import osqp


def geometry(head,token):
    x=(np.asarray(token,dtype=float)-head['mean'])/head['scale']
    w0,w1,w2=[head[f'{k}.weight'].astype(float) for k in (0,2,4)]
    b0,b1,b2=[head[f'{k}.bias'].astype(float) for k in (0,2,4)]
    a=w0@x+b0;m0=a>=0;b=w1@np.maximum(a,0)+b1;m1=b>=0
    second=w1@(m0[:,None]*w0);jac=w2@(m1[:,None]*second)
    return x,a,b,m0,m1,w0,second,jac


def readout(head,token):
    v=(np.asarray(token,dtype=float)-head['mean'])/head['scale']
    for k in (0,2,4):
        v=v@head[f'{k}.weight'].astype(float).T+head[f'{k}.bias'].astype(float)
        if k!=4:v=np.maximum(v,0)
    return v


def project(head,predicted,observed):
    x,a,b,m0,m1,w0,second,jac=geometry(head,predicted)
    target=(observed.astype(float)-head['mean'])/head['scale']-x
    # SVD avoids assuming full row rank or inverting J J^T.
    _,sing,vh=np.linalg.svd(jac,full_matrices=False);rank=int(np.sum(sing>sing.max()*1e-10))
    basis=vh[:rank];null=target-basis.T@(basis@target)
    signs=np.concatenate([np.where(m0,1.,-1.),np.where(m1,1.,-1.)]);pre=np.concatenate([a,b])
    inequality=signs[:,None]*np.concatenate([w0,second]);lower=-signs*pre
    norms=np.linalg.norm(inequality,axis=1);active=norms>1e-14
    assert np.all(lower[~active]<=1e-10)
    aa=np.concatenate([basis,inequality[active]/norms[active,None]])
    lo=np.concatenate([np.zeros(rank),lower[active]/norms[active]])
    hi=np.concatenate([np.zeros(rank),np.full(active.sum(),np.inf)])
    solver=osqp.OSQP();solver.setup(P=sparse.eye(len(x),format='csc'),q=-target,A=sparse.csc_matrix(aa),l=lo,u=hi,eps_abs=1e-9,eps_rel=1e-9,max_iter=20000,polish=True,verbose=False)
    result=solver.solve();d=result.x
    corrected=None if d is None else ((x+d)*head['scale']+head['mean']).astype(np.float32)
    naive=((x+null)*head['scale']+head['mean']).astype(np.float32)
    return corrected,naive,dict(status=result.info.status,iterations=int(result.info.iter),solver_seconds=float(result.info.run_time),rank=rank,primal_residual=float(result.info.pri_res),dual_residual=float(result.info.dua_res))
