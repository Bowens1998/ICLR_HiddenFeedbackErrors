"""Single/dual readout fixed-region projection in a shared standardized space.

Legacy head geometry is reused. Independent acceptance lives in verifier.py.
No outcome metric enters projection, norm matching, or numerical shrinking.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/visual'))


class ProjectionFailure(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


@dataclass
class Direction:
    delta: np.ndarray  # In shared standardized coordinates; FP64.
    solver: dict


def project(heads, predicted, observed, reference_head=None, solver_factory=None):
    # GPU rollout imports matching/Direction but does not run a solver. Keep the
    # pinned CPU solver dependencies local to actual QP execution.
    import osqp
    from scipy import sparse
    from readout_fiber import geometry
    if solver_factory is None:solver_factory=osqp.OSQP
    if not heads:
        raise ValueError('At least one readout required')
    reference_head = heads[0] if reference_head is None else reference_head
    z=np.asarray(predicted,dtype=np.float64);obs=np.asarray(observed,dtype=np.float64)
    scale=np.asarray(reference_head['scale'],dtype=np.float64)
    if z.shape!=obs.shape or scale.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(obs).all() or not (scale>0).all():
        raise ValueError('Nonfinite or inconsistent projection inputs')
    target=(obs-z)/scale;jacs=[];inequalities=[];bounds=[]
    for head in heads:
        _,a,b,m0,m1,w0,second,jac=geometry(head,z)
        ratio=scale/np.asarray(head['scale'],dtype=float)
        if not np.isfinite(ratio).all() or not (ratio>0).all():raise ValueError('Invalid normalization')
        signs=np.r_[np.where(m0,1.,-1.),np.where(m1,1.,-1.)]
        jacs.append(jac*ratio)
        inequalities.append(signs[:,None]*np.concatenate([w0,second])*ratio)
        bounds.append(-signs*np.r_[a,b])
    jac=np.concatenate(jacs);_,s,vh=np.linalg.svd(jac,full_matrices=False)
    rank=int(np.count_nonzero(s>s.max()*1e-10)) if s.size and s.max()>0 else 0
    eq=vh[:rank];ineq=np.concatenate(inequalities);lower=np.concatenate(bounds)
    row_norm=np.linalg.norm(ineq,axis=1);active=row_norm>1e-14
    if np.any(lower[~active]>1e-10):raise ProjectionFailure('Inconsistent zero inequality row',{'stage':'assembly'})
    matrix=np.concatenate([eq,ineq[active]/row_norm[active,None]])
    lo=np.r_[np.zeros(rank),lower[active]/row_norm[active]]
    hi=np.r_[np.zeros(rank),np.full(active.sum(),np.inf)]
    solver=solver_factory()
    try:
        solver.setup(P=sparse.eye(len(z),format='csc'),q=-target,A=sparse.csc_matrix(matrix),
                     l=lo,u=hi,eps_abs=1e-9,eps_rel=1e-9,max_iter=20000,polish=True,verbose=False)
        result=solver.solve()
    except Exception as exc:
        raise ProjectionFailure('QP raised an exception',{'stage':'solver','exception':repr(exc)}) from exc
    info=result.info
    detail=dict(status=info.status,iterations=int(info.iter),seconds=float(info.run_time),rank=rank,
                osqp_version=osqp.__version__,head_count=len(heads),
                primal_residual=float(getattr(info,'prim_res',getattr(info,'pri_res',np.nan))),
                dual_residual=float(getattr(info,'dual_res',getattr(info,'dua_res',np.nan))))
    if str(info.status).lower()!='solved' or result.x is None or not np.isfinite(result.x).all():
        raise ProjectionFailure('QP did not solve; this is not a zero displacement',detail)
    return Direction(np.asarray(result.x,dtype=np.float64),detail)


def match_family(tokens, directions, head_sets, reference_heads=None, shrink_factors=tuple(2.**-i for i in range(9))):
    """Match the prespecified B pair or A/C six/twelve directions; retain zeros."""
    from verifier import verify_token
    if len(tokens) not in [2,6,12] or not(len(tokens)==len(directions)==len(head_sets)):
        raise ValueError('Expected exactly two, six or twelve family members')
    reference_heads=[h[0] for h in head_sets] if reference_heads is None else reference_heads
    if len(reference_heads)!=len(tokens):raise ValueError('Normalizer count mismatch')
    norms=np.array([np.linalg.norm(d.delta) for d in directions]);target=float(norms.min())
    if not np.isfinite(norms).all():raise ProjectionFailure('Nonfinite norm',{'stage':'matching'})
    attempts=[]
    for shrink in shrink_factors:
        corrected=[];checks=[]
        for token,direction,heads,ref,norm in zip(tokens,directions,head_sets,reference_heads,norms):
            alpha=0. if norm==0 else shrink*target/norm
            replacement=(np.asarray(token,dtype=np.float64)+alpha*direction.delta*ref['scale']).astype(np.float32)
            check=verify_token(token,replacement,heads,ref,expected_norm=shrink*target)
            corrected.append(replacement);checks.append(check)
        attempts.append(dict(shrink=float(shrink),checks=checks))
        if all(check['accepted'] for check in checks):
            return np.stack(corrected),dict(status='ACCEPTED_MATCHED_FAMILY',family_size=len(tokens),
                native_norms=norms.tolist(),unshrunk_common_norm=target,effective_norm=shrink*target,
                shrink_factor=float(shrink),legitimate_zero_norm=target==0,attempts=attempts)
    raise ProjectionFailure('FP32 insertion fails numerical acceptance for entire family',
                            dict(stage='verification',attempts=attempts,native_norms=norms.tolist()))
