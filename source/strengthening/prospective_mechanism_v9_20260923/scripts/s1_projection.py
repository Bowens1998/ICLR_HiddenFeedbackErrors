"""Versioned S1 shared-coordinate QP and eight-member common-norm insertion.

QP algebra is retained from strengthening/adapters/projection.py. The new
wrapper permits exactly the S1 roster; original source files are untouched.
Only ReLU A/C heads enter this module. Reserved evaluator D is not an argument.
"""
from dataclasses import dataclass
import importlib.util
import numpy as np
from s1_common import ROOT
from s1_readout import relu_geometry

_spec=importlib.util.spec_from_file_location('_v9_original_independent_verifier',ROOT/'strengthening/adapters/verifier.py')
_verifier=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_verifier)
verify_token=_verifier.verify_token
MEMBERS=[dict(condition=c,objective=o,source=s) for c in ['A','AC']
         for o in ['decoded_teacher','physical_labels'] for s in ['actual','donor']]


class ProjectionFailure(RuntimeError):
    def __init__(self,message,detail):
        super().__init__(message);self.detail=detail


@dataclass
class Direction:
    delta: np.ndarray
    solver: dict


def project(heads,predicted,observed,reference_head=None,solver_factory=None):
    from scipy import sparse
    if solver_factory is None:
        import osqp
        solver_factory=osqp.OSQP;version=osqp.__version__
    else:
        version='injected_engineering_test_solver'
    if not heads:raise ValueError('At least one ReLU readout required')
    reference_head=heads[0] if reference_head is None else reference_head
    z=np.asarray(predicted,np.float64);obs=np.asarray(observed,np.float64)
    scale=np.asarray(reference_head['scale'],np.float64)
    if z.shape!=obs.shape or scale.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(obs).all() or not (scale>0).all():
        raise ValueError('Nonfinite or inconsistent projection inputs')
    target=(obs-z)/scale;jacs=[];inequalities=[];bounds=[]
    for head in heads:
        _,a,b,m0,m1,w0,second,jac=relu_geometry(head,z)
        ratio=scale/np.asarray(head['scale'],np.float64)
        if not np.isfinite(ratio).all() or not (ratio>0).all():raise ValueError('Invalid normalization')
        signs=np.r_[np.where(m0,1.,-1.),np.where(m1,1.,-1.)]
        jacs.append(jac*ratio)
        inequalities.append(signs[:,None]*np.concatenate([w0,second])*ratio)
        bounds.append(-signs*np.r_[a,b])
    jac=np.concatenate(jacs);_,sing,vh=np.linalg.svd(jac,full_matrices=False)
    rank=int(np.count_nonzero(sing>sing.max()*1e-10)) if sing.size and sing.max()>0 else 0
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
        osqp_version=version,head_count=len(heads),
        primal_residual=float(getattr(info,'prim_res',getattr(info,'pri_res',np.nan))),
        dual_residual=float(getattr(info,'dual_res',getattr(info,'dua_res',np.nan))))
    if str(info.status).lower()!='solved' or result.x is None or not np.isfinite(result.x).all():
        raise ProjectionFailure('QP did not solve; this is not a zero displacement',detail)
    return Direction(np.asarray(result.x,np.float64),detail)


def added_rank(head_A,head_C,token,relative_tolerance=1e-10,absolute_tolerance=0.):
    """One shared threshold for nested Jacobians in A displacement coordinates."""
    ja=relu_geometry(head_A,token)[-1]
    jc=relu_geometry(head_C,token)[-1]*(head_A['scale']/head_C['scale'])
    joint=np.concatenate([ja,jc]);sj=np.linalg.svd(joint,compute_uv=False);sa=np.linalg.svd(ja,compute_uv=False)
    threshold=max(absolute_tolerance,relative_tolerance*(float(sj.max()) if len(sj) else 0.))
    ra=int(np.count_nonzero(sa>threshold));rj=int(np.count_nonzero(sj>threshold))
    return dict(rank_A=ra,rank_A_plus_C=rj,added_rank=rj-ra,singular_values_A=sa.tolist(),
        singular_values_joint=sj.tolist(),threshold=threshold,relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,scope='Descriptive common-coordinate ranks; not a selection or exclusion gate')


def match_eight_family(tokens,directions,head_sets,reference_heads,member_ids=None,
                       shrink_factors=tuple(2.**-i for i in range(9)),tolerance=1e-6):
    if member_ids is None:member_ids=MEMBERS
    if member_ids!=MEMBERS or not(len(tokens)==len(directions)==len(head_sets)==len(reference_heads)==8):
        raise ValueError('The complete ordered eight-direction family is mandatory')
    if tuple(shrink_factors)!=tuple(2.**-i for i in range(9)) or tolerance!=1e-6:
        raise ValueError('Fixed S1 shrink/tolerance contract changed')
    for i,heads in enumerate(head_sets):
        if len(heads)!=(1 if i<4 else 2):raise ValueError('A-only must constrain only A; joint must constrain A/C')
        if heads[0] is not reference_heads[i]:raise ValueError('A must define the common displacement metric')
    # Each objective's actual/donor anchor is identical, and both conditions
    # intervene at the same free prediction, not a separately generated history.
    for j in [0,2]:
        for k in [j+1,j+4,j+5]:np.testing.assert_array_equal(tokens[j],tokens[k])
    ref=reference_heads[0]
    for r in reference_heads[1:]:
        for key in ['mean','scale','target_mean','target_scale','0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']:
            np.testing.assert_array_equal(ref[key],r[key])
    for heads in head_sets[5:]:
        for key in head_sets[4][1]:np.testing.assert_array_equal(head_sets[4][1][key],heads[1][key])
    norms=np.asarray([np.linalg.norm(d.delta) for d in directions],np.float64)
    if not np.isfinite(norms).all():raise ProjectionFailure('Nonfinite direction norm',{'stage':'matching'})
    target=float(norms.min());attempts=[]
    ranks=[dict(objective=MEMBERS[j]['objective'],**added_rank(ref,head_sets[4][1],tokens[j])) for j in [0,2]]
    for shrink in shrink_factors:
        corrected=[];checks=[]
        for token,direction,heads,reference,norm in zip(tokens,directions,head_sets,reference_heads,norms):
            alpha=0. if norm==0 else shrink*target/norm
            replacement=(np.asarray(token,np.float64)+alpha*direction.delta*reference['scale']).astype(np.float32)
            check=verify_token(token,replacement,heads,reference,expected_norm=shrink*target,tol=tolerance)
            corrected.append(replacement);checks.append(check)
        attempts.append(dict(shrink=float(shrink),checks=checks))
        if all(check['accepted'] for check in checks):
            return np.stack(corrected),dict(status='ACCEPTED_S1_EIGHT_MEMBER_FAMILY',family_size=8,members=MEMBERS,
                native_norms=norms.tolist(),unshrunk_common_norm=target,effective_norm=shrink*target,
                shrink_factor=float(shrink),legitimate_zero_norm=target==0,attempts=attempts,ranks=ranks,
                solvers=[d.solver for d in directions])
    raise ProjectionFailure('FP32 numerical acceptance failed for entire S1 family',
        dict(stage='verification',native_norms=norms.tolist(),attempts=attempts,ranks=ranks))


def project_eight_family(predicted,guides,head_A,head_C,projector=project):
    if set(predicted)!={'decoded_teacher','physical_labels'} or set(guides)!={'actual','donor'}:
        raise ValueError('Both fixed objectives and guidance sources are required')
    tokens=[];directions=[];heads=[]
    for member in MEMBERS:
        hs=[head_A] if member['condition']=='A' else [head_A,head_C]
        token=predicted[member['objective']]
        tokens.append(token);heads.append(hs)
        directions.append(projector(hs,token,guides[member['source']],reference_head=head_A))
    corrected,report=match_eight_family(tokens,directions,heads,[head_A]*8)
    return corrected,np.stack([d.delta for d in directions]),report
