"""Independent FP64 full nonlinear evaluation of actual FP32 inserted tokens.

Does not call the projection geometry, its Jacobian, or its solver flags.
"""
import numpy as np


def forward(head,token):
    x=(np.asarray(token,dtype=np.float64)-head['mean'])/head['scale']
    a=np.asarray(head['0.weight'],float)@x+head['0.bias']
    b=np.asarray(head['2.weight'],float)@np.maximum(a,0)+head['2.bias']
    y=np.asarray(head['4.weight'],float)@np.maximum(b,0)+head['4.bias']
    return y,a,b


def verify_token(original,replacement,heads,reference_head,expected_norm=None,tol=1e-6):
    if np.asarray(replacement).dtype!=np.float32:
        raise ValueError('Must verify the FP32 token actually inserted into the model')
    rows=[]
    for head in heads:
        y0,a0,b0=forward(head,original);y1,a1,b1=forward(head,replacement)
        output=float(np.max(np.abs(y1-y0)))
        region=float(max(0.,np.max(-np.where(a0>=0,1.,-1.)*a1),np.max(-np.where(b0>=0,1.,-1.)*b1)))
        physical=np.max(np.abs((y1-y0)*head['target_scale']))
        rows.append(dict(normalized_output_deviation=output,region_violation=region,
                         physical_output_max_deviation=float(physical)))
    actual_norm=float(np.linalg.norm((np.asarray(replacement,float)-np.asarray(original,float))/reference_head['scale']))
    finite=bool(np.isfinite(replacement).all() and np.isfinite(actual_norm) and
                all(np.isfinite(list(row.values())).all() for row in rows))
    norm_ok=expected_norm is None or bool(np.isclose(actual_norm,expected_norm,atol=tol,rtol=tol))
    return dict(accepted=finite and norm_ok and all(row['normalized_output_deviation']<=tol and row['region_violation']<=tol for row in rows),
                heads=rows,actual_norm=actual_norm,expected_norm=expected_norm,norm_ok=norm_ok,finite=finite)
