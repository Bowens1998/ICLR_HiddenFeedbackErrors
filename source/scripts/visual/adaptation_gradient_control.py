"""Prespecified per-step unit global gradient norm for matched-objective controls."""
import torch


def unit_global_gradient(parameters):
    gradients=[p.grad for p in parameters if p.grad is not None]
    if not gradients:raise ValueError('no gradients')
    if any(g.is_sparse or g.dtype!=torch.float32 for g in gradients):
        raise ValueError('requires dense float32 model gradients')
    if any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('nonfinite gradient; no gradient modified')
    norm=torch.sqrt(sum(g.detach().double().square().sum() for g in gradients))
    if not torch.isfinite(norm) or norm<=0:raise ValueError('invalid or zero global norm')
    with torch.no_grad():
        for g in gradients:g.copy_((g.double()/norm).to(g.dtype))
    after=torch.sqrt(sum(g.detach().double().square().sum() for g in gradients))
    if not torch.isfinite(after) or abs(float(after)-1.)>1e-6:
        raise AssertionError('unit gradient norm not achieved')
    return dict(before_norm=float(norm),after_norm=float(after))
