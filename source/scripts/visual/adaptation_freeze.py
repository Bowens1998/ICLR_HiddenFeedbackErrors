"""Explicit parameter/buffer boundary for matched dynamics-only continuation."""
import torch

TRAINABLE_ROOTS=('predictor','action_encoder','pred_proj')


def configure_dynamics_only(model):
    for name in ('encoder','projector',*TRAINABLE_ROOTS):
        if not hasattr(model,name):raise ValueError(f'missing model component {name}')
    # Both continuation branches use eval-mode forward arithmetic: no dropout
    # or BatchNorm running-buffer updates. Gradients remain enabled as requested.
    model.eval()
    names=[]
    for name,param in model.named_parameters():
        allow=name.split('.')[0] in TRAINABLE_ROOTS;param.requires_grad_(allow)
        if allow:names.append(name)
    if not names:raise ValueError('empty adaptation parameter set')
    allowed=set(names)
    frozen={name:value.detach().clone() for name,value in model.state_dict().items() if name not in allowed}
    return dict(trainable_names=names,frozen=frozen)


def verify_frozen(model,boundary):
    actual=[name for name,p in model.named_parameters() if p.requires_grad]
    if actual!=boundary['trainable_names']:raise AssertionError('trainable membership changed')
    if any(m.training for m in model.modules()):raise AssertionError('evaluation arithmetic mode changed')
    state=model.state_dict()
    if set(state)-set(actual)!=set(boundary['frozen']):raise AssertionError('state membership changed')
    for name,old in boundary['frozen'].items():
        new=state[name]
        if new.dtype!=old.dtype or new.shape!=old.shape or not torch.equal(
                new.detach().contiguous().reshape(-1).view(torch.uint8),
                old.contiguous().reshape(-1).view(torch.uint8)):
            raise AssertionError(f'frozen state changed: {name}')
