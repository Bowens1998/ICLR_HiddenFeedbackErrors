"""State-supervised control with an unconstrained 192-dimensional recurrent token."""
import torch
from torch import nn
from factorial_model import make_model as make_base,state_features
ARMS=['transformer_latent_state','gru_latent_state']

def attach_state_head(model,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+960000)
        model.state_head=nn.Sequential(nn.Linear(192,256),nn.GELU(),nn.Linear(256,6))
    return model

def make_model(official,config,arm,seed):
    if arm not in ARMS:raise ValueError(arm)
    base=make_base(official,config,arm.replace('_latent_state','_jepa'),seed)
    return attach_state_head(base,seed)

def state_losses(model,predicted_tokens,observed_tokens,targets):
    prediction=(model.state_head(predicted_tokens)-targets[:,1:]).square().mean()
    observation=(model.state_head(observed_tokens)-targets).square().mean()
    return prediction,observation
