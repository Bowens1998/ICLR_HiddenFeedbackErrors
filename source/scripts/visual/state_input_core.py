"""Privileged-state diagnostic using the same GRU temporal core and six state tokens."""
from torch import nn
from factorial_model import make_model

def make_state_core(official,config,seed=3072):
    model=make_model(official,config,'gru_state',seed)
    model.encoder=nn.Identity();model.projector=nn.Identity()
    return model
