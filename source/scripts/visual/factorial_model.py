"""Separate temporal architecture from prediction target without changing history."""
import torch
from torch import nn
from lewm_adapter import build

ARMS=['transformer_jepa','gru_jepa','transformer_state','gru_state']

class GRUPredictor(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(384,384,2,batch_first=True,dropout=.1);self.output=nn.Linear(384,192)
    def forward(self,x,action):
        y,_=self.gru(torch.cat([x,action],-1));return self.output(y)

class StateInput(nn.Module):
    def __init__(self,core):
        super().__init__();self.lift=nn.Linear(6,192);self.core=core
    def forward(self,x,action):return self.core(self.lift(x),action)

def make_model(official,config,arm,seed):
    assert arm in ARMS
    # Construct identical common parameters before adding any arm-specific modules.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=build(official,config)
        torch.manual_seed(seed+910000)
        if arm.startswith('gru'):model.predictor=GRUPredictor()
        if arm.endswith('state'):
            from module import MLP
            torch.manual_seed(seed+920000)
            model.projector=MLP(input_dim=192,output_dim=6,hidden_dim=2048,norm_fn=nn.BatchNorm1d)
            model.pred_proj=MLP(input_dim=192,output_dim=6,hidden_dim=2048,norm_fn=nn.BatchNorm1d)
            model.predictor=StateInput(model.predictor)
    return model

def state_features(state):
    """Four positions and circular orientation; velocity is not a single-frame target."""
    return torch.cat([state[...,:4],state[...,4:5].sin(),state[...,4:5].cos()],-1)
