"""DA-style action supervision adaptation; not an exact published reproduction.

Four observed tokens at raw offsets 0,5,10,15; three normalized action
blocks connecting them. No predicted tokens, simulator states or test goals.
LayerNorm follows each hidden Linear and precedes GELU, an explicit local
choice because the inspected appendix does not specify operator ordering.
"""
import torch
from torch import nn


class ActionAuxiliary(nn.Module):
    def __init__(self, latent_dim=192, action_dim=10, seed=967001):
        super().__init__();self.latent_dim=latent_dim;self.action_dim=action_dim
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            def head():
                return nn.Sequential(nn.Linear(2*latent_dim,256),nn.LayerNorm(256),nn.GELU(),
                                     nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,action_dim))
            self.inverse=head();self.goal=head()

    def losses(self, observed, actions):
        if observed.ndim!=3 or observed.shape[1:]!=(4,self.latent_dim):
            raise ValueError('Expected B x 4 x latent_dim observed image tokens')
        if actions.shape!=(len(observed),3,self.action_dim):
            raise ValueError('Expected exactly three intervening normalized action blocks')
        current=observed[:,:3]
        next_observed=observed[:,1:]
        goal=observed[:,-1:].expand(-1,3,-1)
        inverse=self.inverse(torch.cat([current,next_observed],-1))
        goal_action=self.goal(torch.cat([current,goal],-1))
        return {'inverse':(inverse-actions).square().mean(),'goal':(goal_action-actions).square().mean()}

    def objective(self, observed, actions, mode):
        if mode not in ['none','inverse','inverse_goal']:raise ValueError(mode)
        losses=self.losses(observed,actions)
        # Retain identical head initialization/capacity; inactive losses carry no gradient.
        total=observed.new_zeros(())
        if mode!='none':total=total+.1*losses['inverse']
        if mode=='inverse_goal':total=total+.1*losses['goal']
        return total,losses
