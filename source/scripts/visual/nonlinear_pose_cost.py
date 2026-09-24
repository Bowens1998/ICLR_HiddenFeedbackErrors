"""Frozen PushT readout intervention; native latent rollout is unchanged.

Heads output standardized [agent_x, agent_y, block_x, block_y, sin, cos].
Target normalization must be fitted on training trajectories only.
"""
import numpy as np
import torch


def numpy_pose(tokens, head):
    x = (np.asarray(tokens, dtype=np.float64) - head['mean']) / head['scale']
    for i in (0, 2, 4):
        x = x @ head[f'{i}.weight'].T + head[f'{i}.bias']
        if i < 4:
            x = np.maximum(x, 0)
    return x * head['target_scale'] + head['target_mean']


def numpy_cost(endpoint, goal):
    endpoint, goal = np.asarray(endpoint), np.asarray(goal)
    angle = np.arctan2(endpoint[..., 4], endpoint[..., 5]) - np.arctan2(goal[..., 4], goal[..., 5])
    angle = np.angle(np.exp(1j * angle))
    return np.square(endpoint[..., 2:4] - goal[..., 2:4]).sum(-1) + 900 * angle**2


class NonlinearPoseCost:
    def __init__(self, latent, endpoint_head, goal_head):
        if latent.score_space != 'latent':
            raise ValueError('Requires a native latent rollout')
        self.latent_cost = latent
        self.endpoint = self.prepare(endpoint_head, latent.device)
        self.goal_head = self.prepare(goal_head, latent.device)
        self.goal_pose = self.forward(latent.goal, self.goal_head)

    @staticmethod
    def prepare(head, device):
        h = {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in head.items()}
        if h['target_mean'].shape != (6,) or h['target_scale'].shape != (6,):
            raise ValueError('Expected six pose targets')
        if any(not torch.isfinite(v).all() for v in h.values()):
            raise ValueError('Nonfinite readout')
        if not (h['scale'] > 0).all() or not (h['target_scale'] > 0).all():
            raise ValueError('Normalization scales must be positive')
        return h

    @staticmethod
    def forward(tokens, head):
        x = (tokens.to(torch.float64) - head['mean']) / head['scale']
        for i in (0, 2, 4):
            x = x @ head[f'{i}.weight'].T + head[f'{i}.bias']
            if i < 4:
                x = x.relu()
        return x * head['target_scale'] + head['target_mean']

    def __call__(self, actions):
        _, tokens = self.latent_cost(actions)
        pose = self.forward(tokens, self.endpoint)
        angle = torch.atan2(pose[..., 4], pose[..., 5]) - torch.atan2(self.goal_pose[4], self.goal_pose[5])
        angle = torch.atan2(angle.sin(), angle.cos())
        cost = (pose[..., 2:4] - self.goal_pose[2:4]).square().sum(-1) + 900 * angle.square()
        return cost, tokens

    @property
    def goal(self):
        return self.latent_cost.goal

    def verify_native(self, actions):
        return self.latent_cost.verify_native(actions)
