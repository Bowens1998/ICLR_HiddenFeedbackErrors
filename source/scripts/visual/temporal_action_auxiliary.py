"""Mechanism intervention, not a novel action representation method.

Keep head capacity/init and original modes; isolate orthogonal temporal errors.
"""
import torch
from action_auxiliary import ActionAuxiliary


def temporal_components(prediction, target):
    if prediction.shape != target.shape or prediction.shape[-1] != 10:
        raise ValueError('Expected matching five-by-two flattened action blocks')
    error = (prediction - target).reshape(*target.shape[:-1], 5, 2)
    average = error.mean(-2, keepdim=True)
    return average.square().mean(), (error - average).square().mean()


class TemporalActionAuxiliary(ActionAuxiliary):
    def losses(self, observed, actions):
        parts = super().losses(observed, actions)
        pred = self.inverse(torch.cat([observed[:, :3], observed[:, 1:]], -1))
        parts['inverse_mean'], parts['inverse_variation'] = temporal_components(pred, actions)
        return parts

    def objective(self, observed, actions, mode):
        if mode in ['none', 'inverse', 'inverse_goal']:
            return super().objective(observed, actions, mode)
        if mode not in ['inverse_mean', 'inverse_variation', 'inverse_half']:
            raise ValueError(mode)
        parts = self.losses(observed, actions)
        key = 'inverse' if mode == 'inverse_half' else mode
        coefficient = .05 if mode == 'inverse_half' else .1
        return coefficient * parts[key], parts
