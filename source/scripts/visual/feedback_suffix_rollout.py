"""Continue a shared feedback state at t=5 under a fixed 20-action pool.

This interface accepts actions and latent history only. Physical outcomes and
candidate-specific future images have no input path to the scorer.
"""
import torch


def rollout_suffixes(model, initial_history, feedback, past_actions, suffixes,
                     action_mean, action_std):
    """Return [horizon, candidate, latent] at t=10,15,20,25.

    initial_history contains E(o_-10), E(o_-5), E(o_0). past_actions
    contains a_-5,...,a_4, so the first aligned window is U_5, not U_0.
    feedback is the ONE shared z_5, broadcast across all candidate futures.
    """
    if initial_history.ndim != 2 or initial_history.shape[0] != 3:
        raise ValueError('Expected three initial observation tokens')
    if feedback.shape != initial_history.shape[1:]:
        raise ValueError('Expected one shared feedback token')
    if past_actions.shape != (10, 2):
        raise ValueError('Expected the two action blocks preceding t=5')
    if suffixes.ndim != 3 or suffixes.shape[1:] != (20, 2):
        raise ValueError('Expected a candidate pool with twenty future actions')
    count = len(suffixes)
    all_actions = torch.cat([past_actions[None].expand(count, -1, -1), suffixes], 1)
    blocks = all_actions.reshape(count, 6, 10)
    encoded = model.action_encoder((blocks - action_mean.repeat(5)) / action_std.repeat(5))
    shared = torch.cat([initial_history[1:], feedback[None]], 0)
    history = shared[None].expand(count, -1, -1).clone()
    tokens = []
    for step in range(4):
        token = model.predict(history[:, -3:], encoded[:, step:step + 3])[:, -1:]
        history = torch.cat([history, token], 1)
        tokens.append(token[:, 0])
    return torch.stack(tokens)
