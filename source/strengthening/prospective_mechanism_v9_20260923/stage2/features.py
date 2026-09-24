"""Draft S2 scalar extraction from complete, already decoded physical poses.

This module loads no model, readout, file or response. Its caller must first
authenticate q_g, each head's own normalization, raw inputs, all four-direction
families, and the complete split. These arithmetic functions do not establish
that provenance or authorize test-response access.
"""
import numpy as np

GROUPS = (0, 2, 4)
OBJECTIVES = ('decoded_teacher', 'physical_labels')
BRANCHES = ('free', 'actual', 'donor', 'reset')
CONDITIONS = ('T0', 'T1')
HORIZONS = (5, 10, 15, 20, 25)
POSE = ('agent_x', 'agent_y', 'block_x', 'block_y', 'sin_theta', 'cos_theta')
STRATA = tuple(f'pool{p}/{o}' for p in range(3) for o in OBJECTIVES)
ORDINARY = ('free_error_5', 'free_error_25', 'observed_history_error_25', 'common_norm_squared')


def _array(value, shape, name):
    a = np.asarray(value)
    if a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: complete finite FP64 physical-coordinate array {shape} required')
    return a


def _axes(axes, *, response=False):
    expected = dict(groups=list(GROUPS), objectives=list(OBJECTIVES), horizons=list(HORIZONS), pose=list(POSE))
    expected['conditions' if response else 'branches'] = list(CONDITIONS if response else BRANCHES)
    if axes != expected:
        raise ValueError('Missing, reordered or additional scientific axes')


def _size(split):
    if split not in ('calibration', 'test'):
        raise ValueError('Only complete calibration or test splits are supported')
    return 256 if split == 'calibration' else 512


def _block_error(prediction, truth):
    # Coordinates 0:2 are the agent, not the block. No target-scale weighting.
    with np.errstate(over='raise', invalid='raise'):
        error = np.sum(np.square(prediction[..., 2:4] - truth[..., 2:4]), axis=-1, dtype=np.float64)
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite block error blocks the complete population')
    return error


def probe_features(prediction, observed_history_prediction, truth, effective_common_norm, *, split, axes):
    """Physical q_g poses: N×pool×objective×branch×horizon×pose.

    Truth is N×pool×horizon×pose, shared by both objectives. The norm is the
    post-backoff scalar for a complete four-member family, N×pool. Neither
    realized FP32 per-direction norms nor raw minimum QP norms are substitutes.
    """
    _axes(axes); n = _size(split)
    p = _array(prediction, (n, 3, 2, 4, 5, 6), 'probe prediction')
    o = _array(observed_history_prediction, (n, 3, 2, 5, 6), 'observed-history prediction')
    t = _array(truth, (n, 3, 5, 6), 'probe truth')
    m = _array(effective_common_norm, (n, 3), 'post-backoff family norm')
    if np.any(m < 0):
        raise ValueError('A displacement norm cannot be negative')
    np.testing.assert_array_equal(o[..., 0, :], p[:, :, :, 0, 0, :])
    errors = _block_error(p, t[:, :, None, None, :, :])
    observed_errors = _block_error(o, t[:, :, None, :, :])
    with np.errstate(over='raise', invalid='raise'):
        dose = np.broadcast_to(np.square(m)[..., None], (n, 3, 2))
        ordinary = np.stack([errors[..., 0, 0], errors[..., 0, -1], observed_errors[..., -1], dose], axis=-1)
        signed_g = errors[..., 2, -1] - errors[..., 1, -1]
    if not np.isfinite(ordinary).all() or not np.isfinite(signed_g).all():
        raise ValueError('Nonfinite scalar extraction blocks the complete population')
    # Do not replace a small G by zero, even in a declared zero-dose family.
    return dict(ordinary=ordinary.reshape(n, 6, 4), signed_g=signed_g.reshape(n, 6),
                block_errors=errors.reshape(n, 6, 4, 5),
                observed_history_block_errors=observed_errors.reshape(n, 6, 5),
                stratum_ids=STRATA, ordinary_columns=ORDINARY,
                error_units='position_squared', norm_units='g_A_standardized_latent_norm',
                input_provenance_verified_by_this_module=False)


def response_values(prediction, truth, *, split, axes):
    """Free q_g poses N×pool×objective×condition×horizon×pose.

    A test caller must pass the independent prediction-seal-first response
    access gate before these decoded predictions may exist. This function
    itself performs arithmetic only and must not be used as that access gate.
    """
    _axes(axes, response=True); n = _size(split)
    p = _array(prediction, (n, 3, 2, 2, 5, 6), 'response prediction')
    t = _array(truth, (n, 3, 5, 6), 'response truth')
    errors = _block_error(p, t[:, :, None, None, :, :])
    y = errors[..., 0, -1] - errors[..., 1, -1]
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite response blocks the complete population')
    return dict(response=y.reshape(n, 6), block_errors=errors.reshape(n, 6, 2, 5),
                stratum_ids=STRATA, response_units='position_squared',
                response_forecast_mse_units='position_to_the_fourth',
                response_access_verified_by_this_module=False)
