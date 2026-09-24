"""C: fixed two-layer ReLU; D: exact reused smooth residual GELU architecture."""
import importlib.util
import numpy as np
from s1_common import ARCHITECTURES, ROOT

_V8_PATH = ROOT / 'strengthening/presubmission_v8_20260922/independent_readout/scripts/readout.py'
_spec = importlib.util.spec_from_file_location('_v9_frozen_v8_readout', _V8_PATH)
_v8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v8)
pose_errors = _v8.pose_errors
qualification_gate = _v8.qualification_gate
serialize_model = _v8.serialize_model


def build_model(role):
    if role == 'D':
        return _v8.build_model(ARCHITECTURES['D'])
    if role != 'C':
        raise ValueError('Unknown fitted head role')
    from torch import nn
    return nn.Sequential(nn.Linear(192, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 6))


def validate_relu(head):
    sizes = [len(head['mean']), len(head['0.bias']), len(head['2.bias']), len(head['4.bias'])]
    for k, a, b in zip([0, 2, 4], sizes[:-1], sizes[1:]):
        if np.shape(head[f'{k}.weight']) != (b, a) or np.shape(head[f'{k}.bias']) != (b,):
            raise ValueError('Invalid ReLU serialized layer layout')
    if np.shape(head['scale']) != (sizes[0],) or np.shape(head['target_scale']) != (sizes[-1],) or np.shape(head['target_mean']) != (sizes[-1],):
        raise ValueError('Invalid normalizer dimensions')
    if not all(np.isfinite(v).all() for v in head.values()) or not (head['scale'] > 0).all() or not (head['target_scale'] > 0).all():
        raise ValueError('Nonfinite or invalid ReLU head')


def relu_forward(head, tokens, physical=False):
    validate_relu(head)
    x = (np.asarray(tokens, np.float64) - head['mean']) / head['scale']
    for k in [0, 2, 4]:
        x = x @ np.asarray(head[f'{k}.weight'], np.float64).T + head[f'{k}.bias']
        if k != 4:
            x = np.maximum(x, 0)
    return x * head['target_scale'] + head['target_mean'] if physical else x


def numpy_forward(tokens, head, role):
    if role == 'D':
        return _v8.numpy_forward(tokens, head, ARCHITECTURES['D'])
    if role == 'C':
        return relu_forward(head, tokens, physical=True)
    raise ValueError('Unknown fitted head role')


def relu_geometry(head, token):
    """Same fixed-region algebra as scripts/visual/readout_fiber.py:geometry."""
    validate_relu(head)
    x = (np.asarray(token, np.float64) - head['mean']) / head['scale']
    w0, w1, w2 = [np.asarray(head[f'{k}.weight'], np.float64) for k in [0, 2, 4]]
    a = w0 @ x + head['0.bias']; m0 = a >= 0
    b = w1 @ np.maximum(a, 0) + head['2.bias']; m1 = b >= 0
    second = w1 @ (m0[:, None] * w0)
    jac = w2 @ (m1[:, None] * second)
    return x, a, b, m0, m1, w0, second, jac
