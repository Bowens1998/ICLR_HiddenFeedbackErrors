"""S1 immutable I/O, role guards and fixed scientific head configuration."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
STUDY_ID = 'prospective_mechanism_v9_s1_20260923'
GROUPS = [0, 1]
HEAD_ROLES = ['C', 'D']
OLD_ROLES = ['fit', 'validation', 'qualification']
DOMAINS = ['expert', 'planner']
PARENT_COUNTS = {'expert': {'fit': 1024, 'validation': 256, 'qualification': 256},
                 'planner': {'fit': 1024, 'validation': 512, 'qualification': 512}}
ARCHITECTURES = {
    'C': dict(name='relu_512_two_hidden_v1', input_dim=192, width=512, output_dim=6),
    'D': dict(name='gelu_residual_256_two_blocks_v1', input_dim=192, output_dim=6,
              width=256, blocks=2, residual_factor=.5, gelu_approximation='none', skip_bias=False)}
FIT = dict(updates=8000, batch_size=256, expert_samples_per_batch=128,
           validation_interval=250, learning_rate=1e-4, normalizer_std_floor=1e-6,
           seed_namespace='v9_s1_fit/{role}/group_{group}')
SPLIT_NAMESPACE = 'v9_s1_split/{domain}/{old_role}/{parent_id}'
METRICS = ['six_normalized_mse', 'block_position_mse', 'agent_position_mse', 'wrapped_angle_mse']
SOURCE_CACHE_PROTOCOL = 'fafd933d37ef0ab5af9b6ec90ad424f6d6ab4563259b88239e398d26da9daec7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def checked_json(path, expected):
    if not expected or sha(path) != expected:
        raise ValueError(f'Unbound or changed JSON: {path}')
    return json.loads(Path(path).read_text())


def namespace_seed(root, namespace):
    return int.from_bytes(hashlib.sha256(f'{root}:{namespace}'.encode()).digest()[:4], 'big')


def validate_protocol(cfg):
    fixed = {'study_id': STUDY_ID, 'root_seed': 20260923, 'groups': GROUPS,
             'head_roles': HEAD_ROLES, 'source_cache_protocol_sha256': SOURCE_CACHE_PROTOCOL,
             'head_design': ARCHITECTURES, 'split_seed_namespace': SPLIT_NAMESPACE}
    for key, value in fixed.items():
        if cfg.get(key) != value:
            raise ValueError(f'Unsupported or changed S1 design field: {key}')
    for key, value in FIT.items():
        if cfg.get('fit', {}).get(key) != value:
            raise ValueError(f'Changed S1 fit field: {key}')
    q = cfg.get('qualification', {})
    if q.get('max_ratio_to_g_A') != 1.1 or q.get('domains') != DOMAINS or q.get('metrics') != METRICS:
        raise ValueError('Changed S1 qualification contract')
    return cfg


def load_protocol(path, expected):
    return validate_protocol(checked_json(path, expected))


def view_role(head_role, old_role):
    if head_role not in HEAD_ROLES or old_role not in OLD_ROLES:
        raise ValueError('Unknown head or data role')
    return ('pose_constraint_' if head_role == 'C' else 'reserved_evaluator_') + old_role


def require_view(meta, head_role, old_role, protocol_sha, group, domain):
    expected = dict(role=view_role(head_role, old_role), head_role=head_role,
                    old_role=old_role, protocol_sha256=protocol_sha, group=group, domain=domain)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Forbidden, changed or cross-head data role')


def mixed_moments(expert, planner, floor=1e-6):
    expert, planner = np.asarray(expert, np.float64), np.asarray(planner, np.float64)
    if expert.ndim != 2 or planner.ndim != 2 or expert.shape[1] != planner.shape[1] or not len(expert) or not len(planner):
        raise ValueError('Both finite observed domains are mandatory')
    if not np.isfinite(expert).all() or not np.isfinite(planner).all():
        raise ValueError('Nonfinite fitting data')
    mean = .5 * (expert.mean(0) + planner.mean(0))
    second = .5 * ((expert * expert).mean(0) + (planner * planner).mean(0))
    return mean, np.maximum(np.sqrt(np.maximum(second - mean * mean, 0)), floor)


def load_view(views, report, *, head_role, old_role, group, domain, protocol_sha, allowed_roles):
    if old_role not in allowed_roles:
        raise ValueError('This entry point cannot open this data role')
    key = f'group_{group}/{head_role}/{old_role}_{domain}'
    entry = report['views'][key]
    meta = checked_json(Path(views) / entry['metadata'], entry['metadata_sha256'])
    require_view(meta, head_role, old_role, protocol_sha, group, domain)
    ip = Path(views) / meta['row_indices']
    if sha(ip) != meta['row_indices_sha256'] or sha(meta['source_arrays']) != meta['source_arrays_sha256']:
        raise ValueError('View indices or immutable source changed')
    source_meta = checked_json(meta['source_metadata'], meta['source_metadata_sha256'])
    if (source_meta.get('role') != 'g_eval_' + old_role or
            source_meta.get('arrays_sha256') != meta['source_arrays_sha256'] or
            source_meta.get('group') != group or source_meta.get('domain') != domain or
            source_meta.get('protocol_sha256') != SOURCE_CACHE_PROTOCOL):
        raise ValueError('Source cache role or binding changed')
    indices = np.load(ip, allow_pickle=False)
    if indices.ndim != 1 or indices.dtype != np.int64 or len(indices) != meta['rows'] or len(np.unique(indices)) != len(indices):
        raise ValueError('Invalid view row roster')
    with np.load(meta['source_arrays'], allow_pickle=False) as z:
        if not len(indices) or indices.min() < 0 or indices.max() >= len(z['observed']):
            raise ValueError('Out-of-range or empty view')
        data = {k: z[k][indices] for k in ['observed', 'labels', 'parent_ids', 'pixel_sha256']}
    if sorted(np.unique(data['parent_ids']).tolist()) != meta['parent_ids']:
        raise ValueError('View parent roster changed')
    if hashlib.sha256(np.ascontiguousarray(data['pixel_sha256']).tobytes()).hexdigest() != meta['pixel_array_sha256']:
        raise ValueError('View pixel roster changed')
    if not np.isfinite(data['observed']).all() or not np.isfinite(data['labels']).all():
        raise ValueError('Nonfinite observed input')
    return data


def source_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('s1_*.py'))}
