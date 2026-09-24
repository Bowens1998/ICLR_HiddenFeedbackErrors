"""Independent v9 S1 arithmetic; no project evaluator/model/projection imports.

Target interface, not a claim that production files already exist. Only execute
on a hash-bound verification input manifest after root's scientific opening gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import erf

PRIMARY = ['decoded_teacher/U_AC', 'decoded_teacher/S_AC',
           'physical_labels/U_AC', 'physical_labels/S_AC']
SECONDARY = ['decoded_teacher/U_A_minus_U_AC', 'decoded_teacher/insertion_displacement_suppression',
             'physical_labels/U_A_minus_U_AC', 'physical_labels/insertion_displacement_suppression']
NAMES = PRIMARY + SECONDARY


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound_path(record, base):
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    if file_sha(path) != record['sha256']:
        raise ValueError('Bound input checksum mismatch: ' + str(path))
    return path


def finite_array(value, label):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite ' + label)
    return a


def independent_gelu_forward(tokens, parameters, width=256, blocks=2, factor=0.5, batch=512):
    """FP64 physical-output forward, erf formulation, no Torch/production code."""
    z = finite_array(tokens, 'tokens')
    mean = finite_array(parameters['mean'], 'input mean')
    scale = finite_array(parameters['scale'], 'input scale')
    out_mean = finite_array(parameters['target_mean'], 'output mean')
    out_scale = finite_array(parameters['target_scale'], 'output scale')
    if mean.ndim != 1 or z.shape[-1] != len(mean) or scale.shape != mean.shape:
        raise ValueError('Input/normalizer shape mismatch')
    if out_mean.ndim != 1 or out_scale.shape != out_mean.shape:
        raise ValueError('Output normalizer shape mismatch')
    if not (scale > 0).all() or not (out_scale > 0).all():
        raise ValueError('Nonpositive normalizer scale')
    dims = {'input': (width, len(mean)), 'output': (len(out_mean), width),
            'skip': (len(out_mean), len(mean))}
    for b in range(blocks):
        dims[f'blocks.{b}.fc1'] = (width, width)
        dims[f'blocks.{b}.fc2'] = (width, width)
    weights = {}
    for layer, shape in dims.items():
        w = finite_array(parameters[layer + '.weight'], layer + ' weight')
        if w.shape != shape:
            raise ValueError('Wrong layer shape: ' + layer)
        bias = None if layer == 'skip' else finite_array(parameters[layer + '.bias'], layer + ' bias')
        if bias is not None and bias.shape != (shape[0],):
            raise ValueError('Wrong layer bias shape: ' + layer)
        weights[layer] = (w, bias)
    def linear(x, name):
        w, bias = weights[name]
        y = np.matmul(x, w.T)
        return y if bias is None else y + bias
    def gelu(x):
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    flat = z.reshape(-1, len(mean))
    result = np.empty((len(flat), len(out_mean)), dtype=np.float64)
    for start in range(0, len(flat), batch):
        x = (flat[start:start + batch] - mean) / scale
        h = gelu(linear(x, 'input'))
        for b in range(blocks):
            h = h + factor * linear(gelu(linear(h, f'blocks.{b}.fc1')), f'blocks.{b}.fc2')
        result[start:start + len(x)] = (linear(h, 'output') + linear(x, 'skip')) * out_scale + out_mean
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite independent output')
    return result.reshape(z.shape[:-1] + (len(out_mean),))


def goal_vectors(predictions, truth, goal_count=256, stream_count=4):
    """Return N x 8 goal vectors; fixed architecture/stream averages stay inside goal."""
    pred = finite_array(predictions, 'reserved-readout predictions')
    target = finite_array(truth, 'truth')
    expected = (2, 2, 2, goal_count, stream_count, 4, 5, 6)
    if pred.shape != expected or target.shape != (2, goal_count, stream_count, 5, 6):
        raise ValueError('Shape/roster mismatch')
    # Constraint is not allowed to change either copied free or full-reset branch.
    for branch in (0, 3):
        if not np.array_equal(pred[:, :, 0, :, :, branch], pred[:, :, 1, :, :, branch]):
            raise ValueError('Free/reset branches differ across constraint copies')
    delta = pred[..., 2:4] - target[:, None, None, :, :, None, :, 2:4]
    loss = np.sum(delta * delta, axis=-1)  # dx^2 + dy^2, never divide by two.
    columns = {}
    descriptive = {}
    for objective, name in enumerate(('decoded_teacher', 'physical_labels')):
        single = loss[:, objective, 0, :, :, :, -1]
        joint = loss[:, objective, 1, :, :, :, -1]
        # Each sliced array is [architecture, goal, stream, branch].
        avg = lambda x: np.mean(x, axis=(0, 2))
        u_a = avg(single[..., 0] - single[..., 1])
        u_ac = avg(joint[..., 0] - joint[..., 1])
        columns[name + '/U_AC'] = u_ac
        columns[name + '/S_AC'] = avg(joint[..., 2] - joint[..., 1])
        columns[name + '/U_A_minus_U_AC'] = u_a - u_ac
        current = pred[:, objective, :, :, :, :, 0, 2:4]
        # [architecture, constraint, goal, stream, branch, block coordinate].
        shift = np.sum((current[..., 1, :] - current[..., 0, :]) ** 2, axis=-1)
        columns[name + '/insertion_displacement_suppression'] = avg(shift[:, 0] - shift[:, 1])
        descriptive[name] = {'U_A_goal': u_a, 'U_AC_goal': u_ac,
                            'insertion_shift_sq_A_goal': avg(shift[:, 0]),
                            'insertion_shift_sq_AC_goal': avg(shift[:, 1])}
    result = np.column_stack([columns[n] for n in NAMES])
    if result.shape != (goal_count, 8):
        raise AssertionError('Wrong goal-vector assembly')
    return result, descriptive


def count_weighted_bootstrap(vectors, indices, expected_draws=20000):
    values = finite_array(vectors, 'goal vectors')
    draw = np.asarray(indices)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError('Expected eight contrast vectors')
    n = len(values)
    if draw.shape != (expected_draws, n) or draw.dtype != np.int64:
        raise ValueError('Bootstrap shape/dtype changed')
    if np.any(draw < 0) or np.any(draw >= n):
        raise ValueError('Out-of-range goal index')
    # Multiplicities per goal, independent of production's index-gather algorithm.
    counts = np.zeros((expected_draws, n), dtype=np.int32)
    for row in range(expected_draws):
        counts[row] = np.bincount(draw[row], minlength=n)
    if not np.array_equal(counts.sum(1), np.full(expected_draws, n)):
        raise AssertionError('Bootstrap count corruption')
    boot = counts @ values / n
    bounds = np.quantile(boot, [0.00625, 0.99375], axis=0, method='linear')
    means = values.mean(0)
    rows = []
    for j, name in enumerate(NAMES):
        lo, hi = map(float, bounds[:, j])
        rows.append({'id': name, 'family': 'primary' if j < 4 else 'secondary',
                     'mean': float(means[j]), 'lower': lo, 'upper': hi,
                     'nominal_coverage': 0.9875,
                     'classification': 'positive' if lo > 0 else ('negative' if hi < 0 else 'unresolved')})
    return rows, boot


def validate_accepted_population(acceptance, binding):
    """Independent receipt/roster checks; no production admission import."""
    expected_pairs = {(g, stream) for g in (0, 1) for stream in (0, 1, 2, 3)}
    rows = acceptance.get('populations', [])
    if (acceptance.get('status') != 'FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING' or
            acceptance.get('study_id') != binding['study_id'] or
            acceptance.get('protocol_sha256') != binding['design']['sha256'] or
            acceptance.get('axis_labels') != binding['axis_labels'] or
            acceptance.get('complete_population_accepted') is not True or
            acceptance.get('population_count') != 8 or acceptance.get('goal_count') != 256 or
            len(rows) != 8 or {(r['group'], r['stream']) for r in rows} != expected_pairs):
        raise ValueError('Accepted population identity or complete roster changed')
    flags = ('complete_finite_shapes', 'native_identity_checks', 'insertion_and_full_function_checks')
    if any(r.get('count') != 256 or any(r.get(flag) is not True for flag in flags) for r in rows):
        raise ValueError('Incomplete accepted population checks')
    if any(acceptance.get(k) is not False for k in ('D_weights_deserialized', 'D_predictions_opened', 'effects_computed')):
        raise ValueError('Reserved evaluation was opened before population acceptance')


def validate_reserved_head_binding(acceptance, reserved):
    """Supplied evaluators must be the two D checkpoints in actual acceptance."""
    selected = [h for h in acceptance.get('heads', []) if h.get('head_role') == 'D']
    if (len(selected) != 2 or {h['group'] for h in selected} != {0, 1} or
            [h['group'] for h in reserved] != [0, 1]):
        raise ValueError('Accepted/supplied reserved head roster changed')
    hashes = {h['group']: h['sha256'] for h in selected}
    if any(not isinstance(h['sha256'], str) or len(h['sha256']) != 64 or
           h['sha256'] != hashes[h['group']] for h in reserved):
        raise ValueError('Reserved head differs from accepted selected D checkpoint')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    binding_path = Path(args.binding)
    if file_sha(binding_path) != args.binding_sha256:
        raise ValueError('Verification binding changed')
    cfg = json.loads(binding_path.read_text())
    if cfg['study_id'] != 'prospective_mechanism_v9_s1_20260923' or cfg['scientific_opening_gate'] != 'ACCEPTED_ALL_FROZEN_INPUTS_AND_NUMERICAL_FAMILIES':
        raise ValueError('Wrong study or scientific opening gate absent')
    if cfg['axis_labels'] != {'groups': [0, 1], 'objectives': ['decoded_teacher', 'physical_labels'],
            'constraints': ['A', 'AC'], 'streams': [0, 1, 2, 3],
            'branches': ['free', 'actual', 'donor', 'reset'], 'horizons': [5, 10, 15, 20, 25],
            'pose': ['agent_x', 'agent_y', 'block_x', 'block_y', 'sin_theta', 'cos_theta']}:
        raise ValueError('Frozen axis labels changed')
    base = binding_path.parent
    design_path = bound_path(cfg['design'], base)
    design = json.loads(design_path.read_text())
    if design['study_id'] != cfg['study_id']:
        raise ValueError('Design/study identity mismatch')
    acceptance_path = bound_path(cfg['acceptance'], base)
    acceptance = json.loads(acceptance_path.read_text())
    validate_accepted_population(acceptance, cfg)
    locked_ids = [r['id'] for r in design['statistics']['primary'] + design['statistics']['secondary']]
    if locked_ids != NAMES or design['statistics']['individual_two_sided_coverage'] != 0.9875:
        raise ValueError('Statistical lock differs from independent contract')
    scores_path = bound_path(cfg['scores_archive'], base)
    boot_path = bound_path(cfg['bootstrap_archive'], base)
    with np.load(scores_path, allow_pickle=False) as data:
        predictions = np.array(data[cfg['scores_archive']['predictions_key']], dtype=np.float64)
        truth = np.array(data[cfg['scores_archive']['truth_key']], dtype=np.float64)
        norms = finite_array(data[cfg['scores_archive']['norm_key']], 'common norms')
    if norms.shape != (2, 256, 4) or (norms < 0).any():
        raise ValueError('Invalid common-norm accounting')
    with np.load(boot_path, allow_pickle=False) as data:
        draws = data[cfg['bootstrap_archive']['indices_key']]
    regenerated_draws = np.random.default_rng(design['statistics']['bootstrap_seed']).integers(
        0, 256, size=(20000, 256), dtype=np.int64)
    if not np.array_equal(draws, regenerated_draws):
        raise ValueError('Bootstrap indices differ from frozen PCG64 seed')
    vectors, descriptive = goal_vectors(predictions, truth)
    rows, boot = count_weighted_bootstrap(vectors, draws)
    forward = {'status': 'NOT_RUN_NO_RAW_TOKEN_INPUT',
               'scope': 'Arithmetic reconstruction from bound physical predictions only'}
    if 'raw_tokens_archive' in cfg and 'raw_token_shards' in cfg:
        raise ValueError('Choose exactly one raw-token layout')
    if 'raw_tokens_archive' in cfg or 'raw_token_shards' in cfg:
        validate_reserved_head_binding(acceptance, cfg['reserved_readouts'])
        heads = []
        for entry in cfg['reserved_readouts']:
            head_path = bound_path(entry, base)
            with np.load(head_path, allow_pickle=False) as data:
                heads.append(dict(data))
        max_abs = 0.0
        token_count = 0
    if 'raw_tokens_archive' in cfg:
        raw_path = bound_path(cfg['raw_tokens_archive'], base)
        with np.load(raw_path, allow_pickle=False) as data:
            tokens = np.array(data[cfg['raw_tokens_archive']['tokens_key']])
        if tokens.shape != (2, 2, 2, 256, 4, 4, 5, 192):
            raise ValueError('Raw token population changed')
        for group, head in enumerate(heads):
            reference = independent_gelu_forward(tokens[group], head)
            residual = np.abs(reference - predictions[group])
            max_abs = max(max_abs, float(residual.max()))
            if not np.allclose(reference, predictions[group], rtol=1e-12, atol=1e-9):
                raise ValueError('Reserved evaluator forward mismatch')
        token_count = int(np.prod(tokens.shape[:-1]))
    if 'raw_token_shards' in cfg:
        coverage = np.zeros((2, 256), dtype=bool)
        for item in cfg['raw_token_shards']:
            group = item['group']
            if group not in (0, 1):
                raise ValueError('Raw shard group outside frozen roster')
            raw_path = bound_path(item, base)
            with np.load(raw_path, allow_pickle=False) as data:
                tokens = np.asarray(data[item['tokens_key']])
                ids = np.asarray(data[item['goal_indices_key']])
            if ids.ndim != 1 or ids.dtype != np.int64 or not len(ids):
                raise ValueError('Raw shard goal-ID contract changed')
            if (ids < 0).any() or (ids >= 256).any() or len(np.unique(ids)) != len(ids):
                raise ValueError('Invalid or repeated raw-shard goal IDs')
            if coverage[group, ids].any():
                raise ValueError('Duplicate raw-shard coverage')
            if tokens.shape != (2, 2, len(ids), 4, 4, 5, 192):
                raise ValueError('Raw shard population/axes changed')
            reference = independent_gelu_forward(tokens, heads[group])
            expected = np.take(predictions[group], ids, axis=2)
            max_abs = max(max_abs, float(np.max(np.abs(reference - expected))))
            if not np.allclose(reference, expected, rtol=1e-12, atol=1e-9):
                raise ValueError('Reserved evaluator shard-forward mismatch')
            coverage[group, ids] = True
            token_count += int(np.prod(tokens.shape[:-1]))
        if not coverage.all():
            raise ValueError('Incomplete raw-token population; no subset pass')
    if 'raw_tokens_archive' in cfg or 'raw_token_shards' in cfg:
        forward = {'status': 'PASS_COMPLETE_RESERVED_FORWARD', 'max_abs': max_abs,
                   'token_count': token_count, 'rtol': 1e-12, 'atol': 1e-9,
                   'scope': 'All bound saved tokens; not raw-image encoder or world-model rollout reconstruction'}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'independent_vectors_and_bootstrap.npz', goal_vectors=vectors,
                        bootstrap_means=boot, contrast_ids=np.asarray(NAMES),
                        **{f'{o}_{k}': v for o, d in descriptive.items() for k, v in d.items()})
    report = {'status': 'PASS_INDEPENDENT_STAGE1_ARITHMETIC',
              'binding_sha256': args.binding_sha256, 'reference_source_sha256': file_sha(__file__),
              'design_sha256': cfg['design']['sha256'],
              'acceptance_sha256': cfg['acceptance']['sha256'],
              'bootstrap_unit': 'recipient goal', 'goals': 256, 'draws': 20000,
              'fixed_groups': [0, 1], 'fixed_streams': [0, 1, 2, 3],
              'zero_common_norm_families': int((norms == 0).sum()),
              'contrasts': rows, 'forward': forward,
              'vectors_sha256': file_sha(out / 'independent_vectors_and_bootstrap.npz'),
              'limits': 'Does not qualify heads, certify projection feasibility, establish physical-state equality, or identify mediation. Each family has four fixed slots; no omitted goals or model strata.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
