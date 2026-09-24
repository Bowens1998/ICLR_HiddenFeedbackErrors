"""Independent pre-qualification verification of all four frozen S1 heads.

Only selected checkpoints, fit reports, immutable metadata and saved validation
forward fixtures are read. No Torch or project readout/fit implementation import.
Normalizer origin is authenticated through frozen fit provenance; its moments are
not recomputed from fitting caches by this bounded verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

STUDY = 'prospective_mechanism_v9_s1_20260923'
PURPOSE = 'INDEPENDENT_ALL_FOUR_SELECTED_HEADS_PREQUALIFICATION'
RTOL, ATOL, FLOOR = 1e-12, 1e-9, 1e-6
ROSTER = {(g, role) for g in (0, 1) for role in ('C', 'D')}
PREFIX = 'strengthening/prospective_mechanism_v9_20260923/scripts/'
V8 = 'strengthening/presubmission_v8_20260922/independent_readout/scripts/readout.py'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class BoundInputs:
    def __init__(self, base):
        self.base, self.opened = Path(base), {}

    def file(self, record, expected=None):
        checksum = record['sha256']
        if (not isinstance(checksum, str) or len(checksum) != 64 or
                any(c not in '0123456789abcdef' for c in checksum) or
                (expected is not None and checksum != expected)):
            raise ValueError('Asset checksum differs from immutable selected identity')
        path = Path(record['path'])
        if not path.is_absolute(): path = self.base / path
        path = path.resolve()
        if digest(path) != checksum: raise ValueError('Changed bound asset: ' + str(path))
        self.opened[str(path)] = checksum
        return path

    def json(self, record, expected=None):
        return json.loads(self.file(record, expected).read_text())


def complete_roster(rows, name):
    if len(rows) != 4 or {(r['group'], r['head_role']) for r in rows} != ROSTER:
        raise ValueError(name + ': exact four-head roster required')
    return {(r['group'], r['head_role']): r for r in rows}


def finite_nonnegative(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError('Invalid ' + name)
    return float(value)


def validate_history(report, selected_row, config):
    settings = config['fit']
    if settings['updates'] != 8000 or settings['validation_interval'] != 250:
        raise ValueError('Unsupported fixed validation schedule')
    history = report['history']
    if (report.get('updates') != 8000 or len(history) != 32 or
            [r['step'] for r in history] != list(range(250, 8001, 250))):
        raise ValueError('Incomplete or altered 250-to-8000 validation schedule')
    for row in history:
        domains = row['domain_validation']
        if set(domains) != {'expert', 'planner'}: raise ValueError('Missing validation domain')
        expert = finite_nonnegative(domains['expert'], 'expert validation MSE')
        planner = finite_nonnegative(domains['planner'], 'planner validation MSE')
        score = finite_nonnegative(row['balanced_six_normalized_validation_mse'], 'validation score')
        finite_nonnegative(row['train_loss'], 'training loss')
        if score != .5 * (expert + planner): raise ValueError('Validation domain balance changed')
    best = min(history, key=lambda r: (r['balanced_six_normalized_validation_mse'], r['step']))
    selected = report['selected']
    for key in ('step', 'balanced_six_normalized_validation_mse', 'domain_validation', 'train_loss'):
        if selected.get(key) != best[key]: raise ValueError('Selected checkpoint is not earliest validation minimum')
    if selected.get('file') != 'selected.npz' or selected_row.get('selected_step') != best['step']:
        raise ValueError('Selected file or lock step differs from earliest minimum')
    return best


def expected_shapes(role):
    shapes = dict(mean=(192,), scale=(192,), target_mean=(6,), target_scale=(6,))
    if role == 'C':
        for layer, shape in [('0', (512, 192)), ('2', (512, 512)), ('4', (6, 512))]:
            shapes[layer + '.weight'], shapes[layer + '.bias'] = shape, (shape[0],)
    elif role == 'D':
        for layer, shape in [('input', (256, 192)), ('output', (6, 256)), ('skip', (6, 192))]:
            shapes[layer + '.weight'] = shape
            if layer != 'skip': shapes[layer + '.bias'] = (shape[0],)
        for block in range(2):
            for layer in ('fc1', 'fc2'):
                name = f'blocks.{block}.{layer}'
                shapes[name + '.weight'], shapes[name + '.bias'] = (256, 256), (256,)
    else:
        raise ValueError('Unknown head role')
    return shapes


def validate_parameters(head, role):
    shapes = expected_shapes(role)
    if set(head) != set(shapes): raise ValueError('Unexpected/missing selected parameters')
    for name, shape in shapes.items():
        dtype = np.float64 if name in ('mean', 'scale', 'target_mean', 'target_scale') else np.float32
        if head[name].shape != shape or head[name].dtype != dtype or not np.isfinite(head[name]).all():
            raise ValueError('Invalid shape/dtype/finite parameter: ' + name)
    if (head['scale'] < FLOOR).any() or (head['target_scale'] < FLOOR).any():
        raise ValueError('Normalizer scale violates the frozen fit floor')


def independent_forward(tokens, head, role):
    """Explicit FP64 contractions; C ReLU or D erf-GELU, then own physical scale."""
    validate_parameters(head, role)
    if tokens.ndim != 2 or tokens.shape[1] != 192 or not np.isfinite(tokens).all():
        raise ValueError('Invalid validation token matrix')
    x = (tokens.astype(np.float64) - head['mean']) / head['scale']
    def linear(values, layer, bias=True):
        y = np.einsum('ni,oi->no', values, head[layer + '.weight'].astype(np.float64), optimize=False)
        return y + head[layer + '.bias'].astype(np.float64) if bias else y
    if role == 'C':
        hidden = np.maximum(linear(x, '0'), 0.)
        hidden = np.maximum(linear(hidden, '2'), 0.)
        normalized = linear(hidden, '4')
    else:
        def gelu(value): return .5 * value * (1. + erf(value / np.sqrt(2.)))
        hidden = gelu(linear(x, 'input'))
        for block in range(2):
            hidden = hidden + .5 * linear(gelu(linear(hidden, f'blocks.{block}.fc1')), f'blocks.{block}.fc2')
        normalized = linear(hidden, 'output') + linear(x, 'skip', bias=False)
    result = normalized * head['target_scale'] + head['target_mean']
    if not np.isfinite(result).all(): raise ValueError('Nonfinite independent forward')
    return result


def compare_forward(actual, expected, label):
    if actual.shape != (64, 6) or expected.shape != (64, 6) or not np.isfinite(expected).all():
        raise ValueError('Wrong complete saved forward fixture: ' + label)
    delta = np.abs(actual - expected)
    allowance = ATOL + RTOL * np.abs(expected)
    if not np.all(delta <= allowance): raise ValueError('Independent forward mismatch: ' + label)
    return dict(maximum_absolute_difference=float(delta.max()), maximum_tolerance_fraction=float((delta / allowance).max()),
                rtol=RTOL, atol=ATOL, passed=True)


def verify(binding_path, binding_sha, output):
    if Path(output).exists(): raise ValueError('Never overwrite independent verification output')
    reader = BoundInputs(Path(binding_path).resolve().parent)
    binding = reader.json(dict(path=str(Path(binding_path).resolve()), sha256=binding_sha))
    if binding.get('study_id') != STUDY or binding.get('purpose') != PURPOSE:
        raise ValueError('Wrong study or verification role')
    config = reader.json(binding['protocol']); protocol_sha = binding['protocol']['sha256']
    implementation = reader.json(binding['head_implementation'])
    inputs = reader.json(binding['fitting_inputs'])
    selected = reader.json(binding['selected_lock'])
    if (config.get('study_id') != STUDY or config.get('groups') != [0, 1] or config.get('head_roles') != ['C', 'D'] or
            config['fit'].get('selection') != 'balanced_six_normalized_validation_mse_then_earlier_step' or
            config['fit'].get('normalizer_std_floor') != FLOOR):
        raise ValueError('Wrong frozen design')
    if (implementation.get('status') != 'S1_HEAD_IMPLEMENTATION_FROZEN_BEFORE_INPUT_AUDIT_OR_FITS' or
            implementation.get('protocol_sha256') != protocol_sha or
            inputs.get('status') != 'S1_FOUR_FIXED_FITS_INPUTS_FROZEN' or
            inputs.get('protocol_sha256') != protocol_sha or
            inputs.get('implementation_lock_sha256') != binding['head_implementation']['sha256']):
        raise ValueError('Missing frozen head implementation/input chain')
    source = implementation['files']
    if (selected.get('status') != 'ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION' or
            selected.get('protocol_sha256') != protocol_sha or selected.get('formal_authorized') is not False or
            selected.get('source_sha256') != source[PREFIX + 'freeze_selected_heads.py']):
        raise ValueError('All-four prequalification selected lock required')
    selected_rows = complete_roster(selected['heads'], 'selected lock')
    assets = complete_roster(binding['heads'], 'downloaded asset binding')
    tasks = complete_roster(inputs['task_roster'], 'fixed fit inputs')
    adapters = {Path(k).name: v for k, v in source.items() if k.startswith(PREFIX) and Path(k).name.startswith('s1_')}
    expected_optimizer = dict(name='Adam', lr=config['fit']['learning_rate'], betas=[.9, .999], eps=1e-8,
        weight_decay=0., amsgrad=False, maximize=False, foreach=False, fused=False)
    results, predictions = [], {}
    # Verify every identity/report before any checkpoint/fixture array is opened.
    admitted = []
    for group, role in sorted(ROSTER):
        key = (group, role); row, entry = selected_rows[key], assets[key]
        if row['views'] != inputs['views'] or row['views_report_sha256'] != inputs['views_report_sha256']:
            raise ValueError('Selected head no longer uses the frozen shared views')
        folder = Path(tasks[key]['output'])
        for remote, filename in [('checkpoint', 'selected.npz'), ('fit_report', 'report.json'), ('forward_fixture', 'forward_verification.npz')]:
            if Path(row[remote]) != folder / filename: raise ValueError('Selected asset outside its fixed fit task')
        report = reader.json(entry['fit_report'], row['fit_report_sha256'])
        if (report.get('status') != 'PASS_S1_FIXED_HEAD_FIT' or report.get('protocol_sha256') != protocol_sha or
                report.get('group') != group or report.get('head_role') != role or
                report.get('views') != row['views'] or report.get('views_report_sha256') != row['views_report_sha256'] or
                report.get('architecture') != config['head_design'][role] or report.get('optimizer') != expected_optimizer or
                report.get('source_sha256') != source[PREFIX + 'fit_head.py'] or
                report.get('adapter_source_sha256') != adapters or report.get('reused_gelu_source_sha256') != source[V8]):
            raise ValueError('Fit/report/source/normalizer selection provenance changed')
        namespace = config['fit']['seed_namespace'].format(role=role, group=group)
        seed = int.from_bytes(hashlib.sha256(f"{config['root_seed']}:{namespace}".encode()).digest()[:4], 'big')
        if report.get('seed_namespace') != namespace or report.get('seed') != seed: raise ValueError('Fixed fit seed changed')
        best = validate_history(report, row, config)
        checkpoint = reader.file(entry['checkpoint'], row['checkpoint_sha256'])
        fixture = reader.file(entry['forward_fixture'], row['forward_fixture_sha256'])
        if report['selected']['sha256'] != row['checkpoint_sha256'] or report['forward_fixture_sha256'] != row['forward_fixture_sha256']:
            raise ValueError('Fit report does not bind its selected checkpoint and validation fixture')
        admitted.append((group, role, row, report, best, checkpoint, fixture))
    for group, role, row, report, best, checkpoint, fixture in admitted:
        with np.load(checkpoint, allow_pickle=False) as data:
            if set(data.files) != set(expected_shapes(role)): raise ValueError('Wrong selected head archive schema')
            head = dict(data)
        validate_parameters(head, role)
        with np.load(fixture, allow_pickle=False) as data:
            if set(data.files) != {'tokens', 'torch_prediction', 'numpy_prediction'}:
                raise ValueError('Wrong validation fixture keys; no qualification/effect input permitted')
            tokens, torch_output, saved_numpy = data['tokens'], data['torch_prediction'], data['numpy_prediction']
        if tokens.shape != (64, 192) or tokens.dtype != np.float32 or not np.isfinite(tokens).all():
            raise ValueError('All 32 expert plus 32 planner validation tokens required')
        for values in (torch_output, saved_numpy):
            if values.shape != (64, 6) or values.dtype != np.float64 or not np.isfinite(values).all():
                raise ValueError('Recorded Torch64/NumPy64 fixture must be complete and finite')
        if report.get('independent_forward_max_abs') != float(np.max(np.abs(torch_output - saved_numpy))):
            raise ValueError('Reported original fixture residual is inconsistent')
        computed = independent_forward(tokens, head, role)
        torch_check = compare_forward(computed, torch_output, 'recorded Torch64')
        numpy_check = compare_forward(computed, saved_numpy, 'recorded original NumPy64')
        results.append(dict(group=group, head_role=role, checkpoint_sha256=row['checkpoint_sha256'],
            fit_report_sha256=row['fit_report_sha256'], forward_fixture_sha256=row['forward_fixture_sha256'],
            fixture_rows=64, selected_step=best['step'], validation_records=32,
            selected_balanced_validation_mse=best['balanced_six_normalized_validation_mse'],
            earliest_minimum_confirmed=True, torch64=torch_check, original_numpy64=numpy_check,
            normalizers=dict(input_dtype=str(head['mean'].dtype), target_dtype=str(head['target_mean'].dtype),
                input_scale_min=float(head['scale'].min()), target_scale_min=float(head['target_scale'].min()),
                shape_finite_floor_checked=True, fit_moments_recomputed=False)))
        predictions[f'group_{group}_{role}'] = computed
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'independent_forward_predictions.npz', **predictions)
    receipt = dict(status='PASS_INDEPENDENT_ALL_FOUR_SELECTED_HEADS', study_id=STUDY,
        binding_sha256=binding_sha, protocol_sha256=protocol_sha,
        selected_lock_sha256=binding['selected_lock']['sha256'], head_implementation_sha256=binding['head_implementation']['sha256'],
        fitting_inputs_sha256=binding['fitting_inputs']['sha256'], heads=results, head_count=4, total_fixture_rows=256,
        source_sha256=digest(__file__), authenticated_inputs=reader.opened,
        predictions_sha256=digest(out / 'independent_forward_predictions.npz'),
        qualification_or_effects_opened=False, production_code_imported=False,
        scope='All four selected heads and complete saved validation fixtures; separate FP64 ReLU/erf-GELU computation. '
        'Selection history, fit/source identities and own normalizer shapes/values/usage checked. Fit-only normalizer '
        'origin and validation-token provenance are source-authenticated, not independently rederived from full raw fit/validation caches. '
        'Does not qualify the heads or certify training reproducibility or intervention effects.')
    (out / 'report.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True); parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.binding, args.binding_sha256, args.output)


if __name__ == '__main__': main()
