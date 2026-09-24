"""Independent saved-qualification arithmetic; no production metric imports.

This checks saved physical predictions, not a new forward pass or intervention.
A faithfully reproduced blocked qualification is a complete valid verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

METRICS = ('six_normalized_mse', 'block_position_mse', 'agent_position_mse', 'wrapped_angle_mse')
DOMAINS = ('expert', 'planner')
ROSTER = {(g, h) for g in (0, 1) for h in ('C', 'D')}
DESIGN_SHA = '5ce812fe7259bb7489bef214d733d2f370d6eb7650db0f3b7a9d1860f3e41fe3'
TOLERANCE = dict(rtol=1e-12, atol=1e-9)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bound(path, expected, inventory):
    p = Path(path).resolve()
    if not expected or digest(p) != expected:
        raise ValueError(f'Changed or unbound file: {p}')
    if str(p) in inventory and inventory[str(p)] != expected:
        raise ValueError('Conflicting immutable file identity')
    inventory[str(p)] = expected
    return p


def read(path, expected, inventory):
    return json.loads(bound(path, expected, inventory).read_text())


def errors(prediction, truth, scale):
    p, t, s = [np.asarray(x, np.float64) for x in (prediction, truth, scale)]
    if (p.ndim != 2 or p.shape != t.shape or p.shape[1] != 6 or s.shape != (6,) or
            not len(p) or any(not np.isfinite(x).all() for x in (p, t, s)) or (s <= 0).any()):
        raise ValueError('Invalid complete physical poses or g_A scales')
    components = [(p[:, j] - t[:, j]) ** 2 for j in range(6)]
    normalized = sum(components[j] / (s[j] ** 2) for j in range(6)) / 6.
    angle_difference = np.arctan2(p[:, 4], p[:, 5]) - np.arctan2(t[:, 4], t[:, 5])
    wrapped = (angle_difference + np.pi) % (2 * np.pi) - np.pi
    return dict(six_normalized_mse=normalized,
                block_position_mse=components[2] + components[3],
                agent_position_mse=components[0] + components[1],
                wrapped_angle_mse=wrapped * wrapped)


def metric_gate(current, reference, threshold):
    valid = bool(np.isfinite(current) and np.isfinite(reference) and reference > 0)
    ratio = float(current / reference) if valid else None
    return ratio, bool(valid and ratio <= threshold)


def self_test():
    # Known spatial sums: agent=3^2+4^2=25, block=5^2+12^2=169.
    # +170 versus -170 degrees must wrap to -20 degrees, not +340.
    p = np.array([[3., 4., 5., 12., np.sin(np.deg2rad(170)), np.cos(np.deg2rad(170))]])
    t = np.array([[0., 0., 0., 0., np.sin(np.deg2rad(-170)), np.cos(np.deg2rad(-170))]])
    s = np.array([3., 2., 5., 4., 2., 3.])
    result = errors(p, t, s)
    np.testing.assert_allclose(result['agent_position_mse'], [25.], rtol=0, atol=0)
    np.testing.assert_allclose(result['block_position_mse'], [169.], rtol=0, atol=0)
    np.testing.assert_allclose(result['six_normalized_mse'], [(15. + np.sin(np.deg2rad(170))**2)/6], **TOLERANCE)
    np.testing.assert_allclose(result['wrapped_angle_mse'], [np.deg2rad(20)**2], **TOLERANCE)
    assert metric_gate(11., 10., 1.1) == (1.1, True)
    assert metric_gate(12., 10., 1.1) == (1.2, False)
    assert metric_gate(0., 0., 1.1) == (None, False)
    broken = p.copy(); broken[0, 0] = np.nan
    try:
        errors(broken, t, s)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite physical prediction was admitted')
    return dict(status='PASS_ANALYTICAL_SYNTHETIC_FIXTURE', physical_coordinate_sums=True,
                reference_normalizer=True, angle_branch_cut=True, threshold_boundary=True,
                failed_gate_retained=True, zero_reference_blocked=True, nonfinite_rejected=True)


def verify(a):
    if Path(a.output).exists():
        raise ValueError('Never overwrite a verification report')
    inputs = {}; lock = read(a.verifier_lock, a.verifier_lock_sha256, inputs)
    own_sha = digest(__file__)
    if (lock.get('status') != 'VERIFIER_FROZEN_AFTER_SYNTHETIC_BEFORE_ACTUAL_ARRAY_RECOMPUTATION' or
            lock.get('source_sha256') != own_sha or lock.get('protocol_sha256') != DESIGN_SHA or
            lock.get('synthetic_test') != self_test() or lock.get('numeric_tolerance') != TOLERANCE):
        raise ValueError('Independent verifier source/synthetic boundary changed')
    inputs[str(Path(__file__).resolve())] = own_sha
    cfg = read(a.protocol, a.protocol_sha256, inputs)
    if a.protocol_sha256 != DESIGN_SHA:
        raise ValueError('Wrong frozen S1 protocol')
    q = cfg['qualification']
    if q['max_ratio_to_g_A'] != 1.1 or q['domains'] != list(DOMAINS) or q['metrics'] != list(METRICS):
        raise ValueError('Changed qualification threshold or metric roster')
    selected = read(a.selected, a.selected_sha256, inputs)
    references = read(a.a_bindings, a.a_bindings_sha256, inputs)
    report = read(a.qualification, a.qualification_sha256, inputs)
    if (selected.get('status') != 'ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION' or
            selected.get('protocol_sha256') != DESIGN_SHA or len(selected.get('heads', [])) != 4 or
            {(r['group'], r['head_role']) for r in selected['heads']} != ROSTER):
        raise ValueError('Missing or changed complete selected-head roster')
    if (report.get('protocol_sha256') != DESIGN_SHA or report.get('selected_lock_sha256') != a.selected_sha256 or
            report.get('a_bindings_sha256') != a.a_bindings_sha256 or report.get('all_four_completed') is not True or
            len(report.get('rows', [])) != 4 or {(r['group'], r['head_role']) for r in report['rows']} != ROSTER):
        raise ValueError('Qualification receipt identities or completeness changed')
    if not (Path(a.qualification).parent / 'DONE').is_file():
        raise ValueError('Qualification did not complete')
    inputs[str((Path(a.qualification).parent / 'DONE').resolve())] = digest(Path(a.qualification).parent / 'DONE')
    selected_hash = {(r['group'], r['head_role']): r['checkpoint_sha256'] for r in selected['heads']}
    scales = {}; reference_hash = {}
    for group in (0, 1):
        rows = [r['head_A'] for r in references['groups'] if r['group'] == group]
        if len(rows) != 1:
            raise ValueError('Missing/duplicate reference A group')
        reference_hash[group] = rows[0]['sha256']
        path = Path(a.reference_heads) / f'group_{group}_head_A.npz'
        with np.load(bound(path, rows[0]['sha256'], inputs), allow_pickle=False) as z:
            scales[group] = z['target_scale'].copy()
        if scales[group].shape != (6,) or not np.isfinite(scales[group]).all() or (scales[group] <= 0).any():
            raise ValueError('Invalid hash-bound g_A target scales')
    verified_rows = []; checks = []; maximum_saved_error_difference = 0.
    parent_rosters = {}; paths_seen = set()
    for row in sorted(report['rows'], key=lambda r: (r['group'], r['head_role'])):
        g, h = row['group'], row['head_role']
        if (row['checkpoint_sha256'] != selected_hash[g, h] or row['reference_head_sha256'] != reference_hash[g] or
                set(row.get('files', {})) != set(DOMAINS) or set(row.get('scores', {})) != set(DOMAINS) or
                set(row.get('reference_scores', {})) != set(DOMAINS) or set(row['gate']['checks']) != set(DOMAINS)):
            raise ValueError('Changed per-head identity/domain binding')
        current = {}; baseline = {}; saved_domains = []
        for domain in DOMAINS:
            entry = row['files'][domain]
            expected_file = f'group_{g}/{h}/{domain}.npz'
            if entry['file'] != expected_file:
                raise ValueError('Changed group/head/domain archive mapping')
            path = bound(Path(a.qualification).parent / entry['file'], entry['sha256'], inputs)
            if path in paths_seen:
                raise ValueError('Repeated domain archive')
            paths_seen.add(path)
            with np.load(path, allow_pickle=False) as z:
                arrays = dict(z)
            required = {'prediction', 'reference_prediction', 'truth', 'parent_ids', 'pixel_sha256'}
            required.update(prefix + metric for prefix in ('head_', 'reference_') for metric in METRICS)
            if set(arrays) != required:
                raise ValueError('Missing or additional qualification archive fields')
            n = entry['rows']; expected_parents = 128 if domain == 'expert' else 256
            parents, pixels = arrays['parent_ids'], arrays['pixel_sha256']
            if (parents.shape != (n,) or parents.dtype.kind not in 'iu' or pixels.shape != (n,) or
                    pixels.dtype.kind not in 'SU' or len(np.unique(parents)) != expected_parents or
                    entry['parents'] != expected_parents or any(arrays[k].shape != (n, 6)
                    for k in ('prediction', 'reference_prediction', 'truth'))):
                raise ValueError('Incomplete rows, physical outputs or qualification parents')
            parent_rosters[g, h, domain] = set(map(int, parents))
            computed = errors(arrays['prediction'], arrays['truth'], scales[g])
            old = errors(arrays['reference_prediction'], arrays['truth'], scales[g])
            current[domain] = {}; baseline[domain] = {}
            if any(set(row[k][domain]) != set(METRICS) for k in ('scores', 'reference_scores')) or set(row['gate']['checks'][domain]) != set(METRICS):
                raise ValueError('Missing metric or gate check')
            for metric in METRICS:
                for prefix, values in [('head_', computed[metric]), ('reference_', old[metric])]:
                    saved = arrays[prefix + metric]
                    if saved.shape != (n,) or not np.isfinite(saved).all():
                        raise ValueError('Missing/nonfinite saved per-frame errors')
                    np.testing.assert_allclose(values, saved, **TOLERANCE)
                    maximum_saved_error_difference = max(maximum_saved_error_difference, float(np.max(np.abs(values - saved))))
                current[domain][metric] = float(np.mean(computed[metric]))
                baseline[domain][metric] = float(np.mean(old[metric]))
                np.testing.assert_allclose(current[domain][metric], row['scores'][domain][metric], **TOLERANCE)
                np.testing.assert_allclose(baseline[domain][metric], row['reference_scores'][domain][metric], **TOLERANCE)
                ratio, passed = metric_gate(current[domain][metric], baseline[domain][metric], q['max_ratio_to_g_A'])
                saved_gate = row['gate']['checks'][domain][metric]
                if ratio is None:
                    if saved_gate['ratio'] is not None:
                        raise ValueError('Invalid denominator gate was not retained')
                else:
                    np.testing.assert_allclose(ratio, saved_gate['ratio'], rtol=1e-12, atol=1e-12)
                if saved_gate['passed'] is not passed:
                    raise ValueError('Saved qualification pass flag differs')
                checks.append(dict(group=g, head_role=h, domain=domain, metric=metric,
                    head_mean=current[domain][metric], reference_mean=baseline[domain][metric], ratio=ratio, passed=passed))
            saved_domains.append(dict(domain=domain, rows=n, parents=expected_parents, sha256=entry['sha256']))
        head_pass = all(c['passed'] for c in checks if c['group'] == g and c['head_role'] == h)
        if row['gate']['passed'] is not head_pass:
            raise ValueError('Saved aggregate head gate differs')
        verified_rows.append(dict(group=g, head_role=h, checkpoint_sha256=selected_hash[g, h],
            reference_head_sha256=reference_hash[g], domains=saved_domains, passed=head_pass))
    for group in (0, 1):
        for domain in DOMAINS:
            if parent_rosters[group, 'C', domain] & parent_rosters[group, 'D', domain]:
                raise ValueError('Qualification C/D parents overlap within domain')
    passed = all(c['passed'] for c in checks)
    expected_status = 'COMPLETE_S1_QUALIFICATION_PASS' if passed else 'COMPLETE_S1_QUALIFICATION_BLOCKED_GATES'
    if (len(checks) != 32 or report.get('status') != expected_status or report.get('all_gates_passed') is not passed or
            report.get('formal_qualification_gate_passed') is not passed):
        raise ValueError('Saved complete qualification outcome differs')
    result = dict(status='PASS_INDEPENDENT_COMPLETE_QUALIFICATION_REPRODUCTION',
        scientific_qualification_status=expected_status, formal_qualification_gate_passed=passed,
        all_four_completed=True, heads=verified_rows, domains_verified=8, metric_gates_verified=32,
        metric_gates_passed=sum(c['passed'] for c in checks), checks=checks,
        maximum_saved_per_frame_error_difference=maximum_saved_error_difference,
        numeric_tolerance=TOLERANCE, source_sha256=own_sha, verifier_lock_sha256=a.verifier_lock_sha256,
        protocol_sha256=DESIGN_SHA, qualification_sha256=a.qualification_sha256,
        selected_lock_sha256=a.selected_sha256, a_bindings_sha256=a.a_bindings_sha256,
        inputs_sha256=inputs, python=sys.version, numpy=np.__version__,
        scope='Independent arithmetic reproduction of all saved observed qualification predictions/labels '
        'and per-frame metrics; actual A files hash-bound and their target scales used. Selected C/D hashes '
        'cross-bound to receipt, but no new head forward pass or intervention effects opened. A complete '
        'blocked scientific gate is retained as a valid reproduced result.')
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=result['status'], qualification=expected_status,
        metric_gates_verified=32, metric_gates_passed=result['metric_gates_passed'], output=str(out), sha256=digest(out))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--self-test', action='store_true')
    for key in ('protocol', 'selected', 'a-bindings', 'qualification', 'verifier-lock'):
        p.add_argument('--' + key); p.add_argument('--' + key + '-sha256')
    p.add_argument('--reference-heads'); p.add_argument('--output'); a = p.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), sort_keys=True)); return
    if any(v is None for k, v in vars(a).items() if k != 'self_test'):
        p.error('All hash-bound inputs, local reference-heads directory and output are required')
    verify(a)


if __name__ == '__main__':
    main()
