"""Independently accept saved fixed-action pilot arrays and paired goal metrics.

No Torch/GPU execution or experiment selection. Requires original accepted
horizon arrays locally; archived old trajectories must match bit for bit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def decode(tokens, head):
    value = (tokens.astype(np.float64) - head['mean']) / head['scale']
    for layer in (0, 2, 4):
        value = np.einsum('...d,od->...o', value, head[f'{layer}.weight']) + head[f'{layer}.bias']
        if layer != 4:
            value = np.clip(value, 0, None)
    return value * head['target_scale'] + head['target_mean']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'pilot', 'horizon', 'protocol', 'pilot-protocol'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    run, pilot, horizon = map(Path, (args.run, args.pilot, args.horizon))
    report, fit, accepted = [read(p) for p in (run/'report.json', pilot/'report.json', pilot/'acceptance.json')]
    assert report['group'] == fit['group'] == accepted['group'] == 0
    assert accepted['status'] == 'PASS_CALIBRATION_AND_CONTINUATION_ARRAYS'
    assert accepted['head_gate_passed'] and fit['head_gate_passed']
    assert report['pilot_report_sha256'] == accepted['report_sha256'] == sha(pilot/'report.json')
    assert report['pilot_acceptance_sha256'] == sha(pilot/'acceptance.json')
    assert report['protocol_sha256'] == accepted['protocol_sha256'] == sha(args.pilot_protocol)
    assert report['source_sha256'] == sha(Path(__file__).with_name('evaluate_readout_calibration_pilot.py'))
    hr, ha = read(horizon/'report.json'), read(horizon/'acceptance.json')
    assert ha['report_sha256'] == sha(horizon/'report.json')
    assert report['plan_sha256'] == hr['plan_sha256']
    heads = {}
    for tag, original in [('old', 'original_head.npz'), ('new', 'fitted_head.npz')]:
        assert sha(run/(tag+'_head.npz')) == report['head_sha256'][tag]
        heads[tag] = dict(np.load(run/(tag+'_head.npz')))
        source = dict(np.load(pilot/original))
        assert set(source) == set(heads[tag])
        for key in source:
            np.testing.assert_array_equal(heads[tag][key], source[key])
    for key in ('mean', 'scale', 'target_mean', 'target_scale'):
        np.testing.assert_array_equal(heads['old'][key], heads['new'][key])
    assert len(report['rows']) == 4
    assert {(r['head'], r['objective']) for r in report['rows']} == {
        (h, o) for h in ('old', 'new') for o in ('decoded_teacher', 'physical_labels')}
    errors, observations, baseline, anchors, maximum = {}, {}, None, 0, 0.
    for row in report['rows']:
        path = run/row['file']
        assert sha(path) == row['sha256'] and row['frozen_tensors_unchanged']
        tag, obj = row['head'], row['objective']
        index = 3 if obj == 'decoded_teacher' else 4
        source_row = next(r for r in hr['rows'] if r['model_index'] == index)
        source_path = horizon/source_row['file']
        assert sha(source_path) == source_row['sha256'] == row['source_horizon_file_sha256']
        expected_weights = source_row['entry']['weights_sha256'] if tag == 'old' else next(
            r['weights_sha256'] for r in fit['continuations'] if r['objective'] == obj)
        assert row['weights_sha256'] == expected_weights
        data, original = dict(np.load(path)), dict(np.load(source_path))
        for key in ('true_pose', 'observed_tokens', 'seeds', 'reference_routes'):
            np.testing.assert_array_equal(data[key], original[key])
        assert data['true_pose'].shape == (4, 128, 5, 6)
        assert len(np.unique(data['seeds'])) == 128
        if baseline is None:
            baseline = data
        for key in ('true_pose', 'observed_tokens', 'seeds', 'reference_routes'):
            np.testing.assert_array_equal(data[key], baseline[key])
        for branch in ('free', 'teacher', 'observed'):
            tokens = data[branch+'_tokens']
            assert tokens.shape == (4, 128, 5, 192) and np.isfinite(tokens).all()
            prediction = decode(tokens, heads[tag])
            saved = data[branch+'_pose']
            assert saved.shape == (4, 128, 5, 6) and np.isfinite(saved).all()
            np.testing.assert_allclose(prediction, saved, rtol=1e-11, atol=1e-8)
            maximum = max(maximum, float(np.max(np.abs(prediction-saved))))
            error = np.square(prediction[..., 2:4] - data['true_pose'][..., 2:4]).sum(-1).mean(0)
            assert error.shape == (128, 5)
            errors[(tag, obj, branch)] = error
            if branch == 'observed':
                if tag in observations:
                    np.testing.assert_array_equal(error, observations[tag])
                observations[tag] = error
            if tag == 'old' and branch != 'observed':
                np.testing.assert_array_equal(tokens, original[branch+'_tokens'])
                anchors += 512
        np.testing.assert_array_equal(data['free_tokens'][..., 0, :], data['teacher_tokens'][..., 0, :])
    assert anchors == report['old_complete_trajectory_anchors'] == 2048
    draws = np.random.default_rng(1393001).integers(0, 128, size=(20000, 128))

    def estimate(values):
        assert values.shape == (128, 5) and np.isfinite(values).all()
        # Draws are shared across contrasts, preserving all within-goal pairing.
        samples = np.empty((20000, 5))
        for start in range(0, 20000, 500):
            samples[start:start+500] = values[draws[start:start+500]].mean(1)
        bounds = np.quantile(samples, [.025, .975], axis=0)
        return {'mean': values.mean(0).tolist(), 'ci95_low': bounds[0].tolist(), 'ci95_high': bounds[1].tolist()}

    absolute = {'/'.join(key): estimate(value) for key, value in errors.items()}
    gaps, changes = {}, {}
    for branch in ('free', 'teacher'):
        per_head = {}
        for tag in ('old', 'new'):
            per_head[tag] = errors[(tag, 'physical_labels', branch)] - errors[(tag, 'decoded_teacher', branch)]
            gaps[tag+'/'+branch] = estimate(per_head[tag])
        changes[branch] = estimate(per_head['new'] - per_head['old'])
    summary = {
        'status': 'PASS_FIXED_ACTION_ARRAYS_AND_PAIRED_GOAL_RECONSTRUCTION',
        'group': 0, 'goals': 128, 'reference_streams': 4, 'horizons': [5, 10, 15, 20, 25],
        'bootstrap_draws': 20000, 'bootstrap_seed': 1393001, 'interval': 'exploratory paired-goal percentile 95%',
        'absolute_position_mse': absolute, 'labels_minus_teacher': gaps,
        'new_minus_old_objective_gap': changes,
        'new_minus_old_observed_readout_mse': estimate(observations['new'] - observations['old']),
        'old_complete_trajectory_anchors': anchors, 'maximum_decode_discrepancy': maximum,
        'report_sha256': sha(run/'report.json'), 'source_sha256': sha(__file__),
        'evaluation_protocol_sha256': sha(args.protocol), 'pilot_protocol_sha256': sha(args.pilot_protocol),
        'source_horizon_report_sha256': sha(horizon/'report.json'),
        'scope': 'Conditional on one validation-gated Transformer; all reused goals and fixed streams retained. GRU failed the predeclared calibration gate and is reported separately. Calibration and readout geometry both change. This is not a new-goal or architecture-level confirmation.'}
    (run/'local_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'status': summary['status'], 'summary_sha256': sha(run/'local_summary.json'),
                      'endpoint_gap_changes': {k: v['mean'][-1] for k, v in changes.items()}}))


if __name__ == '__main__':
    main()
