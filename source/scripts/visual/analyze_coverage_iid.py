"""Complete IID static holdout diagnostic; no training or manipulation gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nonlinear_pose_cost import numpy_pose


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(pred, target):
    position = np.square(pred[:, 2:4] - target[:, 2:4]).sum(-1)
    angle = np.arctan2(pred[:, 4], pred[:, 5]) - np.arctan2(target[:, 4], target[:, 5])
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return dict(position_mse=position, angle_mse=angle**2,
                position_precision=(position < 400).astype(float),
                angle_precision=(abs(angle) < np.pi/9).astype(float),
                joint_precision=((position < 400) & (abs(angle) < np.pi/9)).astype(float))


def summarize(values, indices):
    values = np.asarray(values)
    assert values.shape == (512,) and np.isfinite(values).all()
    ci = np.quantile(values[indices].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95=ci.tolist())


def main():
    p = argparse.ArgumentParser()
    for name in ['evaluations', 'fits', 'bank', 'plan', 'config', 'holdout-config', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    bank, fits, evaluations = map(Path, [a.bank, a.fits, a.evaluations])
    cfg = json.loads(Path(a.config).read_text())
    plan = json.loads(Path(a.plan).read_text())
    bm = json.loads((bank/'report.json').read_text())
    ba = json.loads((bank/'acceptance.json').read_text())
    assert ba['status'] == 'PASS' and not ba['engineering']
    assert ba['report_sha256'] == sha(bank/'report.json')
    assert ba['verifier_sha256'] == sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert bm['config_sha256'] == sha(a.holdout_config) and bm['replica'] == 0
    row = next(v for v in bm['rows'] if v['arm'] == 'broad')
    assert row['count'] == 512
    for name, expected in row['files_sha256'].items():
        assert sha(bank/'broad'/name) == expected
    with np.load(bank/'broad/poses.npz') as z:
        truth, source_index = z['target'], z['source_index']
    assert truth.shape == (512, 6) and len(set(source_index)) == 512
    # Shared resampling preserves goal pairing across all six fixed backbones.
    indices = np.random.default_rng(1254901).integers(0, 512, size=(10000, 512))
    arms = ['expert', 'broad', 'original_reference']
    all_values = {arm: [] for arm in arms}
    per_model, provenance = [], []
    for i in range(6):
        run, fit = evaluations/f'job_{i}', fits/f'job_{i}'
        r = json.loads((run/'report.json').read_text())
        fr = json.loads((fit/'report.json').read_text())
        fa = json.loads((fit/'acceptance.json').read_text())
        assert r['status'] == 'COMPLETE_FROZEN_IID_EVALUATION' and r['index'] == fr['index'] == i
        assert fa['status'] == 'PASS' and not fa['engineering'] and not fr['engineering']
        assert fa['report_sha256'] == r['fit_report_sha256'] == sha(fit/'report.json')
        assert r['fit_acceptance_sha256'] == sha(fit/'acceptance.json')
        assert fa['verifier_sha256'] == sha(Path(__file__).with_name('accept_coverage_readouts.py'))
        assert r['source_sha256'] == sha(Path(__file__).with_name('evaluate_coverage_iid.py'))
        assert r['plan_sha256'] == fr['plan_sha256'] == sha(a.plan)
        assert r['config_sha256'] == fr['config_sha256'] == sha(a.config)
        assert r['bank_report_sha256'] == sha(bank/'report.json')
        assert r['holdout_config_sha256'] == sha(a.holdout_config)
        assert r['bank_acceptance_sha256'] == sha(bank/'acceptance.json')
        entry = plan['models'][plan['routes'][i*8+2]['model_index']]
        assert fr['backbone_sha256'] == entry['weights_sha256']
        assert r['goal_file_sha256'] == sha(run/'goals.npz')
        with np.load(run/'goals.npz') as z:
            np.testing.assert_array_equal(z['source_index'], source_index)
            np.testing.assert_array_equal(z['target'], truth)
            tokens = z['tokens']
        assert tokens.shape == (512, 192) and np.isfinite(tokens).all()
        assert [v['arm'] for v in r['rows']] == arms
        vals = {}
        for row in r['rows']:
            arm = row['arm']
            if arm == 'original_reference':
                hp = Path(entry['goal_head']['path'])
                expected = entry['goal_head']['sha256']
            else:
                hp = fit/arm/'weights.npz'
                fitrow = next(v for v in fr['rows'] if v['arm'] == arm)
                assert fitrow['updates'] == 2000 and fitrow['labels'] == 512
                expected = fitrow['files_sha256']['weights.npz']
            assert sha(hp) == expected == row['head_sha256']
            assert row['file'] == f'{arm}.npz' and row['file_sha256'] == sha(run/row['file'])
            with np.load(run/row['file']) as z:
                pred = z['prediction']
            assert pred.shape == (512, 6) and np.isfinite(pred).all()
            np.testing.assert_allclose(pred, numpy_pose(tokens, dict(np.load(hp))), rtol=1e-10, atol=1e-9)
            vals[arm] = metrics(pred, truth)
            all_values[arm].append(vals[arm])
        contrasts = {k: summarize(vals['broad'][k]-vals['expert'][k], indices) for k in vals['expert']}
        per_model.append(dict(index=i, arm=entry['arm'],
                              metrics={arm: {k: summarize(v, indices) for k, v in vals[arm].items()} for arm in arms},
                              broad_minus_expert=contrasts))
        provenance.append(dict(index=i, report_sha256=sha(run/'report.json')))
    pooled = {arm: {k: np.stack([v[k] for v in all_values[arm]]).mean(0)
                    for k in all_values[arm][0]} for arm in arms}
    contrasts = {k: summarize(pooled['broad'][k]-pooled['expert'][k], indices) for k in pooled['expert']}
    result = dict(status='PASS_COMPLETE_IID_MATRIX', models=6, heads=18, images=512,
                  bootstrap=dict(draws=10000, seed=1254901, unit='shared IID static image', interval='percentile'),
                  pooled_metrics={arm: {k: summarize(v, indices) for k, v in pooled[arm].items()} for arm in arms},
                  pooled_broad_minus_expert=contrasts, per_model=per_model, provenance=provenance,
                  bank_report_sha256=sha(bank/'report.json'), holdout_config_sha256=sha(a.holdout_config), config_sha256=sha(a.config),
                  source_sha256=sha(__file__), scope='Conditional on six fixed backbones and three sampled training pools. Secondary intervals nominal, no multiplicity correction. Static IID localization only; no planning claim or reopening of the failed prospective manipulation gate. Original reference has a larger label budget. Saved predictions independently recomputed; goal encodings inherit native evaluation and are not independently re-encoded here.')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print('PASS_COMPLETE_IID_MATRIX')


if __name__ == '__main__':
    main()
