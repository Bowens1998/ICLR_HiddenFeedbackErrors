"""Reconstruct every prospective contrast from complete saved recipient arrays.

No model training, simulator replay, new sample selection or protocol changes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def bootstrap(values, seed):
    values = np.asarray(values, np.float64)
    assert values.ndim == 2 and np.isfinite(values).all()
    n = len(values)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty((20000, values.shape[1]), np.float64)
    for start in range(0, 20000, 250):
        ids = rng.integers(0, n, size=(250, n), dtype=np.int64)
        counts = np.array([np.bincount(row, minlength=n) for row in ids])
        out[start:start + 250] = counts @ values / n
    return out


def check(actual, expected, atol=1e-9, rtol=1e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('PROSPECTIVE_VERIFICATION.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output path to preserve earlier receipts.')
    start = time.monotonic()
    manifest = read('MANIFEST.json')
    for name, record in manifest['files'].items():
        if name.startswith('archives/'):
            continue
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record['sha256'], name

    d = ROOT / 'prospective/data'
    first = read('prospective/data/s1_report.json')
    spec = read('prospective/protocols/DESIGN.lock.json')['statistics']
    rows = first['primary'] + first['secondary']
    with np.load(d / 's1_goal_contrasts.npz', allow_pickle=False) as z:
        assert set(z.files) == {r['id'] for r in rows}
        v1 = np.column_stack([z[r['id']] for r in rows])
    assert v1.shape == (256, 8)
    boot1 = bootstrap(v1, spec['bootstrap_seed'])
    for i, row in enumerate(rows):
        check(v1[:, i].mean(), row['estimate'])
        check(np.quantile(boot1[:, i], [.00625, .99375]), row['ci'])

    second = read('prospective/data/s2_statistics.json')
    with np.load(d / 's2_recomputed.npz', allow_pickle=False) as z:
        v2 = z['statistics__paired_recipient_scores'][:, None]
        assert v2.shape == (512, 1)
        boot2 = bootstrap(v2, second['bootstrap_seed'])[:, 0]
        check(boot2, z['statistics__bootstrap_means'], rtol=1e-10)
        check(v2.mean(), second['paired_improvement'], rtol=1e-10)
        check(np.quantile(boot2, [.025, .975]), second['ci_95'], rtol=1e-10)
        check(z['statistics__stratum_improvements'], second['stratum_improvements'], rtol=1e-10)
        # Verify fixed regression prediction arithmetic from supplied bases and coefficients.
        for branch in ['baseline', 'augmented']:
            scale = z['calibration__target_scale']
            pred = (z['predictions__test_basis_' + branch] @ z['calibration__' + branch + '_slopes']
                    + z['calibration__' + branch + '_intercepts']) * scale
            check(pred, z['predictions__' + branch + '_prediction'], rtol=1e-10)

    third = read('prospective/data/s3_report.json')
    with np.load(d / 's3_statistics.npz', allow_pickle=False) as z:
        selected = z['selected_costs']
        assert selected.shape == (512, 3, 2, 2, 4)
        v3 = np.column_stack([
            (selected[:, :, 1, :, 1] - selected[:, :, 1, :, b]).mean((1, 2))
            for b in [0, 2]])
        check(v3, z['goal_contrasts'])
        check(v3, z['independent_goal_contrasts'])
        boot3 = bootstrap(v3, third['bootstrap_seed'])
        check(boot3, z['bootstrap_means'])
        check(boot3, z['independent_bootstrap_means'])
        for i, row in enumerate(third['primary']):
            check(v3[:, i].mean(), row['mean'])
            check(np.quantile(boot3[:, i], [.0125, .9875]), [row['lower'], row['upper']])
        check(selected.mean(0), third['descriptive_selected_cost_means'])
        np.testing.assert_array_equal(selected, z['lowest_index_selected_costs'])
    report = dict(status='PASS_COMPLETE_PROSPECTIVE_SAVED_CONTRAST_RECONSTRUCTION',
        s1_recipients=256, s1_contrasts=8, s2_recipients=512, s2_contrasts=1,
        s3_recipients=512, s3_contrasts=2, bootstrap_draws_per_test=20000,
        all_negative_and_unresolved_results_retained=True,
        scope='Complete saved recipient contrasts, fixed regression prediction arithmetic, selected-cost contrast arithmetic and fixed bootstrap statistics. Original full readout/selection/model/simulator checks are retained receipts; not rerun by this command.',
        wall_seconds=time.monotonic() - start)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
