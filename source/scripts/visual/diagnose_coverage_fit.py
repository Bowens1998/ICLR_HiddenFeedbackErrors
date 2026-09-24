"""Descriptive training fit and physical proximity; no tuning or causal claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha, metrics
from nonlinear_pose_cost import numpy_pose
from audit_pose_state_coverage import coverage


def main():
    p = argparse.ArgumentParser()
    for name in ['fits', 'evaluations', 'summary', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    assert summary['status'] == 'PASS_COMPLETE_MATRIX' and summary['models'] == 6
    rows = []
    training_targets = {}
    for i in range(6):
        fit = Path(a.fits)/f'job_{i}'
        ev = Path(a.evaluations)/f'job_{i}'
        er = json.loads((ev/'report.json').read_text())
        assert sha(ev/'report.json') == summary['provenance'][i]['report_sha256']
        assert sha(ev/'goals.npz') == er['goal_file_sha256']
        assert sha(fit/'report.json') == er['fit_report_sha256']
        assert sha(fit/'acceptance.json') == er['fit_acceptance_sha256']
        fr = json.loads((fit/'report.json').read_text())
        with np.load(ev/'goals.npz') as z:
            target = z['target']
        for arm in ['expert', 'broad']:
            row = next(v for v in fr['rows'] if v['arm'] == arm)
            for name, expected in row['files_sha256'].items():
                assert sha(fit/arm/name) == expected
            with np.load(fit/arm/'features.npz') as z:
                train_target, encoded = z['target'], z['encoded']
            assert train_target.shape == (512, 6)
            key = (i//2, arm)
            if key in training_targets:
                np.testing.assert_array_equal(train_target, training_targets[key])
            else:
                training_targets[key] = train_target
            head = dict(np.load(fit/arm/'weights.npz'))
            pred = numpy_pose(encoded, head)
            with np.load(fit/arm/'train_prediction.npz') as z:
                np.testing.assert_allclose(pred, z['prediction'], rtol=1e-10, atol=1e-9)
            mse = np.mean(((pred-train_target)/head['target_scale'])**2)
            np.testing.assert_allclose(mse, row['train_standardized_mse'], rtol=1e-10, atol=1e-12)
            train_metrics = {k: float(v.mean()) for k, v in metrics(pred, train_target).items()}
            test_metrics = {k: v['mean'] for k, v in summary['per_model'][i]['metrics'][arm].items()}
            rows.append(dict(index=i, replica=i//2, arm=arm,
                             train_metrics=train_metrics, goal_metrics=test_metrics,
                             goal_physical_proximity=coverage(target, train_target)))
    pooled = {arm: {split: {k: float(np.mean([r[split][k] for r in rows if r['arm'] == arm]))
                           for k in rows[0][split]}
                    for split in ['train_metrics', 'goal_metrics']} for arm in ['expert', 'broad']}
    result = dict(status='COMPLETE_DESCRIPTIVE_DIAGNOSTIC', rows=rows, pooled=pooled,
                  summary_sha256=sha(a.summary), source_sha256=sha(__file__),
                  dependencies={n: sha(Path(__file__).with_name(n)) for n in ['analyze_coverage_goals.py', 'audit_pose_state_coverage.py', 'nonlinear_pose_cost.py']},
                  scope='Post-outcome descriptive analysis; training accuracy is resubstitution, not generalization. Physical nearest-neighbor coverage uses privileged goal states and fixed task thresholds; not a deployable risk signal or proof of representation sufficiency. No head or hyperparameter selection. No intervals comparing distinct train/goal populations.')
    with Path(a.output).open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(pooled, indent=2))
    for r in rows[::2]:
        print(r['index'], r['arm'], r['goal_physical_proximity'])


if __name__ == '__main__':
    main()
